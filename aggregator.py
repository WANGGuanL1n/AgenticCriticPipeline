"""
Aggregator — 聚合 dimension critics + artist critics 的输出
核心机制：
  1. Per-channel confidence-weighted average（维度层 + 艺术家层分治）
  2. Disagreement penalty — 艺术家之间分歧大时降权，防止 reward hacking
  3. Directive merger — 多个 critic 的相似建议合并并 boost priority
  4. Should-stop 判断
"""
from .gap_types import GapChannel, GapScore, RepairDirective, RepairAction, AggregateGap
from .soul_skills.artists import ArtistProfile
import math
from statistics import stdev
from collections import defaultdict


def aggregate(
    dimension_scores: dict[str, GapScore],
    artist_scores: dict[str, dict[str, GapScore]],  # artist_name → {channel_name: GapScore}
    artist_profiles: dict[str, ArtistProfile],
    artist_on_domain: dict[str, float],
) -> AggregateGap:
    """
    Aggregate dimension and artist critic scores into a single gap report.
    """
    # ----- 1. Aggregate dimension critics -----
    dim_by_channel: dict[GapChannel, list[GapScore]] = defaultdict(list)
    for score in dimension_scores.values():
        dim_by_channel[score.channel].append(score)

    dim_aggregated: dict[GapChannel, GapScore] = {}
    for channel, scores in dim_by_channel.items():
        if not scores:
            continue
        weighted_sum = sum(s.score * s.confidence for s in scores)
        confidence_sum = sum(s.confidence for s in scores)
        avg_score = weighted_sum / confidence_sum if confidence_sum > 0 else 6
        avg_conf = confidence_sum / len(scores) if scores else 0.5
        rationales = " | ".join(s.rationale for s in scores)
        dim_aggregated[channel] = GapScore(
            channel=channel, score=avg_score, confidence=avg_conf / 3,  # dimension 降权
            rationale=rationales, critic_name="dim.aggregated"
        )

    # ----- 2. Aggregate artist critics with disagreement penalty -----
    artist_by_channel: dict[GapChannel, list[dict]] = defaultdict(list)
    for artist_name, channel_dict in artist_scores.items():
        profile = artist_profiles.get(artist_name)
        on_domain = artist_on_domain.get(artist_name, 0.3)
        for ch_name, gs in channel_dict.items():
            try:
                ch = GapChannel[ch_name]
            except KeyError:
                continue
            # Confidence capped by on_domain
            effective_conf = min(gs.confidence, max(0.3, on_domain))
            # Perceptual bias boost: if artist's primary channel, keep confidence high
            if profile and ch in profile.perceptual_bias[:2]:
                effective_conf = max(effective_conf, gs.confidence * 0.9)
            else:
                effective_conf = effective_conf * 0.6  # off-bias penalty
            artist_by_channel[ch].append({
                "score": gs.score,
                "confidence": effective_conf,
                "rationale": gs.rationale,
                "artist": artist_name,
            })

    artist_aggregated: dict[GapChannel, GapScore] = {}
    consensus_scores = []
    for channel, entries in artist_by_channel.items():
        if not entries:
            continue
        scores = [e["score"] for e in entries]
        disagreements = 0.0
        if len(scores) > 1:
            try:
                disagreement = min(1.0, stdev(scores) / 5.0)
            except:
                disagreement = 0.5
        consensus_scores.append(1 - disagreement)
        penalty = 1 - disagreement * 0.5

        weighted_sum = sum(e["score"] * e["confidence"] * penalty for e in entries)
        confidence_sum = sum(e["confidence"] * penalty for e in entries)
        avg_score = weighted_sum / confidence_sum if confidence_sum > 0 else 6
        avg_conf = min(0.85, confidence_sum / len(entries) * penalty if entries else 0.5)
        rationales = " | ".join(f"[{e['artist']}] {e['rationale'][:80]}" for e in entries)
        artist_aggregated[channel] = GapScore(
            channel=channel, score=avg_score, confidence=avg_conf,
            rationale=rationales, critic_name="artist.aggregated"
        )

    artist_consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.5

    # ----- 3. Merge dimension + artist -----
    merged: dict[GapChannel, GapScore] = {}
    for channel in GapChannel:
        d_score = dim_aggregated.get(channel)
        a_score = artist_aggregated.get(channel)
        if d_score and a_score:
            w_sum = d_score.score * d_score.confidence + a_score.score * a_score.confidence
            c_sum = d_score.confidence + a_score.confidence
            merged[channel] = GapScore(
                channel=channel,
                score=w_sum / c_sum if c_sum > 0 else 6,
                confidence=(d_score.confidence + a_score.confidence) / 2,
                rationale=f"DIM: {d_score.rationale[:60]} | ART: {a_score.rationale[:60]}",
                critic_name="merged",
            )
        elif d_score:
            merged[channel] = d_score
        elif a_score:
            merged[channel] = a_score

    # ----- 4. Overall alignment -----
    all_confs = [gs.confidence for gs in merged.values()]
    all_scores = [gs.score for gs in merged.values()]
    if all_confs:
        weighted = sum(s * c for s, c in zip(all_scores, all_confs))
        total_conf = sum(all_confs)
        overall_alignment = max(0.0, 1.0 - weighted / (total_conf * 10))
        overall_confidence = sum(all_confs) / len(all_confs)
    else:
        overall_alignment = 0.5
        overall_confidence = 0.3

    # ----- 5. Repair directives -----
    directives = _build_directives(merged, artist_scores, artist_by_channel)

    # ----- 6. Should stop -----
    should_stop = overall_alignment > 0.85 or (overall_alignment > 0.7 and max(all_scores) < 5 if all_scores else False)

    return AggregateGap(
        channel_scores=merged,
        overall_alignment=overall_alignment,
        overall_confidence=overall_confidence,
        repair_directives=directives,
        should_stop=should_stop,
        artist_consensus=artist_consensus,
        dimension_critic_summary=_summarize(dim_aggregated),
        artist_panel_summary=_summarize(artist_aggregated),
    )


def _build_directives(merged, artist_scores, artist_by_channel) -> list[RepairDirective]:
    """Build prioritized repair directives from merged scores + artist suggestions"""
    directives = []
    channel_action_map = {
        GapChannel.STRUCTURAL: RepairAction.PROMPT_TUNE,
        GapChannel.STYLISTIC: RepairAction.PROMPT_TUNE,
        GapChannel.SEMANTIC: RepairAction.REGENERATE,
        GapChannel.QUALITY: RepairAction.INPAINT,
    }

    for channel, gs in merged.items():
        if gs.score >= 6:
            action = channel_action_map.get(channel, RepairAction.PROMPT_TUNE)
            priority = (gs.score / 10) * 0.9 + gs.confidence * 0.1
            # Count supporters
            n_supporters = 1  # from dimension
            if artist_by_channel.get(channel):
                n_supporters += sum(1 for e in artist_by_channel[channel] if e["score"] >= 6)
            directives.append(RepairDirective(
                action=action,
                channel=channel,
                priority=priority,
                description=f"Fix {channel.name.lower()} gap (score={gs.score:.1f}, conf={gs.confidence:.2f})",
                n_supporters=n_supporters,
            ))

    directives.sort(key=lambda d: -d.priority)
    return directives[:5]


def _summarize(scores_dict: dict) -> str:
    parts = []
    for ch, gs in scores_dict.items():
        parts.append(f"{ch.name}: {gs.score:.1f} (conf={gs.confidence:.2f})")
    return " | ".join(parts)
