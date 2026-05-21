"""
Pipeline — 主编排器：src→target→critique→repair 完整闭环
用法：
    pipeline = GapPipeline(vlm_api_key="sk-...", vlm_base_url="http://...", task_type="portrait")
    result = pipeline.evaluate(src_path, target_path)
    print(result["gap"].overall_alignment)
"""
from .gap_types import GapChannel, GapScore, RepairDirective, RepairAction, AggregateGap, RoutingDecision, CritiqueTrajectory
from .dimension_critics import DIMENSION_CRITICS, snap_to_anchor
from .soul_skills.artists import ARTIST_PROFILES, ArtistProfile, _make_artist_critique_prompt
from .router import route
from .aggregator import aggregate
from .backends import VLMBackend, DetectorBackend, OCREngine, IQAEngine, PaletteExtractor, OpenAICompatVLM
from dataclasses import dataclass, field
import json, traceback


@dataclass
class PipelineState:
    """Per-trajectory state, survives across multi-turn evaluate() calls"""
    round: int = 0
    history_alignment: list[float] = field(default_factory=list)
    history_gaps: list[AggregateGap] = field(default_factory=list)
    stale_count: int = 0  # consecutive rounds without significant improvement

    def update(self, gap: AggregateGap):
        self.round += 1
        self.history_gaps.append(gap)
        self.history_alignment.append(gap.overall_alignment)
        if len(self.history_alignment) >= 2:
            improvement = gap.overall_alignment - self.history_alignment[-2]
            if improvement < 0.02:
                self.stale_count += 1
            else:
                self.stale_count = 0

    def get_prior_weak_channels(self) -> list[GapChannel]:
        """Return channels that were weak in the last round"""
        if not self.history_gaps:
            return []
        last = self.history_gaps[-1]
        return [ch for ch, gs in last.channel_scores.items() if gs.score >= 6]


class GapPipeline:
    """Main pipeline for src→target gap evaluation with soul-skill artist panel"""

    def __init__(self,
                 vlm_api_key: str = "",
                 vlm_base_url: str = "",
                 vlm_model: str = "gemini-3-flash-preview",
                 vlm: VLMBackend = None,
                 detector: DetectorBackend = None,
                 ocr: OCREngine = None,
                 iqa: IQAEngine = None,
                 palette: PaletteExtractor = None,
                 task_type: str = "portrait"):
        self.task_type = task_type

        # Real VLM if credentials provided, else explicit override, else mock
        if vlm_api_key and vlm_base_url:
            self._vlm = OpenAICompatVLM(vlm_api_key, vlm_base_url, vlm_model)
        elif vlm:
            self._vlm = vlm
        else:
            from .backends import MockVLM
            self._vlm = MockVLM()

        self.backends = {"vlm": self._vlm, "detector": detector, "ocr": ocr, "iqa": iqa, "palette": palette}
        from .backends import MockDetector, MockOCR, MockIQA, MockPalette
        self.backends.setdefault("detector", MockDetector())
        self.backends.setdefault("ocr", MockOCR())
        self.backends.setdefault("iqa", MockIQA())
        self.backends.setdefault("palette", MockPalette())

    def evaluate(self, src_path: str, target_path: str,
                 target_descriptor: str = "",
                 state: PipelineState = None) -> dict:
        """
        Evaluate src→target gap. Returns dict with full critique result.
        Call this each turn of the agentic refinement loop.
        """
        if state is None:
            state = PipelineState()

        # Step 1: Route critics
        prior_weak = state.get_prior_weak_channels()
        routing = route(
            task_type=self.task_type,
            target_descriptor=target_descriptor,
            prior_weak_channels=prior_weak,
            history_round=state.round,
        )

        # Step 2: Run dimension critics (always-on subset)
        dim_scores = {}
        critic_kwargs_map = {
            "compositional": ["vlm", "detector"],
            "stylistic": ["vlm", "palette"],
            "semantic": ["vlm", "detector"],
            "quality": ["vlm", "iqa"],
            "palette_diff": ["palette"],
            "text_fidelity": ["ocr"],
            "anatomy": ["vlm"],
        }
        for name in routing.selected_dimension_critics:
            fn = DIMENSION_CRITICS.get(name)
            if fn:
                try:
                    kwargs = {k: self.backends[k] for k in critic_kwargs_map.get(name, []) if k in self.backends}
                    dim_scores[name] = fn(src_path, target_path, **kwargs)
                except Exception as e:
                    dim_scores[name] = GapScore(
                        channel=GapChannel.QUALITY, score=6, confidence=0.1,
                        rationale=f"Error: {e}", critic_name=name
                    )

        # Step 3: Run artist critics via VLM (real or mock fallback)
        artist_channel_scores = {}
        for artist_key in routing.selected_artist_critics:
            profile = ARTIST_PROFILES.get(artist_key)
            if not profile:
                continue
            on_domain = routing.artist_on_domain.get(artist_key, 0.3)

            try:
                prompt = _make_artist_critique_prompt(profile, src_path, target_path)
                raw = self._vlm.score(prompt, [src_path, target_path])
                channel_dict = _parse_artist_response(raw, artist_key, profile, on_domain)
            except Exception as e:
                # Fallback to mock on any error (network, parse, etc.)
                print(f"  [WARN] artist {artist_key} VLM call failed: {e} — falling back to mock")
                channel_dict = _mock_artist_scores(profile, on_domain)

            artist_channel_scores[artist_key] = channel_dict

        # Step 4: Aggregate
        gap = aggregate(
            dimension_scores=dim_scores,
            artist_scores=artist_channel_scores,
            artist_profiles=ARTIST_PROFILES,
            artist_on_domain=routing.artist_on_domain,
        )

        # Step 5: Update state and check stall
        state.update(gap)
        should_stop = gap.should_stop or state.stale_count >= 5

        # Step 6: Build full trajectory (all raw critic data preserved)
        trajectory = CritiqueTrajectory(
            routing=routing,
            dimension_scores=dim_scores,
            artist_scores=artist_channel_scores,
            aggregated=gap,
            round=state.round,
            stale_count=state.stale_count,
            should_stop=should_stop,
            src_path=src_path,
            target_path=target_path,
            task_type=self.task_type,
            target_descriptor=target_descriptor,
            prior_weak_channels=[ch.name for ch in prior_weak],
        )

        # Step 7: Format output for Planner
        planner_observation = _format_for_planner(gap, routing)

        return {
            "gap": gap,
            "routing": routing,
            "should_stop": should_stop,
            "planner_observation": planner_observation,
            "trajectory": trajectory,
            "round": state.round,
            "stale_count": state.stale_count,
            "reward": gap.overall_alignment,  # scalar reward for GRPO
        }


# ─── Artist VLM parsing ───

def _parse_artist_response(raw: dict, artist_key: str, profile: ArtistProfile, on_domain: float) -> dict:
    """Parse VLM JSON response into channel GapScores"""
    channels_raw = raw.get("channels", {})
    channel_dict = {}
    channel_map = {
        "STRUCTURAL": GapChannel.STRUCTURAL,
        "STYLISTIC": GapChannel.STYLISTIC,
        "SEMANTIC": GapChannel.SEMANTIC,
        "QUALITY": GapChannel.QUALITY,
    }
    for ch_name, ch_data in channels_raw.items():
        channel = channel_map.get(ch_name.upper())
        if not channel:
            continue
        score = float(ch_data.get("score", 6))
        rationale = ch_data.get("rationale", "")
        confidence = 0.4 + on_domain * 0.3
        channel_dict[channel.name] = GapScore(
            channel=channel,
            score=snap_to_anchor(score),
            confidence=min(1.0, confidence),
            rationale=rationale,
            critic_name=f"artist.{artist_key}",
        )
    # Fill any missing channels with defaults
    for channel in GapChannel:
        if channel.name not in channel_dict:
            channel_dict[channel.name] = GapScore(
                channel=channel, score=6, confidence=0.3,
                rationale="No evaluation provided",
                critic_name=f"artist.{artist_key}",
            )
    return channel_dict


# ─── Mock fallback (preserved from original) ───

def _mock_artist_scores(profile: ArtistProfile, on_domain: float) -> dict:
    """Generate mock artist scores when VLM is unavailable"""
    import hashlib
    channel_dict = {}
    axioms = profile.axioms
    voice_hints = {
        GapChannel.STRUCTURAL: ("composition", "layout", "arrangement"),
        GapChannel.STYLISTIC: ("palette", "light", "brushwork"),
        GapChannel.SEMANTIC: ("subject", "entities", "what is depicted"),
        GapChannel.QUALITY: ("finish", "rendering", "execution"),
    }
    for channel in GapChannel:
        seed = int(hashlib.md5(f"{profile.name}:{channel.name}".encode()).hexdigest()[:8], 16)
        base_score = seed % 10
        if channel in profile.perceptual_bias:
            base_score = max(0, base_score - 2)
        if on_domain < 0.4:
            base_score = min(10, base_score + 2)
        anchors = [0, 2, 4, 6, 8, 10]
        score = min(anchors, key=lambda a: abs(a - base_score))
        axiom_idx = hash(f"{profile.name}:{channel.name}") % len(axioms)
        hint = voice_hints.get(channel, ("aspect",))
        rationale = f'The {hint[0]} does not satisfy: "{axioms[axiom_idx]}"'
        channel_dict[channel.name] = GapScore(
            channel=channel,
            score=score,
            confidence=0.4 + on_domain * 0.3,
            rationale=rationale,
            critic_name=f"artist.{profile.name.lower().split()[0]}",
        )
    return channel_dict


def _format_for_planner(gap: AggregateGap, routing: RoutingDecision) -> str:
    """Format aggregate gap as Planner-readable observation text (injected into context with loss_mask=0)"""
    lines = [f"[Critique Round Summary — overall alignment: {gap.overall_alignment:.2f}]"]
    lines.append(f"Artist panel: {', '.join(routing.selected_artist_critics)}")
    lines.append(f"Consensus: {gap.artist_consensus:.2f}")

    for ch, gs in gap.channel_scores.items():
        lines.append(f"  {ch.name}: score={gs.score:.1f} conf={gs.confidence:.2f} | {gs.rationale[:80]}")

    if gap.repair_directives:
        lines.append("Top repair directives:")
        for d in gap.repair_directives[:3]:
            lines.append(f"  [{d.action.value}] prio={d.priority:.2f} [{d.channel.name}] {d.description[:100]}")

    if gap.should_stop:
        lines.append("VERDICT: STOP — alignment sufficient")

    return "\n".join(lines)
