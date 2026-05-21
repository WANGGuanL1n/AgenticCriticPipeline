"""
Router — 根据任务类型 + target 风格选择合适的专家面板
策略：
  1. Dimension critics 始终全选（always-on 基础层）
  2. Artist critics 根据 target style descriptor 做 text embedding 匹配
  3. 任务类型覆盖：logo → 强制 Mondrian, portrait → 强制 Caravaggio
  4. 历史感知：上一轮的弱通道会触发对应专家补强
"""
from .gap_types import RoutingDecision, GapChannel
from .soul_skills.artists import ARTIST_PROFILES, ArtistProfile
from .dimension_critics import DIMENSION_CRITICS
from typing import Optional


# Simple keyword-based matching (replace with actual embedding in production)
def _keyword_match(descriptor: str, keywords: list[str]) -> float:
    """Simple TF-IDF-like match between descriptor and expertise keywords"""
    desc_lower = descriptor.lower()
    matches = sum(1 for kw in keywords if kw in desc_lower)
    if not keywords:
        return 0.0
    return min(1.0, matches / max(3, len(keywords) * 0.3))


def route(task_type: str,
          target_descriptor: str = "",
          prior_weak_channels: Optional[list[GapChannel]] = None,
          history_round: int = 0) -> RoutingDecision:
    """
    Args:
        task_type: 'portrait', 'landscape', 'logo', 'poster', 'illustration', 'scifi', 'abstract'
        target_descriptor: VLM-generated style description of target image
        prior_weak_channels: from previous round's gap analysis
        history_round: which iteration we're on
    """
    selected_dimension = list(DIMENSION_CRITICS.keys())

    # Score each artist for on_domain match
    artist_scores = {}
    reasoning_lines = []

    for key, profile in ARTIST_PROFILES.items():
        # Domain match from target descriptor
        kw_score = _keyword_match(target_descriptor, profile.on_domain_keywords)

        # Task type override
        task_bonus = 0.0
        if profile.name == "Piet Mondrian" and task_type in ("logo", "poster", "graphic", "abstract"):
            task_bonus = 0.3
            reasoning_lines.append(f"{profile.name}: +0.3 task bonus ({task_type})")
        if profile.name == "Caravaggio" and task_type in ("portrait", "figure", "anatomy"):
            task_bonus = 0.3
            reasoning_lines.append(f"{profile.name}: +0.3 task bonus ({task_type})")
        if profile.name == "Claude Monet" and task_type in ("landscape", "nature", "garden"):
            task_bonus = 0.2
        if profile.name == "Moebius (Jean Giraud)" and task_type in ("illustration", "comic", "graphic novel"):
            task_bonus = 0.25
        if profile.name == "Katsushika Hokusai" and task_type in ("minimalist", "japanese", "pattern"):
            task_bonus = 0.25
        if profile.name == "Greg Rutkowski (school)" and task_type in ("scifi", "fantasy", "concept art"):
            task_bonus = 0.3

        # History-aware reinforcement
        history_bonus = 0.0
        if prior_weak_channels and history_round > 0:
            for ch in prior_weak_channels:
                if ch in profile.perceptual_bias:
                    history_bonus += 0.15
            if history_bonus > 0:
                reasoning_lines.append(f"{profile.name}: +{history_bonus:.2f} history (weak {[ch.name for ch in prior_weak_channels]})")

        score = min(1.0, kw_score + task_bonus + history_bonus)
        artist_scores[key] = score

    # Select top-K (min 2, max 4)
    sorted_artists = sorted(artist_scores.items(), key=lambda x: -x[1])
    min_artists = min(2 if task_type in ("logo", "abstract", "minimalist") else 3, len(sorted_artists))
    threshold = max(0.2, sorted_artists[min_artists - 1][1]) if len(sorted_artists) >= min_artists else 0.2
    selected_artist_keys = [k for k, s in sorted_artists if s >= threshold][:4]

    # Ensure minimum selection
    if len(selected_artist_keys) < 2:
        selected_artist_keys = [k for k, _ in sorted_artists[:2]]

    reasoning = " | ".join(reasoning_lines) if reasoning_lines else f"Top match: {', '.join(selected_artist_keys[:3])}"

    return RoutingDecision(
        selected_dimension_critics=selected_dimension,
        selected_artist_critics=selected_artist_keys,
        artist_on_domain={k: artist_scores[k] for k in selected_artist_keys},
        reasoning=reasoning,
    )
