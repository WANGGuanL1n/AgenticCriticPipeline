"""
Pipeline — 主编排器：src→target→critique→repair 完整闭环
用法：
    pipeline = GapPipeline(vlm=my_vlm, detector=my_detector, ...)
    result = pipeline.evaluate(src_path, target_path, task_type="portrait")
    print(result.overall_alignment)
"""
from .gap_types import GapChannel, GapScore, RepairDirective, RepairAction, AggregateGap, RoutingDecision, CritiqueTrajectory
from .dimension_critics import DIMENSION_CRITICS
from .soul_skills.artists import ARTIST_PROFILES, ArtistProfile
from .router import route
from .aggregator import aggregate
from .backends import VLMBackend, DetectorBackend, OCREngine, IQAEngine, PaletteExtractor
from dataclasses import dataclass, field


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
                 vlm: VLMBackend = None,
                 detector: DetectorBackend = None,
                 ocr: OCREngine = None,
                 iqa: IQAEngine = None,
                 palette: PaletteExtractor = None,
                 task_type: str = "portrait"):
        self.backends = {"vlm": vlm, "detector": detector, "ocr": ocr, "iqa": iqa, "palette": palette}
        self.task_type = task_type
        # Default mock backends if none provided
        from .backends import MockVLM, MockDetector, MockOCR, MockIQA, MockPalette
        self.backends.setdefault("vlm", MockVLM())
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
        # Map each critic to the backend kwargs it accepts
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

        # Step 3: Run artist critics (routed subset)
        artist_channel_scores = {}
        for artist_key in routing.selected_artist_critics:
            profile = ARTIST_PROFILES.get(artist_key)
            if not profile:
                continue
            # Each artist evaluates all 4 channels
            channel_dict = {}
            on_domain = routing.artist_on_domain.get(artist_key, 0.3)
            for channel in GapChannel:
                # Simulate artist critique — in production, call VLM with _make_artist_critique_prompt
                # For now, generate a plausible mock based on the profile
                score = _mock_artist_channel_score(profile, channel, on_domain)
                channel_dict[channel.name] = GapScore(
                    channel=channel,
                    score=score,
                    confidence=0.4 + on_domain * 0.3,  # 0.4-0.7 range
                    rationale=_make_mock_rationale(profile, channel),
                    critic_name=f"artist.{artist_key}",
                )
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


def _mock_artist_channel_score(profile: ArtistProfile, channel: GapChannel, on_domain: float) -> int:
    """Generate a plausible mock score based on artist's perceptual bias"""
    import hashlib
    seed = int(hashlib.md5(f"{profile.name}:{channel.name}".encode()).hexdigest()[:8], 16)
    base_score = (seed % 10)
    # Artists score lower (less gap) on their primary channels
    if channel in profile.perceptual_bias:
        base_score = max(0, base_score - 2)
    # Off-domain artists have higher variance
    if on_domain < 0.4:
        base_score = min(10, base_score + 2)
    # Snap to anchor
    anchors = [0, 2, 4, 6, 8, 10]
    return min(anchors, key=lambda a: abs(a - base_score))


def _make_mock_rationale(profile: ArtistProfile, channel: GapChannel) -> str:
    """Generate an artist-voiced mock rationale"""
    axioms = profile.axioms
    axiom_idx = hash(f"{profile.name}:{channel.name}") % len(axioms)
    axiom = axioms[axiom_idx]
    voice_hints = {
        GapChannel.STRUCTURAL: ("composition", "layout", "arrangement"),
        GapChannel.STYLISTIC: ("palette", "light", "brushwork"),
        GapChannel.SEMANTIC: ("subject", "entities", "what is depicted"),
        GapChannel.QUALITY: ("finish", "rendering", "execution"),
    }
    hint = voice_hints.get(channel, ("aspect",))
    return f'The {hint[0]} does not satisfy: "{axiom}"'


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
