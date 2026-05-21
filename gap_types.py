"""
gap_critique — src→target 差距评估 pipeline
设计原则：
  1. DIMENSION CRITICS（程序化 + VLM）负责"客观"维度
  2. ARTIST CRITICS（Soul Skills）负责"主观/美学"维度
  3. ROUTER 根据任务类型 + target 风格选择专家
  4. AGGREGATOR 聚合两层输出，disagreement penalty 防 reward hacking
  5. PIPELINE 编排完整 src→target→critique→repair loop
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, Optional

class GapChannel(Enum):
    STRUCTURAL = auto()   # 构图/布局
    STYLISTIC = auto()    # 色彩/笔触/光线
    SEMANTIC = auto()     # 内容/画了什么
    QUALITY = auto()      # 完成度/技术质量

class RepairAction(Enum):
    STOP = "STOP"
    INPAINT = "INPAINT"
    PROMPT_TUNE = "PROMPT_TUNE"
    REGENERATE = "REGENERATE"

@dataclass
class GapScore:
    """单个维度上的 src→target 差距评估"""
    channel: GapChannel
    score: float          # 0=完美对齐, 10=完全偏离 | snap to {0,2,4,6,8,10}
    confidence: float     # 0-1
    rationale: str
    critic_name: str

@dataclass
class RepairDirective:
    """一条具体修改指令"""
    action: RepairAction
    channel: GapChannel
    priority: float       # 0-1, 越高越紧急
    description: str
    n_supporters: int = 1  # 几个 critic 支持该指令

@dataclass
class AggregateGap:
    """聚合后的差距报告"""
    channel_scores: dict[GapChannel, GapScore]
    overall_alignment: float       # 0-1, 1=完全对齐
    overall_confidence: float
    repair_directives: list[RepairDirective]
    should_stop: bool
    artist_consensus: float        # artist panel 内部一致度
    dimension_critic_summary: str
    artist_panel_summary: str

@dataclass
class RoutingDecision:
    selected_dimension_critics: list[str]
    selected_artist_critics: list[str]
    artist_on_domain: dict[str, float]  # artist → 0-1 领域匹配度
    reasoning: str


@dataclass
class CritiqueTrajectory:
    """Complete raw trajectory of a single critique run — saved for GRPO/analysis"""
    # Routing
    routing: RoutingDecision
    # Raw dimension critic outputs (name → GapScore)
    dimension_scores: dict[str, GapScore]
    # Raw artist critic outputs (artist_key → channel_name → GapScore)
    artist_scores: dict[str, dict[str, GapScore]]
    # Aggregated result
    aggregated: AggregateGap
    # Round number (for multi-turn refinement)
    round: int = 0
    stale_count: int = 0
    should_stop: bool = False
    # Metadata
    src_path: str = ""
    target_path: str = ""
    task_type: str = ""
    target_descriptor: str = ""
    prior_weak_channels: list[str] = field(default_factory=list)
    # Raw VLM call log — full API response metadata for each critic invocation
    raw_vlm_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict"""
        def gap_score_dict(gs: GapScore) -> dict:
            return {
                "channel": gs.channel.name,
                "score": gs.score,
                "confidence": gs.confidence,
                "rationale": gs.rationale,
                "critic_name": gs.critic_name,
            }

        def aggregate_dict(ag: AggregateGap) -> dict:
            return {
                "channel_scores": {ch.name: gap_score_dict(gs) for ch, gs in ag.channel_scores.items()},
                "overall_alignment": ag.overall_alignment,
                "overall_confidence": ag.overall_confidence,
                "repair_directives": [
                    {"action": d.action.value, "channel": d.channel.name,
                     "priority": d.priority, "description": d.description,
                     "n_supporters": d.n_supporters}
                    for d in ag.repair_directives
                ],
                "should_stop": ag.should_stop,
                "artist_consensus": ag.artist_consensus,
                "dimension_critic_summary": ag.dimension_critic_summary,
                "artist_panel_summary": ag.artist_panel_summary,
            }

        return {
            "round": self.round,
            "stale_count": self.stale_count,
            "should_stop": self.should_stop,
            "task_type": self.task_type,
            "target_descriptor": self.target_descriptor,
            "prior_weak_channels": self.prior_weak_channels,
            "routing": {
                "selected_dimension_critics": self.routing.selected_dimension_critics,
                "selected_artist_critics": self.routing.selected_artist_critics,
                "artist_on_domain": self.routing.artist_on_domain,
                "reasoning": self.routing.reasoning,
            },
            "dimension_scores": {name: gap_score_dict(gs) for name, gs in self.dimension_scores.items()},
            "artist_scores": {
                artist: {ch: gap_score_dict(gs) for ch, gs in channels.items()}
                for artist, channels in self.artist_scores.items()
            },
            "aggregated": aggregate_dict(self.aggregated),
            "raw_vlm_calls": self.raw_vlm_calls,
        }
