"""
gap_critique — src→target Gap Critique Pipeline

Agentic 生图的 Critic 组件：对比 src(SenseNova-U1 生成) 和 target(gpt-image-2 生成)，
通过 dimension critics + soul-skill 艺术家面板评估差距，输出结构化修复指令。

Usage:
    from gap_critique import GapPipeline, ImagePairGenerator
    pipeline = GapPipeline(task_type="portrait")
    gen = ImagePairGenerator(openai_api_key="...")
    pair = gen.generate("a warrior in a garden", style="baroque")
    result = pipeline.evaluate(pair.src_path, pair.target_path)
"""
from .pipeline import GapPipeline, PipelineState
from .generators import ImagePairGenerator, ImagePair, GPTImage2Generator, SenseNovaU1Generator
from .gap_types import GapChannel, GapScore, AggregateGap, RepairDirective, RepairAction
from .router import route
from .aggregator import aggregate

__all__ = [
    "GapPipeline", "PipelineState",
    "ImagePairGenerator", "ImagePair", "GPTImage2Generator", "SenseNovaU1Generator",
    "GapChannel", "GapScore", "AggregateGap", "RepairDirective", "RepairAction",
    "route", "aggregate",
]
