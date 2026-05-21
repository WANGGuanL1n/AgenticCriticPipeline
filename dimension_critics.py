"""
Dimension Critics — 程序化 + VLM 混合，负责"客观"维度的 src→target 差距评估。
这些是 always-on 的基础 critic，不参与路由选择（始终运行）。
"""
from pathlib import Path
from .gap_types import GapChannel, GapScore, RepairDirective, RepairAction
from .backends import VLMBackend, DetectorBackend, OCREngine, IQAEngine, PaletteExtractor
import math


def compositional_gap(src_path: str, target_path: str,
                      vlm: VLMBackend = None, detector: DetectorBackend = None) -> GapScore:
    """结构/构图差距——布局、空间关系"""
    prompt = """Compare the composition and layout of these two images.
Rate how closely the source matches the target in:
- overall structure / arrangement of elements
- spatial relationships
Use anchor: 10=completely different, 8=major differences, 6=moderate, 4=similar with key diffs, 2=nearly identical, 0=identical.
Reply in JSON: {"score": int, "rationale": "..."}
"""
    if vlm:
        result = vlm.score(prompt, [src_path, target_path], "compositional")
        return GapScore(channel=GapChannel.STRUCTURAL,
                        score=snap_to_anchor(result.get("score", 6)),
                        confidence=0,
                        rationale=result.get("rationale", ""),
                        critic_name="dim.compositional")
    return _mock_gap(GapChannel.STRUCTURAL, 4, "Mock compositional check")


def stylistic_gap(src_path: str, target_path: str,
                  vlm: VLMBackend = None, palette: PaletteExtractor = None) -> GapScore:
    """风格差距——色彩/笔触/光线/氛围"""
    prompt = """Compare the style of these two images. Focus on color palette, brushwork, lighting, mood.
Use anchor: 10=massive style gap, 0=indistinguishable style.
Reply JSON: {"score": int, "rationale": "..."}}
"""
    if vlm:
        result = vlm.score(prompt, [src_path, target_path], "stylistic")
        return GapScore(channel=GapChannel.STYLISTIC,
                        score=snap_to_anchor(result.get("score", 6)),
                        confidence=0,
                        rationale=result.get("rationale", ""),
                        critic_name="dim.stylistic")
    return _mock_gap(GapChannel.STYLISTIC, 5, "Mock stylistic check")


def semantic_gap(src_path: str, target_path: str,
                 vlm: VLMBackend = None, detector: DetectorBackend = None) -> GapScore:
    """内容差距——画了什么、实体、场景"""
    prompt = """Compare what is depicted in these two images: entities, objects, scene, subject matter.
Use anchor: 10=completely different content, 0=identical subject matter.
Reply JSON: {"score": int, "rationale": "..."}}
"""
    if vlm:
        result = vlm.score(prompt, [src_path, target_path], "semantic")
        return GapScore(channel=GapChannel.SEMANTIC,
                        score=snap_to_anchor(result.get("score", 6)),
                        confidence=0,
                        rationale=result.get("rationale", ""),
                        critic_name="dim.semantic")
    return _mock_gap(GapChannel.SEMANTIC, 3, "Mock semantic check")


def quality_gap(src_path: str, target_path: str,
                vlm: VLMBackend = None, iqa: IQAEngine = None) -> GapScore:
    """质量差距——完成度、技术细节、分辨率"""
    prompt = """Compare technical quality: finish level, detail resolution, rendering skill.
Use anchor: 10=the source is far rougher, 0=identical finish quality.
Reply JSON: {"score": int, "rationale": "..."}}
"""
    if vlm:
        result = vlm.score(prompt, [src_path, target_path], "quality")
        return GapScore(channel=GapChannel.QUALITY,
                        score=snap_to_anchor(result.get("score", 6)),
                        confidence=0,
                        rationale=result.get("rationale", ""),
                        critic_name="dim.quality")
    return _mock_gap(GapChannel.QUALITY, 4, "Mock quality check")


def palette_diff(src_path: str, target_path: str, palette: PaletteExtractor = None) -> GapScore:
    """调色板差距（程序化检测，高置信度）"""
    if palette:
        src_pal = palette.extract(src_path)
        tgt_pal = palette.extract(target_path)
        # Simplified color difference metric
        diff = 0.0
        for i in range(min(3, len(src_pal.dominant), len(tgt_pal.dominant))):
            dr = abs(src_pal.dominant[i][0] - tgt_pal.dominant[i][0])
            dg = abs(src_pal.dominant[i][1] - tgt_pal.dominant[i][1])
            db = abs(src_pal.dominant[i][2] - tgt_pal.dominant[i][2])
            diff += (dr + dg + db) / (3 * 256)
        score = min(10, diff / 3 * 10)
        return GapScore(channel=GapChannel.STYLISTIC,
                        score=snap_to_anchor(score),
                        confidence=0,
                        rationale=f"Palette distance={diff:.2f}, contrast diff={abs(src_pal.contrast_ratio - tgt_pal.contrast_ratio):.2f}",
                        critic_name="dim.palette_diff")
    return _mock_gap(GapChannel.STYLISTIC, 3, "Mock palette diff")


def text_fidelity(src_path: str, target_path: str, ocr: OCREngine = None) -> GapScore:
    """文字渲染忠实度"""
    if ocr:
        src_text = ocr.extract(src_path)
        tgt_text = ocr.extract(target_path)
        if not tgt_text.strip():
            return GapScore(channel=GapChannel.SEMANTIC, score=0, confidence=0,
                            rationale="No text in target", critic_name="dim.text_fidelity")
        # Simple Levenshtein-like ratio
        score = 10 * (1 - _text_similarity(src_text, tgt_text))
        return GapScore(channel=GapChannel.SEMANTIC,
                        score=snap_to_anchor(score),
                        confidence=0,
                        rationale=f"src='{src_text[:50]}' vs tgt='{tgt_text[:50]}'",
                        critic_name="dim.text_fidelity")
    return _mock_gap(GapChannel.SEMANTIC, 0, "Mock text fidelity")


def anatomy_check(src_path: str, target_path: str,
                  vlm: VLMBackend = None) -> GapScore:
    """解剖学/物理合理性检查"""
    prompt = """Check the source for anatomical/physical implausibility vs target.
Look for: extra fingers, impossible poses, broken proportions.
Rate: 10=severe anatomy issues, 0=physically perfect.
Reply JSON: {"score": int, "rationale": "..."}}
"""
    if vlm:
        result = vlm.score(prompt, [src_path], "anatomy")
        return GapScore(channel=GapChannel.QUALITY,
                        score=snap_to_anchor(result.get("score", 6)),
                        confidence=0,
                        rationale=result.get("rationale", ""),
                        critic_name="dim.anatomy")
    return _mock_gap(GapChannel.QUALITY, 2, "Mock anatomy check")


def snap_to_anchor(raw: float) -> int:
    """Snap to {0, 2, 4, 6, 8, 10}"""
    anchors = [0, 2, 4, 6, 8, 10]
    return min(anchors, key=lambda a: abs(a - raw))


def _mock_gap(channel: GapChannel, score: float, rationale: str) -> GapScore:
    return GapScore(channel=channel, score=int(score), confidence=0,
                    rationale=rationale, critic_name="mock")


def _text_similarity(a: str, b: str) -> float:
    """Simple dice coefficient for text comparison"""
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(len(a_set), len(b_set))


# Dimension critic registry
DIMENSION_CRITICS = {
    "compositional": compositional_gap,
    "stylistic": stylistic_gap,
    "semantic": semantic_gap,
    "quality": quality_gap,
    "palette_diff": palette_diff,
    "text_fidelity": text_fidelity,
    "anatomy": anatomy_check,
}
