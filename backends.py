"""
Backend protocols — 可插拔的实际服务接口
"""
from typing import Protocol, Optional
from dataclasses import dataclass


class VLMBackend(Protocol):
    """VLM 调用后端（Claude API / Qwen-VL / GLM-4V 等）"""
    def generate(self, prompt: str, images: list[str]) -> str: ...
    def score(self, prompt: str, image_path: str, rubric: str) -> dict[str, float]: ...


class DetectorBackend(Protocol):
    """物体检测 / 计数后端（GroundingDINO 等）"""
    def count(self, image_path: str, class_name: str) -> int: ...
    def detect(self, image_path: str, class_names: list[str]) -> dict[str, list[dict]]: ...


class OCREngine(Protocol):
    """OCR 后端"""
    def extract(self, image_path: str) -> str: ...


class IQAEngine(Protocol):
    """图像质量评估（BRISQUE / NIQE 等）"""
    def score(self, image_path: str) -> float: ...


@dataclass
class Palette:
    dominant: list[tuple[int,int,int]]
    color_temp: str  # warm / cool / neutral
    contrast_ratio: float

class PaletteExtractor(Protocol):
    """调色板提取（colorthief 等）"""
    def extract(self, image_path: str) -> Palette: ...


# --------------- Mock backends for testing / cold start ---------------

class MockVLM:
    def generate(self, prompt: str, images: list[str]) -> str:
        return "Mock VLM response."

    def score(self, prompt: str, image_path: str, rubric: str) -> dict[str, float]:
        return {"score": 6.0, "confidence": 0.7}


class MockDetector:
    def count(self, image_path: str, class_name: str) -> int:
        return 1

    def detect(self, image_path: str, class_names: list[str]) -> dict[str, list[dict]]:
        return {c: [{"box": [0.1, 0.1, 0.9, 0.9], "score": 0.9}] for c in class_names}


class MockOCR:
    def extract(self, image_path: str) -> str:
        return "SAMPLE TEXT"


class MockIQA:
    def score(self, image_path: str) -> float:
        return 0.8


class MockPalette:
    def extract(self, image_path: str) -> Palette:
        return Palette(dominant=[(200, 180, 160)], color_temp="warm", contrast_ratio=0.6)
