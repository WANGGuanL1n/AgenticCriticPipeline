"""
Backend protocols — 可插拔的实际服务接口
"""
from typing import Protocol, Optional
from dataclasses import dataclass
import base64, json, urllib.request, urllib.error


class VLMBackend(Protocol):
    """VLM 调用后端"""
    def generate(self, prompt: str, images: list[str]) -> str: ...
    def score(self, prompt: str, images: list[str], rubric: str = "") -> dict: ...


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


# --------------- Real VLM Backend (OpenAI-compatible API) ---------------

def _image_to_data_url(path: str) -> str:
    """Encode image file as base64 data URL"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    suffix = path.rsplit(".", 1)[-1].lower()
    if suffix == "jpg":
        suffix = "jpeg"
    if suffix not in ("png", "jpeg", "gif", "webp"):
        suffix = "png"
    return f"data:image/{suffix};base64,{b64}"


class OpenAICompatVLM:
    """OpenAI-compatible vision API backend (Chat Completions with images)"""

    def __init__(self, api_key: str, base_url: str, model: str = "gemini-3-flash-preview"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._call_log: list[dict] = []  # 记录所有 VLM 调用的原始数据（成功+失败）

    def _chat(self, prompt: str, images: list[str],
              response_format: dict | None = None) -> tuple[str, dict]:
        """Send a chat completion request with images via Vision API.
        Returns (content_text, raw_response_metadata).
        """
        content = [{"type": "text", "text": prompt}]
        for img_path in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(img_path)},
            })

        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if response_format:
            body["response_format"] = response_format

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM API HTTP {e.code}: {body_text[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"VLM API connection failed: {e}") from e

        result = json.loads(raw)
        message = result["choices"][0]["message"]

        # Extract usage / reasoning metadata
        usage = result.get("usage", {})
        completion_details = usage.get("completion_tokens_details", {})
        reasoning_tokens = completion_details.get("reasoning_tokens", 0)

        metadata = {
            "model": result.get("model", self.model),
            "finish_reason": result["choices"][0].get("finish_reason", ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "reasoning_tokens": reasoning_tokens,
        }

        return message["content"], metadata

    def generate(self, prompt: str, images: list[str]) -> str:
        """Send prompt + images, return raw text"""
        text, _ = self._chat(prompt, images)
        return text

    def score(self, prompt: str, images: list[str], rubric: str = "") -> dict:
        """Send prompt + images, return structured dict (with JSON mode).
        Always logs the call (success or failure) to _call_log.
        """
        # Prepare call log entry BEFORE the API call
        call_entry = {
            "rubric": rubric,
            "images_count": len(images),
            "prompt": prompt,
            "success": False,
            "error": None,
            "raw_output": None,
            "usage": None,
        }

        try:
            text, metadata = self._chat(
                prompt + "\n\nReply ONLY with valid JSON, no markdown fences.",
                images,
                response_format={"type": "json_object"},
            )

            # Update call entry with success data
            call_entry["success"] = True
            call_entry["raw_output"] = text
            call_entry["usage"] = metadata

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from markdown fences
                if "```json" in text:
                    text = text.split("```json", 1)[1].split("```", 1)[0]
                elif "```" in text:
                    text = text.split("```", 1)[1].split("```", 1)[0]
                parsed = json.loads(text.strip())

            # Inject VLM usage metadata
            parsed["_vlm_usage"] = metadata
            return parsed

        except Exception as e:
            call_entry["error"] = str(e)[:500]
            raise

        finally:
            self._call_log.append(call_entry)

    def drain_call_log(self) -> list[dict]:
        """Return and clear the accumulated VLM call log (both successful and failed calls)."""
        log = self._call_log.copy()
        self._call_log.clear()
        return log


# --------------- Mock backends for testing / cold start ---------------

class MockVLM:
    def generate(self, prompt: str, images: list[str]) -> str:
        return "Mock VLM response."

    def score(self, prompt: str, images: list[str], rubric: str = "") -> dict:
        return {
            "score": 6, "confidence": 0.7, "rationale": "Mock VLM rationale.",
            "_vlm_usage": {
                "model": "mock", "finish_reason": "stop",
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "reasoning_tokens": 0,
            }
        }

    def drain_call_log(self) -> list[dict]:
        return []


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
