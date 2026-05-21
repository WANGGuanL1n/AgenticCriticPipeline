"""
Image Generators — gpt-image-2 (target) + SenseNova-U1 (src)
约定：SenseNova-U1 = src, gpt-image-2 = target
"""
from dataclasses import dataclass
import os, base64, json, hashlib, numpy as np
from PIL import Image


@dataclass
class ImagePair:
    src_path: str
    target_path: str
    prompt: str
    width: int
    height: int
    src_model: str = "SenseNova-U1"
    target_model: str = "gpt-image-2"


class GPTImage2Generator:
    """gpt-image-2 生成器 (target) — 使用内部 endpoint"""

    def __init__(self, api_key: str = None, api_base: str = None, model: str = "gpt-image-2"):
        self.api_key = api_key or os.environ.get("IMAGE2_API_KEY", "")
        self.api_base = api_base or os.environ.get("IMAGE2_BASE_URL", "https://api.openai.com/v1")
        self.model = model

    def generate(self, prompt: str, output_dir: str = "/tmp/images",
                 width: int = 1024, height: int = 1024) -> str:
        if not self.api_key:
            return self._placeholder(output_dir, prompt, width, height, "target")

        try:
            import requests
            resp = requests.post(
                f"{self.api_base}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "prompt": prompt, "n": 1,
                      "size": f"{width}x{height}", "response_format": "b64_json"},
                timeout=180,
            )
            resp.raise_for_status()
            b64_data = resp.json()["data"][0]["b64_json"]
            os.makedirs(output_dir, exist_ok=True)
            h = abs(hash(prompt)) % 1000000
            path = os.path.join(output_dir, f"target_{h:06d}_{width}x{height}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            return path
        except ImportError:
            return self._placeholder(output_dir, prompt, width, height, "target")
        except Exception as e:
            print(f"  [WARN] gpt-image-2 failed: {e}")
            return self._placeholder(output_dir, prompt, width, height, "target")

    def _placeholder(self, output_dir: str, prompt: str, w: int, h: int, prefix: str) -> str:
        d = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
        img = Image.new("RGB", (w, h), color=((d>>16)&0xFF, (d>>8)&0xFF, d&0xFF))
        os.makedirs(output_dir, exist_ok=True)
        hh = abs(hash(prompt)) % 1000000
        path = os.path.join(output_dir, f"{prefix}_{hh:06d}_{w}x{h}.png")
        img.save(path)
        return path


class SenseNovaU1Generator:
    """SenseNova-U1 生成器 (src) — API 未就绪，生成高斯噪声占位"""

    def __init__(self, api_key: str = None, api_base: str = None):
        self.api_key = api_key or os.environ.get("SENSENOVA_API_KEY", "")
        self.api_base = api_base or os.environ.get("SENSENOVA_API_BASE", "")
        self.ready = bool(self.api_key)

    def generate(self, prompt: str, output_dir: str = "/tmp/images",
                 width: int = 1024, height: int = 1024) -> str:
        if not self.ready:
            return self._gaussian_noise(output_dir, prompt, width, height)

        try:
            import requests
            resp = requests.post(
                f"{self.api_base}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"prompt": prompt, "n": 1, "size": f"{width}x{height}", "response_format": "b64_json"},
                timeout=180,
            )
            resp.raise_for_status()
            b64_data = resp.json()["data"][0]["b64_json"]
            os.makedirs(output_dir, exist_ok=True)
            h = abs(hash(prompt)) % 1000000
            path = os.path.join(output_dir, f"src_{h:06d}_{width}x{height}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            return path
        except Exception as e:
            print(f"  [WARN] SenseNova-U1 failed: {e}")
            return self._gaussian_noise(output_dir, prompt, width, height)

    def _gaussian_noise(self, output_dir: str, prompt: str, w: int, h: int) -> str:
        """高斯噪声占位图 — 模拟未对齐的生成结果"""
        import hashlib
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        noise = rng.randn(h, w, 3) * 50 + 128
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noise)
        os.makedirs(output_dir, exist_ok=True)
        hh = abs(hash(prompt)) % 1000000
        path = os.path.join(output_dir, f"src_{hh:06d}_{w}x{h}.png")
        img.save(path)
        return path


class ImagePairGenerator:
    """SenseNova-U1(src) + gpt-image-2(target)"""

    def __init__(self, image2_key: str = None, image2_base: str = None,
                 image2_model: str = "gpt-image-2",
                 sensenova_key: str = None, sensenova_base: str = None,
                 output_dir: str = "/tmp/images"):
        self.target_gen = GPTImage2Generator(api_key=image2_key, api_base=image2_base, model=image2_model)
        self.src_gen = SenseNovaU1Generator(api_key=sensenova_key, api_base=sensenova_base)
        self.output_dir = output_dir

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> ImagePair:
        target_path = self.target_gen.generate(prompt, self.output_dir, width, height)
        src_path = self.src_gen.generate(prompt, self.output_dir, width, height)
        return ImagePair(src_path=src_path, target_path=target_path,
                         prompt=prompt, width=width, height=height)
