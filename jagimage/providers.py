"""Image engines. Each provider turns (prompt, reference image paths) into PNG bytes.

- openai: gpt-image models via the OpenAI SDK (OPENAI_API_KEY).
- gemini: Gemini image models ("Nano Banana") via the google-genai SDK (GEMINI_API_KEY),
  at 1K/2K/4K resolution with a fixed 3:4 aspect ratio.
"""

import base64
import io
import mimetypes
import os
from contextlib import ExitStack
from typing import List, Optional

GEMINI_DEFAULT_MODEL = "gemini-3-pro-image"
GEMINI_SIZES = ("1K", "2K", "4K")


def to_png(data: bytes) -> bytes:
    """Re-encode any image bytes as PNG (Gemini may return JPEG)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data
    from PIL import Image

    out = io.BytesIO()
    Image.open(io.BytesIO(data)).save(out, format="PNG")
    return out.getvalue()


class OpenAIProvider:
    name = "openai"

    def __init__(self, client=None, model: Optional[str] = None, quality="high", size="1024x1536", fidelity=None,
                 api_key: Optional[str] = None):
        from .agent import DEFAULT_MODEL

        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.client = client
        self.model = model or os.environ.get("JAGIMAGE_MODEL", DEFAULT_MODEL)
        self.quality = quality
        self.size = size
        self.fidelity = fidelity

    def generate(self, prompt: str, refs: List[str]) -> bytes:
        params = dict(model=self.model, prompt=prompt, size=self.size, quality=self.quality,
                      background="opaque", output_format="png", n=1)
        if refs:
            if self.fidelity:
                params["input_fidelity"] = self.fidelity
            with ExitStack() as stack:
                files = [stack.enter_context(open(r, "rb")) for r in refs]
                result = self.client.images.edit(image=files, **params)
        else:
            result = self.client.images.generate(**params)
        return base64.b64decode(result.data[0].b64_json)


class GeminiProvider:
    name = "gemini"

    def __init__(self, client=None, model: Optional[str] = None, image_size="2K", aspect_ratio="3:4",
                 api_key: Optional[str] = None):
        if image_size not in GEMINI_SIZES:
            raise ValueError(f"Gemini image size must be one of {', '.join(GEMINI_SIZES)}")
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key) if api_key else genai.Client()  # else GEMINI_API_KEY / GOOGLE_API_KEY
        self.client = client
        self.model = model or os.environ.get("JAGIMAGE_GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
        self.image_size = image_size
        self.aspect_ratio = aspect_ratio

    def generate(self, prompt: str, refs: List[str]) -> bytes:
        from google.genai import types

        # Reference images first, in order (image 1, image 2 ...), then the instructions.
        contents = []
        for r in refs:
            mime = mimetypes.guess_type(r)[0] or "image/png"
            with open(r, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type=mime))
        contents.append(prompt)
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=self.aspect_ratio, image_size=self.image_size),
        )
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        for cand in resp.candidates or []:
            for part in (cand.content.parts if cand.content else []) or []:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    return to_png(part.inline_data.data)
        reason = ""
        if resp.candidates and getattr(resp.candidates[0], "finish_reason", None):
            reason = f" (finish reason: {resp.candidates[0].finish_reason})"
        raise RuntimeError(f"Gemini returned no image{reason}. Try again or adjust the prompt.")


def make_provider(name: str = "openai", **kw):
    if name == "gemini":
        return GeminiProvider(model=kw.get("model"), image_size=kw.get("image_size", "2K"), api_key=kw.get("api_key"))
    return OpenAIProvider(model=kw.get("model"), quality=kw.get("quality", "high"),
                          size=kw.get("size", "1024x1536"), fidelity=kw.get("fidelity"), api_key=kw.get("api_key"))
