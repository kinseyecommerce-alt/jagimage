"""Image engines. Each provider turns (prompt, reference image paths) into PNG bytes.

- openai: gpt-image models via the OpenAI SDK (OPENAI_API_KEY).
- gemini: Gemini image models ("Nano Banana") via the google-genai SDK (GEMINI_API_KEY),
  at 1K/2K/4K resolution with a fixed 3:4 aspect ratio.
- runpod: your own GPU pod running pilot/runpod/qwen_server.py (Qwen-Image-Edit-2511, open model),
  reached at JAGIMAGE_RUNPOD_URL with the token JAGIMAGE_RUNPOD_TOKEN. You pay RunPod by the hour.
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


    # ---- Batch mode: half price, results within 24 hours (usually much sooner)
    TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    def run_batch(self, jobs, on_status=None, stop=None, state_file=None, poll=30, chunk=None):
        """Sends jobs in batch jobs of at most `chunk` photos (JAGIMAGE_BATCH_CHUNK, default 400), one
        after another, so each stays under a new Gemini account's batch quota (Tier 1: about 2 million
        enqueued tokens per model). Raise it once the project reaches Tier 2."""
        import time

        chunk = int(chunk or os.environ.get("JAGIMAGE_BATCH_CHUNK", "400"))
        say = on_status or (lambda s: None)
        out = {}
        parts = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
        for n, part in enumerate(parts, 1):
            if n > 1 and stop is not None and stop.is_set():
                break
            label = (lambda st, n=n: say(f"{st} (job {n} of {len(parts)})")) if len(parts) > 1 else say
            for attempt in range(6):
                try:
                    out.update(self._run_batch_job(part, label, stop, state_file, poll))
                    break
                except Exception as exc:  # quota full or too many jobs queued: wait, then send again
                    text = str(exc).lower()
                    if "429" not in text and "resource_exhausted" not in text or "credits" in text or attempt == 5:
                        raise
                    label("waiting for Google's batch quota")
                    if stop is not None and stop.wait(min(600, 60 * (attempt + 1))):
                        return out
                    if stop is None:
                        time.sleep(min(600, 60 * (attempt + 1)))
        return out

    def _run_batch_job(self, jobs, on_status=None, stop=None, state_file=None, poll=30):
        """jobs: [(key, prompt, reference paths)]. Sends them as one Gemini batch job and returns
        {key: PNG bytes or Exception}. The job's name is kept in state_file until its results are
        read, so if the app stops, the next start collects that job instead of paying for it twice.
        Returns {} if stopped while waiting (the job keeps running at Google)."""
        import json
        import tempfile
        import time
        from pathlib import Path

        from google.genai import types

        say = on_status or (lambda s: None)
        state = json.loads(Path(state_file).read_text()) if state_file and Path(state_file).exists() else {}
        wanted = {k for k, _, _ in jobs}
        pending = [name for name, keys in state.items() if set(keys) & wanted]
        if pending:  # an earlier job (app stopped while waiting) covers some of these: collect it first
            name = pending[0]
        else:
            uris = {}

            def file_part(path):
                if path not in uris:
                    f = self.client.files.upload(file=path, config=types.UploadFileConfig(mime_type="image/png"))
                    uris[path] = {"fileUri": f.uri, "mimeType": f.mime_type or "image/png"}
                return {"fileData": uris[path]}

            lines = []
            for key, text, refs in jobs:
                lines.append(json.dumps({"key": key, "request": {
                    "contents": [{"role": "user", "parts": [file_part(r) for r in refs] + [{"text": text}]}],
                    "generationConfig": {"responseModalities": ["IMAGE"],
                                         "imageConfig": {"aspectRatio": self.aspect_ratio, "imageSize": self.image_size}}}}))
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
                tmp.write("\n".join(lines))
            src = self.client.files.upload(file=tmp.name, config=types.UploadFileConfig(display_name="jag-studio-batch", mime_type="jsonl"))
            Path(tmp.name).unlink(missing_ok=True)
            job = self.client.batches.create(model=self.model, src=src.name, config={"display_name": f"jag-studio {len(jobs)} images"})
            name = job.name
            if state_file:
                state[name] = sorted(wanted)
                Path(state_file).write_text(json.dumps(state, indent=1))
        while True:
            job = self.client.batches.get(name=name)
            st = getattr(job.state, "name", str(job.state))
            say(st.replace("JOB_STATE_", "").lower())
            if st in self.TERMINAL:
                break
            if stop is not None and stop.wait(poll):
                return {}
            if stop is None:
                time.sleep(poll)
        out = {}
        if st in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED") and job.dest and job.dest.file_name:
            for line in self.client.files.download(file=job.dest.file_name).decode().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("key")
                if row.get("error") or not row.get("response"):
                    out[key] = RuntimeError(f"Gemini batch: {row.get('error') or 'no response'}")
                    continue
                img = None
                for cand in row["response"].get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        data = part.get("inlineData") or part.get("inline_data")
                        if data and data.get("data"):
                            img = base64.b64decode(data["data"])
                out[key] = to_png(img) if img else RuntimeError("Gemini batch returned no image for this request")
        else:
            err = RuntimeError(f"Gemini batch job ended {st.replace('JOB_STATE_', '').lower()}: {getattr(job, 'error', '')}")
            out = {k: err for k in (state.get(name) or wanted)}
        if state_file and name in state:
            del state[name]
            Path(state_file).write_text(json.dumps(state, indent=1))
        return out


class RunPodProvider:
    """Calls the Qwen server on a RunPod pod. 1104x1472 is 3:4 at about Qwen's native 1.6 megapixels."""
    name = "runpod"

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, size: Optional[str] = None,
                 steps: Optional[int] = None, timeout=600, opener=None):
        self.url = (url or os.environ.get("JAGIMAGE_RUNPOD_URL", "")).rstrip("/")
        self.token = token or os.environ.get("JAGIMAGE_RUNPOD_TOKEN", "")
        if not self.url or not self.token:
            raise ValueError("Add your RunPod pod address and token first")
        self.width, self.height = (int(v) for v in (size or os.environ.get("JAGIMAGE_RUNPOD_SIZE", "1104x1472")).split("x"))
        self.steps = int(steps or os.environ.get("JAGIMAGE_RUNPOD_STEPS", "40"))
        self.timeout = timeout
        self.seconds: List[float] = []  # GPU time per image, for the pilot's cost figures
        if opener is None:
            from urllib.request import urlopen as opener
        self.opener = opener

    def generate(self, prompt: str, refs: List[str]) -> bytes:
        import json
        from urllib.error import HTTPError, URLError
        from urllib.request import Request

        images = []
        for r in refs:
            with open(r, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())
        body = json.dumps({"prompt": prompt, "images": images, "width": self.width, "height": self.height,
                           "steps": self.steps, "seed": int.from_bytes(os.urandom(4), "big")}).encode()
        req = Request(self.url + "/generate", data=body, method="POST",
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with self.opener(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 401:
                raise PermissionError(f"401 RunPod token not accepted: {detail}") from exc
            raise RuntimeError(f"RunPod pod error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise ConnectionError(f"RunPod pod unreachable at {self.url}: {exc.reason}") from exc
        self.seconds.append(float(out.get("seconds", 0)))
        return to_png(base64.b64decode(out["png"]))


def make_provider(name: str = "openai", **kw):
    if name == "runpod":
        return RunPodProvider(url=kw.get("url"), token=kw.get("api_key"))
    if name == "gemini":
        return GeminiProvider(model=kw.get("model"), image_size=kw.get("image_size", "2K"), api_key=kw.get("api_key"))
    return OpenAIProvider(model=kw.get("model"), quality=kw.get("quality", "high"),
                          size=kw.get("size", "1024x1536"), fidelity=kw.get("fidelity"), api_key=kw.get("api_key"))
