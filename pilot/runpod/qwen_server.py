"""Base-photo server for a RunPod GPU pod: Qwen-Image-Edit-2511 (Apache 2.0) behind one HTTP endpoint,
so the Studio app on your computer can use the pod like Gemini ("RunPod" engine).

    POST /generate   Authorization: Bearer <JAGIMAGE_RUNPOD_TOKEN>
      {"prompt": "...", "images": ["<base64 png/jpg>", ...], "width": 1104, "height": 1472,
       "steps": 40, "true_cfg_scale": 4.0, "seed": 0}
    -> {"png": "<base64>", "seconds": 31.2}
    GET /health      -> {"ok": true, "model": ..., "gpu": ...}

The first front of a new model has no reference photo; it is made from a plain studio-grey canvas
(the edit model needs an input image), so only one 20B model has to fit on the GPU.
See README.md for the pod setup. Standard library HTTP; torch and diffusers come from the pod image.
"""

import base64
import io
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen-Image-Edit-2511")
TOKEN = os.environ.get("JAGIMAGE_RUNPOD_TOKEN", "")
PORT = int(os.environ.get("PORT", "8000"))
OFFLOAD = os.environ.get("QWEN_OFFLOAD") == "1"  # 48 GB cards (L40S, A6000): slower, but fits
MAX_SIDE = 2048
CANVAS = (242, 242, 242)  # light grey studio backdrop for text-only shots

if not TOKEN:
    raise SystemExit("Set JAGIMAGE_RUNPOD_TOKEN (any long random string; the same one goes in the Studio app)")

print(f"Loading {MODEL} ...", flush=True)
t0 = time.time()
pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
if OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"Loaded in {time.time() - t0:.0f} s", flush=True)
lock = threading.Lock()  # one image at a time on the GPU; extra requests wait their turn


def decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def generate(body: dict) -> dict:
    w, h = int(body.get("width", 1104)), int(body.get("height", 1472))
    if not (256 <= w <= MAX_SIDE and 256 <= h <= MAX_SIDE):
        raise ValueError(f"width and height must be 256-{MAX_SIDE}")
    w, h = w // 16 * 16, h // 16 * 16
    images = [decode(b) for b in body.get("images") or []][:3]
    if not images:
        images = [Image.new("RGB", (w, h), CANVAS)]
    with lock:
        t = time.time()
        out = pipe(image=images, prompt=str(body["prompt"]), negative_prompt=str(body.get("negative") or " "),
                   width=w, height=h, num_inference_steps=int(body.get("steps", 40)),
                   true_cfg_scale=float(body.get("true_cfg_scale", 4.0)), guidance_scale=1.0,
                   num_images_per_prompt=1, generator=torch.manual_seed(int(body.get("seed", 0)))).images[0]
        took = time.time() - t
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {"png": base64.b64encode(buf.getvalue()).decode(), "seconds": round(took, 2)}


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self):
        return secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {TOKEN}")

    def do_GET(self):
        if self.path != "/health":
            return self._json({"error": "not found"}, 404)
        if not self._authorised():
            return self._json({"error": "wrong token"}, 401)
        self._json({"ok": True, "model": MODEL, "gpu": torch.cuda.get_device_name(0), "offload": OFFLOAD})

    def do_POST(self):
        if self.path != "/generate":
            return self._json({"error": "not found"}, 404)
        if not self._authorised():
            return self._json({"error": "wrong token"}, 401)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._json(generate(body))
        except (KeyError, ValueError, TypeError) as exc:
            self._json({"error": str(exc)}, 400)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._json({"error": "GPU out of memory: use a smaller size, or QWEN_OFFLOAD=1"}, 500)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"Serving on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
