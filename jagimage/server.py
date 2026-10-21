"""`jagimage serve`: the Studio web app on your own computer. Pick the category, top and bottom,
their colours and how many models, choose Gemini or OpenAI, and it makes the base photos in a
batch and recolours every colour pair. Runs on 127.0.0.1 only; API keys stay in this process."""

import io
import json
import os
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import library as L
from . import studio

PAGE = Path(__file__).with_name("static") / "studio.html"
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def options() -> dict:
    return {
        "categories": L.CATEGORIES, "tops": studio.TOPS, "bottoms": studio.BOTTOMS,
        "fixed_bottoms": list(studio.FIXED_BOTTOMS), "fabrics": L.FABRICS, "palettes": L.COLOURS,
        "poses": list(L.POSES), "providers": {p: bool(os.environ.get(env)) for p, env in KEY_ENV.items()},
        "gemini_sizes": ["1K", "2K", "4K"], "openai_qualities": ["medium", "high", "xhigh", "max"],
        "bases": {"top": studio.TOP_BASES, "bottom": studio.BOTTOM_BASES},
    }


def default_engine(spec: studio.Spec):
    from .providers import make_provider

    provider = make_provider(spec.provider, quality=spec.quality, image_size=spec.image_size,
                             api_key=os.environ.get(KEY_ENV[spec.provider]))
    return provider.generate


class Job:
    def __init__(self, spec: studio.Spec, out: str, engine, workers: int, stage: str):
        self.id = uuid.uuid4().hex[:8]
        self.spec, self.stage, self.plan = spec, stage, spec.plan()
        self.started, self.finished, self.error = time.time(), None, None
        self.log = deque(maxlen=200)
        self.recent = deque(maxlen=48)
        self.batch = studio.Batch(spec, out=out, generate=engine(spec) if stage != "recolor" else None,
                                  workers=workers, on_event=self.event)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def event(self, e: dict):
        e = {**e, "t": round(time.time() - self.started, 1)}
        if e["kind"] == "recoloured":
            self.recent.appendleft(str(Path(e["path"]).relative_to(self.batch.root)))
        if e["kind"] in ("ai_done", "recoloured", "errors"):
            self.log.appendleft(e)

    def _run(self):
        try:
            self.batch.run(self.stage)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self.finished = time.time()

    def status(self) -> dict:
        c = self.batch.counts
        return {"id": self.id, "name": self.spec.name, "spec": asdict(self.spec), "stage": self.stage, "plan": self.plan,
                "counts": c, "running": self.finished is None, "stopping": self.batch.stop.is_set(),
                "error": self.error, "elapsed": round((self.finished or time.time()) - self.started),
                "ai_progress": c["ai_done"] + c["ai_skipped"], "recolour_progress": c["recoloured"] + c["recolour_skipped"],
                "log": list(self.log)[:40], "recent": list(self.recent)}


class Studio:
    def __init__(self, out="output/studio", engine=default_engine, workers=4):
        self.out, self.engine, self.workers = out, engine, workers
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, body: dict) -> Job:
        stage = body.pop("stage", "all")
        if stage not in ("all", "bases", "recolor"):
            raise ValueError("stage must be all, bases or recolor")
        workers = int(body.pop("workers", self.workers))
        spec = studio.Spec(**body)
        if stage != "recolor" and self.engine is default_engine and not os.environ.get(KEY_ENV[spec.provider]):
            raise ValueError(f"Add your {spec.provider} API key first")
        with self.lock:
            if any(j.finished is None for j in self.jobs.values()):
                raise ValueError("A batch is already running. Stop it or wait for it to finish.")
            job = Job(spec, self.out, self.engine, workers, stage)
            self.jobs[job.id] = job
        job.thread.start()
        return job

    def folders(self) -> list:
        """Batches on disk (also from earlier sessions), newest first."""
        rows = []
        for spec_file in Path(self.out).glob("*/spec.json"):
            rows.append({"name": spec_file.parent.name, "spec": json.loads(spec_file.read_text()),
                         "updated": spec_file.stat().st_mtime})
        return sorted(rows, key=lambda r: -r["updated"])

    def batch_for(self, name: str) -> studio.Batch:
        spec_file = Path(self.out) / name / "spec.json"
        if "/" in name or ".." in name or not spec_file.exists():
            raise FileNotFoundError(name)
        return studio.Batch(studio.Spec(**json.loads(spec_file.read_text())), out=self.out)

    def zip_bytes(self, name: str, what="jpg") -> bytes:
        b = self.batch_for(name)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for r in b.results():
                rel = r["jpg"] if what == "jpg" and r["jpg"] else r["png"]
                ext = Path(rel).suffix
                label = f"{r['model']}_{r['top']}" + (f"__{r['bottom']}" if r["bottom"] else "") + f"_{r['pose']}{ext}"
                z.write(b.root / rel, f"{name}/{label}")
        return buf.getvalue()


def make_handler(app: Studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data).encode())

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
            q = parse_qs(u.query)
            try:
                if not parts:
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if parts == ["api", "options"]:
                    return self._json(options())
                if parts == ["api", "jobs"]:
                    return self._json({"jobs": [j.status() for j in app.jobs.values()], "folders": app.folders()})
                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    job = app.jobs.get(parts[2])
                    return self._json(job.status()) if job else self._json({"error": "no such job"}, 404)
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "results":
                    rows = app.batch_for(parts[2]).results()
                    off, lim = int(q.get("offset", ["0"])[0]), int(q.get("limit", ["60"])[0])
                    return self._json({"total": len(rows), "rows": rows[off:off + lim]})
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "zip":
                    data = app.zip_bytes(parts[2], q.get("what", ["jpg"])[0])
                    return self._send(200, data, "application/zip",
                                      {"Content-Disposition": f'attachment; filename="{parts[2]}.zip"'})
                if parts[0] in ("files", "thumbs") and len(parts) >= 3:
                    root = (Path(app.out) / parts[1]).resolve()
                    path = (root / "/".join(parts[2:])).resolve()
                    if root not in path.parents or not path.is_file():
                        return self._json({"error": "not found"}, 404)
                    if parts[0] == "thumbs":
                        return self._send(200, thumb(path), "image/jpeg")
                    ctype = "image/png" if path.suffix == ".png" else "image/jpeg"
                    return self._send(200, path.read_bytes(), ctype)
                return self._json({"error": "not found"}, 404)
            except FileNotFoundError:
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                return self._json({"error": str(exc)}, 400)

        def do_POST(self):
            parts = [p for p in urlparse(self.path).path.strip("/").split("/") if p]
            try:
                body = self._body()
                if parts == ["api", "plan"]:
                    body.pop("stage", None)
                    body.pop("workers", None)
                    spec = studio.Spec(**body)
                    return self._json({**spec.plan(), "bases": [studio.base_dir(b) for b in spec.bases()],
                                       "pairs": spec.pairs(), "name": spec.name})
                if parts == ["api", "keys"]:
                    for p, env in KEY_ENV.items():
                        if body.get(p):
                            os.environ[env] = body[p].strip()
                    return self._json(options()["providers"])
                if parts == ["api", "jobs"]:
                    return self._json(app.start(body).status())
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
                    job = app.jobs.get(parts[2])
                    if not job:
                        return self._json({"error": "no such job"}, 404)
                    job.batch.stop.set()
                    return self._json(job.status())
                return self._json({"error": "not found"}, 404)
            except (TypeError, ValueError) as exc:
                return self._json({"error": str(exc)}, 400)

    return Handler


_thumb_lock = threading.Lock()


def thumb(path: Path, size=360) -> bytes:
    """Small JPEG for the gallery, cached next to the photo."""
    from PIL import Image

    cache = path.with_name(f".{path.stem}_thumb{size}.jpg")
    with _thumb_lock:
        if not cache.exists() or cache.stat().st_mtime < path.stat().st_mtime:
            im = Image.open(path).convert("RGB")
            im.thumbnail((size, size * 2))
            im.save(cache, quality=85)
    return cache.read_bytes()


def serve(host="127.0.0.1", port=8765, out="output/studio", workers=4, engine=default_engine):
    app = Studio(out=out, engine=engine, workers=workers)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return app, httpd
