"""`jagimage serve`: the Studio web app on your own computer. Pick the category, top and bottom,
their colours and how many models, choose Gemini or OpenAI, and it makes the base photos in a
batch and recolours every colour pair. Runs on 127.0.0.1 only; API keys stay in this process."""

import io
import json
import re
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
from .cloud import Forbidden

PAGE = Path(__file__).with_name("static") / "studio.html"
RECOLOR = Path(__file__).resolve().parents[1] / "web" / "recolor"  # the Jag Recolor web app (index.html + samples)
LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jag Studio: sign in</title><style>body{font:16px system-ui,sans-serif;background:#eef0f3;display:grid;place-items:center;min-height:100vh;margin:0;padding:16px}
form{background:#fff;border:1px solid #d9dde4;border-radius:12px;padding:24px;display:grid;gap:12px;width:min(340px,100%)}h1{margin:0;font-size:22px}
input,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #d9dde4}button{background:#3346a8;color:#fff;border:0}p{color:#a13;margin:0;min-height:1.2em}</style></head>
<body><form id="f"><h1>Jag Studio</h1><input id="un" placeholder="User name" value="admin" autocomplete="username" autocapitalize="none"><input type="password" id="pw" placeholder="Password" autocomplete="current-password" autofocus><button>Sign in</button><p id="e"></p></form>
<script>f.onsubmit=async(ev)=>{ev.preventDefault();const r=await fetch("/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:un.value,password:pw.value})});
if(r.ok)location.href="/";else e.textContent=(await r.json()).error||"Wrong password"}</script></body></html>"""
KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "runpod": "JAGIMAGE_RUNPOD_TOKEN"}
RUNPOD_URL_ENV = "JAGIMAGE_RUNPOD_URL"


def ready() -> dict:
    """Which engines can make images now: a key saved (RunPod also needs the pod address)."""
    return {p: bool(os.environ.get(env)) and (p != "runpod" or bool(os.environ.get(RUNPOD_URL_ENV)))
            for p, env in KEY_ENV.items()}


def drive_folders() -> list:
    """Google Drive for desktop folders on this computer: files saved there upload to Drive by themselves."""
    import glob
    import string

    home = Path.home()
    found = [Path(p) for p in glob.glob(str(home / "Library" / "CloudStorage" / "GoogleDrive-*" / "My Drive"))]
    found += [home / "Google Drive" / "My Drive", home / "Google Drive", home / "GoogleDrive" / "My Drive"]
    if os.name == "nt":
        found += [Path(f"{d}:/My Drive") for d in string.ascii_uppercase[3:]]
    return [str(p) for p in found if p.is_dir()]


def options(app=None) -> dict:
    return {
        "categories": L.CATEGORIES, "tops": studio.TOPS, "bottoms": studio.BOTTOMS,
        "fixed_bottoms": list(studio.FIXED_BOTTOMS), "fabrics": L.FABRICS, "palettes": L.COLOURS,
        "poses": list(L.POSES), "pose_texts": L.POSES, "shots": studio.SHOTS, "shot_defaults": studio.SHOT_DEFAULTS, "providers": ready(), "runpod_url": os.environ.get(RUNPOD_URL_ENV, ""),
        "gemini_sizes": ["1K", "2K", "4K"], "openai_qualities": ["medium", "high", "xhigh", "max"],
        "bases": {"top": studio.TOP_BASES, "bottom": studio.BOTTOM_BASES},
        "out": str(Path(app.out).resolve()) if app else None, "can_remember": bool(app and app.keys_file),
        "cloud": bool(app and app.cloud), "drive_folders": drive_folders(),
        "next_model": {c: studio.ModelLedger(app.out).next_free(c) for c in L.CATEGORIES} if app else {},
    }


def save_keys(path: Path):
    """Keep the saved API keys in a file in the user's own profile (desktop app, opt-in)."""
    keys = {env: os.environ[env] for env in [*KEY_ENV.values(), RUNPOD_URL_ENV] if os.environ.get(env)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(keys))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def load_keys(path: Path):
    """Put remembered keys into this process (only ones not already set)."""
    try:
        keys = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return
    allowed = {*KEY_ENV.values(), RUNPOD_URL_ENV}
    for env, value in keys.items():
        if env in allowed and isinstance(value, str) and value and not os.environ.get(env):
            os.environ[env] = value


GUIDE_LIMIT = 21_000_000  # a guide upload: about 15 MB of file once base64-encoded

def default_engine(spec: studio.Spec):
    from .providers import make_provider

    from .providers import GEMINI_MODELS

    model = GEMINI_MODELS[spec.gemini_model] if spec.provider == "gemini" else None
    provider = make_provider(spec.provider, model=model, quality=spec.quality, image_size=spec.image_size,
                             api_key=os.environ.get(KEY_ENV[spec.provider]), url=os.environ.get(RUNPOD_URL_ENV))

    def generate(prompt, refs):
        return provider.generate(prompt, refs)

    if hasattr(provider, "run_batch"):
        generate.run_batch = provider.run_batch  # batch mode (half price), used when spec.batch is on
    if hasattr(provider, "usage_log"):  # Google's billed token counts, for the batch's real cost
        generate.set_usage_log = lambda path: setattr(provider, "usage_log", path)
    return generate


class Job:
    """One or more batches run one after another, e.g. the same models in several top styles."""

    def __init__(self, specs, out: str, engine, workers: int, stage: str, export=None, export_what="jpg", redo=False, prints=None,
                 checker=None):
        specs = specs if isinstance(specs, list) else [specs]
        self.id = uuid.uuid4().hex[:8]
        self.specs, self.spec, self.stage = specs, specs[0], stage
        self.plan = sum_plans([sp.plan() for sp in specs])
        self.started, self.finished, self.error = time.time(), None, None
        self.log = deque(maxlen=200)
        self.batch_state = None
        self.recent = deque(maxlen=48)
        self.batches = [studio.Batch(sp, out=out, generate=engine(sp) if stage not in ("recolor", "prints") else None,
                                     workers=workers, on_event=self.event, export=export, export_what=export_what, redo=redo,
                                     prints=prints, checker=checker)
                        for sp in specs]
        self.batch = self.batches[0]
        self.export = str(self.batch.export) if self.batch.export else None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def event(self, e: dict):
        e = {**e, "t": round(time.time() - self.started, 1), "style": self.batch.spec.top}
        if e["kind"] == "recoloured" and self.batch is self.batches[0]:
            self.recent.appendleft(str(Path(e["path"]).relative_to(self.batch.root)))
        if e["kind"] in ("ai_done", "recoloured", "errors", "batch"):
            self.log.appendleft(e)
        if e["kind"] == "batch":
            self.batch_state = e

    def _run(self):
        try:
            for b in self.batches:
                self.batch = b
                b.run(self.stage)
                if b.stop.is_set():
                    break
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self.finished = time.time()

    def stop(self):
        for b in self.batches:
            b.stop.set()

    def status(self) -> dict:
        c = {k: sum(b.counts[k] for b in self.batches) for k in self.batches[0].counts}
        fatal = next((b.fatal for b in self.batches if b.fatal), None)
        return {"id": self.id, "name": self.spec.name, "names": [sp.name for sp in self.specs],
                "styles": [sp.top for sp in self.specs], "current": self.batch.spec.name,
                "spec": asdict(self.spec), "stage": self.stage, "plan": self.plan,
                "counts": c, "running": self.finished is None, "stopping": self.batch.stop.is_set(),
                "error": self.error or fatal, "elapsed": round((self.finished or time.time()) - self.started),
                "ai_progress": c["ai_done"] + c["ai_skipped"], "recolour_progress": c["recoloured"] + c["recolour_skipped"],
                "log": list(self.log)[:40], "recent": list(self.recent), "export": self.export,
                "batch": self.batch_state, "actual_cost": self.actual_cost()}

    def actual_cost(self):
        from .pricing import actual_cost

        rows = [r for b in self.batches if (r := actual_cost(b.root / "_usage.jsonl", b.spec.plan()["recoloured_images"]))]
        if not rows:
            return None
        inr = sum(r["inr"] for r in rows)
        req = sum(r["requests"] for r in rows)
        photos = sum(b.spec.plan()["recoloured_images"] for b in self.batches)
        return {"inr": inr, "requests": req, "per_request_inr": round(inr / req, 2), "per_photo_inr": round(inr / max(1, photos), 2),
                "extra_images_inr": sum(r["extra_images_inr"] for r in rows)}


def sum_plans(plans):
    out = dict(plans[0])
    for p in plans[1:]:
        for k in ("ai_images", "recoloured_images", "disk_gb"):
            out[k] = round(out[k] + p[k], 1) if k == "disk_gb" else out[k] + p[k]
    out["styles"] = len(plans)
    return out


def specs_from(body: dict, ledger) -> list:
    """The batches one request makes: its top style plus any extra styles (body["tops"]), all with the
    same models. Model numbers are automatic (faces never used before) unless given."""
    tops = [t for t in (body.pop("tops", None) or []) if t]
    tops = [body.get("top", "HalfSleeveTee")] + [t for t in tops if t != body.get("top", "HalfSleeveTee")]
    if not body.get("first_model"):
        body["first_model"] = ledger.next_free(body.get("category", "Kids_Boys"))
    first = studio.Spec(**body)
    return [first] + [studio.Spec(**{**body, "top": t, "name": "", "same_models": True}) for t in tops[1:]]


class Preview:
    """One test model in one pose (plus its front, which the pose is made from), in the first
    colour pair: 1-2 AI images, so a new pose or prompt can be checked before a whole batch."""

    def __init__(self, spec: studio.Spec, pose: str, model_no: int, out: str, engine):
        import shutil

        self.id = uuid.uuid4().hex[:8]
        top, bottom = spec.pairs()[0]
        body = {**asdict(spec), "models": 1, "first_model": model_no, "top_colours": [top],
                "bottom_colours": [bottom] if bottom and spec.pairing == "each" else [],
                "poses": ["front_main"] + ([pose] if pose != "front_main" else []), "name": "preview"}
        self.spec = studio.Spec(**body)
        self.pose, self.error, self.done = pose, None, False
        root = Path(out) / "_previews"
        shutil.rmtree(root / "preview", ignore_errors=True)  # always a fresh shoot with the current prompt
        self.batch = studio.Batch(self.spec, out=str(root), generate=engine(self.spec), workers=1, retries=2,
                                  on_event=lambda e: e["kind"] == "errors" and setattr(self, "error", e["error"]))
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.batch.run()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self.error = self.batch.fatal or self.error
        self.done = True

    def status(self) -> dict:
        m = self.spec.first_model
        base = self.spec.bases()[0]
        top, bottom = self.spec.pairs()[0]
        rel = lambda p: str(p.relative_to(self.batch.root.parent)) if p.exists() else None
        return {"id": self.id, "done": self.done, "error": self.error, "pose": self.pose, "model": m,
                "top": top, "bottom": bottom, "counts": self.batch.counts,
                "base": rel(self.batch.base_path(m, base, self.pose)),
                "result": rel(self.batch.out_path(m, top, bottom, self.pose))}


class AgentRun:
    """Runs an agent plan's batches one after another (the Studio runs one batch at a time)."""

    def __init__(self, app, bodies: list, label: str):
        self.id, self.app, self.bodies, self.label = uuid.uuid4().hex[:8], app, bodies, label
        self.done, self.job_ids, self.error, self.stopped = 0, [], None, False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for body in self.bodies:
            if self.stopped:
                break
            while any(j.finished is None for j in self.app.jobs.values()):  # wait for a batch already running
                time.sleep(5)
            try:
                job = self.app.start(dict(body))
            except Exception as exc:
                self.error = str(exc)
                break
            self.job_ids.append(job.id)
            job.thread.join()
            if job.error or any(b.fatal for b in job.batches):
                self.error = job.error or next(b.fatal for b in job.batches if b.fatal)
                break
            self.done += 1

    def status(self) -> dict:
        return {"id": self.id, "label": self.label, "batches": len(self.bodies), "done": self.done, "jobs": self.job_ids,
                "error": self.error, "running": self.thread.is_alive(), "stopped": self.stopped}


class Studio:
    def __init__(self, out="output/studio", engine=default_engine, workers=4, keys_file=None, cloud=None):
        self.out, self.engine, self.workers = out, engine, workers
        self.cloud = cloud  # server version: login, Google Drive upload, agent (see cloud.py)
        self.agent_runs = {}
        self.keys_file = Path(keys_file) if keys_file else None  # desktop app: remembered API keys
        self.jobs = {}
        self.previews = {}
        self.lock = threading.Lock()

    def start(self, body: dict) -> Job:
        stage = body.pop("stage", "all")
        redo = stage in ("redo", "redo_prints")  # make finished photos again (no AI)
        stage = {"redo": "recolor", "redo_prints": "prints"}.get(stage, stage)
        if stage not in ("all", "bases", "recolor", "prints"):
            raise ValueError("stage must be all, bases, recolor, redo or prints")
        if stage == "prints" and not self.cloud:
            raise ValueError("Prints come from Google Drive print folders, in the website version")
        workers = max(1, min(16, int(body.pop("workers", self.workers) or 1)))  # parallel AI requests
        export, export_what = (body.pop("export", None) or "").strip() or None, body.pop("export_what", "jpg")
        if export:
            Path(export).expanduser().mkdir(parents=True, exist_ok=True)  # fail now, not after the AI calls
        ledger = studio.ModelLedger(self.out)
        specs = specs_from(body, ledger)
        spec = specs[0]
        hit = ledger.clash(spec) if stage not in ("recolor", "prints") else None  # recolouring makes no new faces
        if hit:
            raise ValueError(f"Models {hit['first']}-{hit['last']} of {spec.category} are already in batch {hit['batch']}, "
                             f"so those faces would repeat. Start at model no. {ledger.next_free(spec.category)} "
                             "(or leave it empty for automatic).")
        if stage not in ("recolor", "prints") and self.engine is default_engine and not ready()[spec.provider]:
            raise ValueError("Add your RunPod pod address and token first" if spec.provider == "runpod" else f"Add your {spec.provider} API key first")
        with self.lock:
            if any(j.finished is None for j in self.jobs.values()):
                raise ValueError("A batch is already running. Stop it or wait for it to finish.")
            checker = None
            if stage == "prints" and spec.print_check:
                if not (ready().get("gemini") or ready().get("openai")):
                    raise ValueError("The print checker needs a Gemini or OpenAI key (Settings → API keys), or turn it off")
                checker = self.cloud.agent.check_print
            job = Job(specs, self.out, self.engine, workers, stage, export, export_what, redo=redo,
                      prints=self.cloud.print_files if self.cloud and stage == "prints" else None, checker=checker)
            for sp in specs:
                ledger.reserve(sp)
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

    def dashboard(self) -> dict:
        """Everything the Dashboard tab shows: totals, one row per batch with its quality-check results,
        the photos that need attention, and what is running now."""
        from .pricing import USD_INR, actual_cost, per_image

        batches, attention = [], []
        for row in self.folders():
            name, spec = row["name"], row["spec"]
            root = Path(self.out) / name
            size = "x".join(str(v) for v in (spec.get("canvas") or (1500, 1500)))  # the batch's marketplace JPG size
            photos = passed = flagged = 0
            de = []
            for qc_file in root.glob("m[0-9][0-9][0-9][0-9]/*/qc.json"):
                try:
                    poses = json.loads(qc_file.read_text())
                except ValueError:
                    continue
                for pose, qc in poses.items():
                    photos += 1
                    de += [v["deltaE"] for k, v in qc.items() if k in ("top", "bottom")]
                    check = qc.get("check")
                    if check is None:
                        continue  # made before every photo was checked
                    if check["passed"]:
                        passed += 1
                        continue
                    flagged += 1
                    rel = qc_file.parent.relative_to(root).as_posix()
                    marks = qc_file.parent / f"{pose}_qc.jpg"
                    attention.append({"batch": name, "model": qc_file.parent.parent.name, "colours": qc_file.parent.name.replace("__", " / ").replace("_", " "),
                                      "pose": pose, "problems": check["problems"], "updated": qc_file.stat().st_mtime,
                                      "photo": f"{rel}/{pose}_{size}.jpg", "marks": f"{rel}/{pose}_qc.jpg" if marks.exists() else None})
            ai_images = sum(1 for f in root.glob("m[0-9][0-9][0-9][0-9]/_base/*/*.png") if not f.stem.endswith("_mask"))
            real = actual_cost(root / "_usage.jsonl", photos)
            provider = spec.get("provider", "gemini")
            setting = {"gemini": spec.get("image_size", "2K"), "openai": spec.get("quality", "medium")}.get(provider, "gpu")
            try:
                est = per_image(provider, setting, None) * ai_images * USD_INR
            except Exception:
                est = 0.0
            uploaded = root / "_drive.txt"
            batches.append({"name": name, "category": spec.get("category"), "top": spec.get("top"), "bottom": spec.get("bottom"),
                            "fabric": spec.get("fabric"), "provider": provider, "updated": row["updated"],
                            "models": len(list(root.glob("m[0-9][0-9][0-9][0-9]"))), "ai_images": ai_images, "photos": photos,
                            "checked": passed + flagged, "passed": passed, "flagged": flagged,
                            "deltaE_avg": round(sum(de) / len(de), 2) if de else None, "deltaE_max": round(max(de), 2) if de else None,
                            "on_drive": len(uploaded.read_text().splitlines()) if uploaded.exists() else 0,
                            "cost_inr": real["inr"] if real else round(est), "cost_is_real": bool(real)})
        totals = {k: sum(b[k] for b in batches) for k in ("models", "ai_images", "photos", "checked", "passed", "flagged", "on_drive", "cost_inr")}
        totals["batches"] = len(batches)
        running = [j.status() for j in self.jobs.values() if j.finished is None]
        return {"totals": totals, "batches": batches, "attention": sorted(attention, key=lambda a: -a["updated"])[:60],
                "running": [{"id": r["id"], "name": r["name"], "counts": r["counts"], "plan": r["plan"]} for r in running]}

    def batch_for(self, name: str) -> studio.Batch:
        spec_file = Path(self.out) / name / "spec.json"
        if not name or "/" in name or ".." in name or not spec_file.exists():
            raise FileNotFoundError(f"There is no batch called {name!r}" if name else "Pick a batch first")
        return studio.Batch(studio.Spec(**json.loads(spec_file.read_text())), out=self.out)

    def zip_bytes(self, name: str, what="jpg") -> bytes:
        b = self.batch_for(name)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for r in b.results():
                rel = r["jpg"] if (what == "jpg" or not r["png"]) and r["jpg"] else r["png"]
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

        def _body(self, limit: int = 2_000_000) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n < 0:
                raise ValueError("bad request size")
            if n > limit:
                # read (and drop) what was sent, so the browser gets this answer instead of a broken connection
                left = min(n, 4 * limit)
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    left -= len(chunk)
                if n > 4 * limit:
                    self.close_connection = True
                raise ValueError("request too large" if limit == 2_000_000 else f"That file is too big: keep it under {limit // 1_400_000} MB")
            raw = self.rfile.read(n)
            # Only JSON from this app's own pages: a form or text/plain POST from another website (which a
            # browser sends without asking) is refused, as JSON needs the browser's permission first.
            if raw and "application/json" not in (self.headers.get("Content-Type") or ""):
                raise ValueError("send JSON (Content-Type: application/json)")
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("send a JSON object")
            return body

        def _host_ok(self) -> bool:
            """Desktop app: answer only pages opened on this computer (not a website pointing its own name at
            127.0.0.1, "DNS rebinding"). The website version checks the sign-in instead."""
            if app.cloud:
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host in ("127.0.0.1", "localhost", "::1", ""):
                return True
            self._json({"error": "open Jag Studio at http://127.0.0.1"}, 403)
            return False

        def _client_ip(self) -> str:
            """Behind Railway's proxy the real address is the last X-Forwarded-For entry (the ones before it are
            whatever the visitor sent)."""
            fwd = [x.strip() for x in (self.headers.get("X-Forwarded-For") or "").split(",") if x.strip()]
            return fwd[-1] if fwd else self.client_address[0]

        def _redirect(self, where, extra=None):
            self._send(302, b"", "text/plain", {"Location": where, **(extra or {})})

        def _signed_in(self, parts) -> bool:
            """Server version: everything but the sign-in page needs the session cookie."""
            self.user = app.cloud and app.cloud.login.user(self.headers.get("Cookie", ""))
            if not app.cloud or parts[:1] == ["login"] or self.user:
                return True
            if parts[:1] == ["api"]:
                self._json({"error": "Sign in first"}, 401)
            else:
                self._redirect("/login")
            return False

        def _admin(self):
            """Server version: settings, keys, users and Drive are for admins only."""
            if app.cloud and app.cloud.users.role(self.user) != "admin":
                raise Forbidden("Only an admin can do this")

        def _origin(self) -> str:
            host = self.headers.get("Host", "localhost")
            local = host.split(":")[0] in ("localhost", "127.0.0.1")
            return ("http://" if local else "https://") + host

        def _cloud_get(self, parts, q):
            c = app.cloud
            if parts == ["login"]:
                return self._send(200, LOGIN_PAGE.encode(), "text/html; charset=utf-8")
            if parts == ["logout"]:
                return self._redirect("/login", {"Set-Cookie": "jag_session=; Path=/; Max-Age=0"})
            if parts == ["api", "drive"]:
                return self._json(c.drive_status())
            if parts == ["api", "me"]:
                return self._json({"user": self.user, "role": c.users.role(self.user)})
            if parts == ["api", "users"]:
                self._admin()
                return self._json({"users": c.users.listing()})
            if parts == ["drive", "connect"]:
                self._admin()
                if not c.drive.configured:
                    return self._json({"error": "Google Drive isn't set up yet: an admin pastes the Google OAuth client in the Google Drive panel"}, 400)
                state = uuid.uuid4().hex
                c.oauth_states[state] = time.time()
                return self._redirect(c.drive.auth_url(self._origin() + "/drive/callback", state))
            if parts == ["drive", "callback"]:
                state, code = q.get("state", [""])[0], q.get("code", [""])[0]
                if not code or time.time() - c.oauth_states.pop(state, 0) > 600:
                    return self._redirect("/?drive=failed")
                c.drive.finish(code, self._origin() + "/drive/callback")
                c.sync.wake.set()
                return self._redirect("/?drive=connected")
            if parts == ["api", "agent", "runs"]:
                return self._json({"runs": [r.status() for r in app.agent_runs.values()]})
            if parts == ["api", "guides"]:
                return self._json({"guides": c.guides()})
            if parts == ["api", "prints"]:
                from .prints import PLACE_NAMES

                return self._json({"rows": c.print_rows(), "places": PLACE_NAMES, "can_read": c.drive.can_read})
            return None

        def _print_preview(self, body) -> dict:
            """One printed photo of a batch at the given position (nothing saved), optionally with the AI
            print checker's verdict. The print is an uploaded file to try, or one from the print folders."""
            import base64
            import io

            from PIL import Image

            from .prints import adjust

            c = app.cloud
            batch = app.batch_for(str(body.get("batch") or ""))
            targets = batch.print_targets()
            if not targets:
                raise ValueError("This batch has no finished front photos to print on")
            colours = sorted({t["top"] for t in targets}, key=[x["top"] for x in targets].index)
            colour = body.get("colour") if body.get("colour") in colours else colours[0]
            target = next(t for t in targets if t["top"] == colour)
            names, art = [], None
            if body.get("art"):
                from .prints import PrintError, open_image

                try:
                    art = open_image(base64.b64decode(str(body["art"]), validate=True))
                except PrintError as exc:
                    raise ValueError(f"That print file can't be used: {exc}")
                except Exception:
                    raise ValueError("That print file can't be read: use a PNG or JPG")
            else:
                files = c.print_files(batch.spec.category, batch.spec.top, colour)
                names = [n for n, _ in files]
                if not files:
                    raise ValueError(f"No print folder for {batch.spec.category} {batch.spec.top} {colour}: add one below, or upload a print file to try")
                pick = next((f for f in files if f[0] == body.get("print")), files[0])
                from .prints import open_image

                art = open_image(str(pick[1]))
            adj = adjust(body.get("dx"), body.get("dy"), body.get("scale"))
            place = body.get("place") or "chest_large"
            out, mask = batch.preview_print(target, art, place, **adj)
            check = None
            if body.get("check"):
                if not (ready().get("gemini") or ready().get("openai")):
                    raise ValueError("The print checker needs a Gemini or OpenAI key (Settings → API keys)")
                check = studio.judge_print(out, art, mask, place, adj, c.agent.check_print)
            shown = out.convert("RGB")
            shown.thumbnail((900, 1200))
            buf = io.BytesIO()
            shown.save(buf, "JPEG", quality=88)
            return {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "colours": colours,
                    "colour": colour, "prints": names, "print": None if body.get("art") else (body.get("print") if body.get("print") in names else names[0]),
                    "model": target["model"], "check": check, **adj}

        def _cloud_post(self, parts, body):
            c = app.cloud
            if parts == ["login"]:
                ip = self._client_ip()
                name = str(body.get("username") or "admin").strip().lower()
                if not c.login.try_login(ip, name, str(body.get("password", ""))):
                    return self._json({"error": "Wrong user name or password"}, 403)
                return self._send(200, b"{}", "application/json", {"Set-Cookie": c.login.cookie(name)})
            if parts == ["api", "guide"]:
                # a marketplace's model photo guidelines (e.g. Myntra's): read by the AI into Studio settings,
                # saved for everyone, so an admin's job like the print folders
                import base64

                self._admin()
                name = str(body.get("filename") or "guide").strip()[:120]
                try:
                    data = base64.b64decode(str(body.get("data") or ""), validate=True)
                except ValueError:
                    raise ValueError("The file didn't arrive in one piece: upload it again")
                if not data:
                    raise ValueError("Choose the guide file first")
                if not (ready().get("gemini") or ready().get("openai")):
                    raise ValueError("Add a Gemini or OpenAI API key in Settings → API keys first")
                return self._json({"guide": c.save_guide(c.agent.read_guide(name, data)), "guides": c.guides()})
            if parts == ["api", "prints", "preview"]:
                return self._json(self._print_preview(body))
            if parts == ["api", "prints", "plan"]:
                # "a bit lower and smaller": the print agent turns the shop's words into a placement
                message = " ".join(str(body.get("message") or "").split())[:500]
                if not message:
                    raise ValueError("Tell the print agent what to change, e.g. a bit lower and smaller")
                if not (ready().get("gemini") or ready().get("openai")):
                    raise ValueError("Add a Gemini or OpenAI API key in Settings → API keys first")
                cur = body.get("current") if isinstance(body.get("current"), dict) else {}
                return self._json(c.agent.plan_print(message, cur))
            if parts == ["api", "prints"]:
                self._admin()
                return self._json({"rows": c.set_print_rows(body.get("rows")), "can_read": c.drive.can_read})
            if len(parts) == 4 and parts[:2] == ["api", "guides"] and parts[3] == "delete":
                self._admin()
                c.delete_guide(unquote(parts[2]))
                return self._json({"guides": c.guides()})
            if parts == ["api", "account", "password"]:
                if not c.users.check(self.user, str(body.get("current", ""))):
                    raise ValueError("Your current password is wrong")
                c.users.set_password(self.user, str(body.get("new", "")))
                return self._send(200, b"{}", "application/json", {"Set-Cookie": c.login.cookie(self.user)})  # stay signed in here
            if parts == ["api", "users"]:
                self._admin()
                c.users.create(body.get("username"), str(body.get("password", "")), body.get("role", "user"))
                return self._json({"users": c.users.listing()})
            if len(parts) == 4 and parts[:2] == ["api", "users"] and parts[3] in ("password", "delete"):
                self._admin()
                name = unquote(parts[2])
                if name not in c.users.users:
                    return self._json({"error": "no such user"}, 404)
                if parts[3] == "password":
                    c.users.set_password(name, str(body.get("password", "")))
                else:
                    if name == self.user:
                        raise ValueError("You can't delete yourself")
                    c.users.delete(name)
                return self._json({"users": c.users.listing()})
            if parts == ["api", "drive", "client"]:
                self._admin()
                c.drive.set_client(body.get("client_id"), body.get("client_secret"))
                return self._json(c.drive_status())
            if parts == ["api", "drive", "disconnect"]:
                self._admin()
                c.drive.disconnect()
                return self._json(c.drive_status())
            if parts == ["api", "drive", "sync"]:
                c.sync.wake.set()
                return self._json(c.drive_status())
            if parts == ["api", "agent"]:
                message = str(body.get("message", "")).strip()
                if not message:
                    raise ValueError("Tell the agent what to make")
                base = {k: body[k] for k in ("provider", "gemini_model", "image_size", "quality", "batch", "one_base", "keep_png", "hands_free") if k in body}
                have, note = ready(), ""
                if not have.get(base.get("provider", "gemini")):  # the chosen engine has no key: use one that has
                    other = next((p for p in ("gemini", "openai") if have[p]), None)
                    if not other:
                        raise ValueError("Add a Gemini or OpenAI API key in Settings → API keys first")
                    note = f"The photos will be made with {dict(gemini='Gemini', openai='OpenAI')[other]}, the AI that has a key saved."
                    base.update(provider=other, batch=bool(base.get("batch")) and other in ("gemini", "openai"))
                    if other == "openai":
                        base["quality"] = base.get("quality") if base.get("quality") in ("medium", "high") else "medium"
                plan = c.agent.plan(message, studio.ModelLedger(app.out), base)
                if note:
                    plan["reply"] = (plan["reply"] + " " + note).strip()
                plan_id = uuid.uuid4().hex[:10]
                c.plans[plan_id] = {"bodies": plan.pop("bodies"), "label": message[:120], "time": time.time()}
                return self._json({**plan, "plan_id": plan_id})
            if parts == ["api", "agent", "start"]:
                plan = c.plans.pop(str(body.get("plan_id", "")), None)
                if not plan or not plan["bodies"]:
                    raise ValueError("That plan is gone or empty. Ask the agent again.")
                if any(r.thread.is_alive() for r in app.agent_runs.values()):
                    raise ValueError("The agent is already running a plan. Stop it or wait for it to finish.")
                run = AgentRun(app, plan["bodies"], plan["label"])
                app.agent_runs[run.id] = run
                run.thread.start()
                return self._json(run.status())
            if len(parts) == 5 and parts[:3] == ["api", "agent", "runs"] and parts[4] == "stop":
                run = app.agent_runs.get(parts[3])
                if not run:
                    return self._json({"error": "no such run"}, 404)
                run.stopped = True
                for j in app.jobs.values():
                    if j.id in run.job_ids and j.finished is None:
                        j.stop()
                return self._json(run.status())
            return None

        def do_GET(self):
            u = urlparse(self.path)
            parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
            q = parse_qs(u.query)
            try:
                if not self._host_ok() or not self._signed_in(parts):
                    return
                if app.cloud and self._cloud_get(parts, q) is not None:
                    return
                if not parts:
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if parts[:1] == ["recolor"]:  # Jag Recolor, shown in the Studio page's Recolor tab
                    rel = "/".join(parts[1:]) or "index.html"
                    path = (RECOLOR / rel).resolve()
                    if RECOLOR.resolve() not in path.parents or not path.is_file():
                        return self._json({"error": "not found"}, 404)
                    ctype = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".png": "image/png"}.get(path.suffix, "application/octet-stream")
                    return self._send(200, path.read_bytes(), ctype)
                if parts == ["api", "options"]:
                    return self._json(options(app))
                if parts == ["api", "dashboard"]:
                    return self._json(app.dashboard())
                if parts == ["api", "jobs"]:
                    return self._json({"jobs": [j.status() for j in app.jobs.values()], "folders": app.folders()})
                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    job = app.jobs.get(parts[2])
                    return self._json(job.status()) if job else self._json({"error": "no such job"}, 404)
                if len(parts) == 3 and parts[:2] == ["api", "preview"]:
                    pv = app.previews.get(parts[2])
                    return self._json(pv.status()) if pv else self._json({"error": "no such preview"}, 404)
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "results":
                    rows = app.batch_for(parts[2]).results()
                    off, lim = int(q.get("offset", ["0"])[0]), int(q.get("limit", ["60"])[0])
                    return self._json({"total": len(rows), "rows": rows[off:off + lim]})
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "zip":
                    data = app.zip_bytes(parts[2], q.get("what", ["jpg"])[0])
                    return self._send(200, data, "application/zip",
                                      {"Content-Disposition": 'attachment; filename="%s.zip"' % re.sub(r"[^A-Za-z0-9_.-]", "_", parts[2])})
                if parts[0] in ("files", "thumbs") and len(parts) >= 3:
                    out_root = Path(app.out).resolve()
                    root = (out_root / parts[1]).resolve()
                    path = (root / "/".join(parts[2:])).resolve()
                    # only files inside one batch folder of the photos folder
                    if root.parent != out_root or root not in path.parents or not path.is_file():
                        return self._json({"error": "not found"}, 404)
                    if parts[0] == "thumbs":
                        return self._send(200, thumb(path), "image/jpeg")
                    ctype = "image/png" if path.suffix == ".png" else "image/jpeg"
                    return self._send(200, path.read_bytes(), ctype)
                return self._json({"error": "not found"}, 404)
            except FileNotFoundError:
                return self._json({"error": "not found"}, 404)
            except Forbidden as exc:
                return self._json({"error": str(exc)}, 403)
            except Exception as exc:
                return self._json({"error": str(exc)}, 400)

        def do_POST(self):
            parts = [p for p in urlparse(self.path).path.strip("/").split("/") if p]
            try:
                if not self._host_ok() or not self._signed_in(parts):
                    return
                body = self._body(GUIDE_LIMIT if parts in (["api", "guide"], ["api", "prints", "preview"]) else 2_000_000)
                if app.cloud and self._cloud_post(parts, body) is not None:
                    return
                if parts == ["api", "preview"]:
                    pose, model_no = body.pop("preview_pose", "front_main"), int(body.pop("preview_model", 0) or 0)
                    for k in ("stage", "workers", "export", "export_what", "tops"):  # a test is the first style only
                        body.pop(k, None)
                    body["first_model"] = body.get("first_model") or studio.ModelLedger(app.out).next_free(body.get("category", "Kids_Boys"))
                    spec = studio.Spec(**body)
                    if pose not in spec.pose_texts():
                        raise ValueError(f"unknown pose {pose}")
                    if app.engine is default_engine and not ready()[spec.provider]:
                        raise ValueError("Add your RunPod pod address and token first" if spec.provider == "runpod" else f"Add your {spec.provider} API key first")
                    if any(not p.done for p in app.previews.values()):
                        raise ValueError("A test photo is already being made")
                    pv = Preview(spec, pose, model_no or spec.first_model, app.out, app.engine)
                    app.previews = {pv.id: pv}
                    return self._json(pv.status())
                if parts == ["api", "prompt"]:
                    # the exact prompt a base photo gets, before any money is spent
                    pose, model_no = body.pop("preview_pose", "front_main"), body.pop("preview_model", None)
                    for k in ("stage", "workers", "export", "export_what", "tops"):  # a test is the first style only
                        body.pop(k, None)
                    body["first_model"] = body.get("first_model") or studio.ModelLedger(app.out).next_free(body.get("category", "Kids_Boys"))
                    spec = studio.Spec(**body)
                    if pose not in spec.pose_texts():
                        raise ValueError(f"unknown pose {pose}")
                    base = spec.bases()[0]
                    m = int(model_no or spec.first_model)
                    ref = "none" if pose == "front_main" else "front"
                    refs = [] if pose == "front_main" else ["this model's approved front_main photo"]
                    return self._json({"prompt": studio.prompt(spec, m, base, pose, ref), "model": m,
                                       "base": studio.base_dir(base), "reference_images": refs})
                if parts == ["api", "plan"]:
                    for k in ("stage", "workers", "export", "export_what"):
                        body.pop(k, None)
                    ledger = studio.ModelLedger(app.out)
                    auto = not body.get("first_model")
                    specs = specs_from(body, ledger)
                    spec = specs[0]
                    hit = ledger.clash(spec)
                    from .pricing import estimate, sum_estimates
                    total = sum_plans([sp.plan() for sp in specs])
                    return self._json({**total, "first_model": spec.first_model, "auto": auto,
                                       "cost": sum_estimates([estimate(sp) for sp in specs], total["recoloured_images"]),
                                       "faces_ready": sum(1 for m in spec.model_numbers() if studio.Batch(spec, out=app.out).face_file(m).exists()),
                                       "clash": hit and f"Models {hit['first']}-{hit['last']} are already in batch {hit['batch']}: faces would repeat. Next free: {ledger.next_free(spec.category)}.", "bases": [studio.base_dir(b) for b in spec.bases()],
                                       "pairs": spec.pairs(), "name": spec.name})
                if parts == ["api", "settings"]:
                    self._admin()
                    out = (body.get("out") or "").strip()
                    if out:
                        if any(j.finished is None for j in app.jobs.values()):
                            raise ValueError("Change the working folder when no batch is running")
                        Path(out).expanduser().mkdir(parents=True, exist_ok=True)
                        app.out = str(Path(out).expanduser())
                    return self._json(options(app))
                if parts == ["api", "keys"]:
                    self._admin()
                    for p, env in KEY_ENV.items():
                        if body.get(p):
                            os.environ[env] = body[p].strip()
                    for p in body.get("clear") or []:  # remove a saved key
                        if p in KEY_ENV:
                            os.environ.pop(KEY_ENV[p], None)
                            if p == "runpod":
                                os.environ.pop(RUNPOD_URL_ENV, None)
                    url = (body.get("runpod_url") or "").strip().rstrip("/")
                    if url:
                        if not url.startswith(("https://", "http://")):
                            raise ValueError("The RunPod address starts with https://, e.g. https://abc123-8000.proxy.runpod.net")
                        os.environ[RUNPOD_URL_ENV] = url
                    if app.keys_file and (body.get("remember") or body.get("clear")):
                        save_keys(app.keys_file)
                    return self._json(ready())
                if parts == ["api", "jobs"]:
                    if body.get("export"):
                        self._admin()  # writing to a folder of one's choice on the server
                    return self._json(app.start(body).status())
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "prints":
                    # Dashboard: the shop's prints (Drive print folders) on this batch's finished front photos (no AI)
                    spec = json.loads((app.batch_for(unquote(parts[2])).root / "spec.json").read_text())
                    want = {k: body[k] for k in ("print_place", "print_dx", "print_dy", "print_scale", "print_check") if k in body}
                    # redo: print again over earlier printed photos (e.g. after moving the print)
                    job = app.start({**spec, **want, "stage": "redo_prints" if body.get("redo") else "prints", "same_models": True})
                    return self._json(job.status())
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "redo":
                    # Dashboard: recolour a whole batch again with the latest recolouring (no AI)
                    spec = json.loads((app.batch_for(unquote(parts[2])).root / "spec.json").read_text())
                    return self._json(app.start({**spec, "stage": "redo", "same_models": True}).status())
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
                    job = app.jobs.get(parts[2])
                    if not job:
                        return self._json({"error": "no such job"}, 404)
                    job.stop()
                    return self._json(job.status())
                return self._json({"error": "not found"}, 404)
            except Forbidden as exc:
                return self._json({"error": str(exc)}, 403)
            except PermissionError as exc:
                return self._json({"error": str(exc)}, 429)
            except (TypeError, ValueError, OSError, RuntimeError, KeyError, AttributeError) as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:  # never drop the connection without an answer
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

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


def serve(host="127.0.0.1", port=8765, out="output/studio", workers=4, engine=default_engine, keys_file=None, cloud=None):
    app = Studio(out=out, engine=engine, workers=workers, keys_file=keys_file)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    default_error = httpd.handle_error

    def handle_error(request, client_address):  # a browser that closed the page mid-download isn't an error
        import sys

        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            default_error(request, client_address)

    httpd.handle_error = handle_error
    if cloud is not None:  # a cloud.Cloud, or True to build one from the environment
        from .cloud import Cloud

        app.cloud = cloud if not isinstance(cloud, bool) else Cloud(
            Path(os.environ.get("JAGIMAGE_DATA") or Path(out).resolve().parent / "jag-data"), lambda: app.out)
        app.keys_file = app.keys_file or (app.cloud.data / "keys.json")  # API keys saved on the server, opt-in
        load_keys(app.keys_file)
        app.cloud.sync.start()
    return app, httpd
