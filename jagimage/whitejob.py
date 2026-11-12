"""White-background copies of the plain model photos in Google Drive, for Amazon and Flipkart main images.

Each photo under a folder of the photos folder ("Cotton Mix/Boys") is downloaded, its studio backdrop and
floor turned pure white (#FFFFFF) with market.on_white (a cutout model keeps the person and the clothes as they
are), and uploaded to the same path under "White background/" ("White background/Cotton Mix/Boys/..."). The
original photos are left as they are. A photo already there is skipped, so a stopped job can be started again."""

import io
import json
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from PIL import Image

ROOT = "White background"
REDO_BEFORE = "2026-11-12T12:38:00Z"  # copies made before the floor-strip fix are made again
WORKERS = 6  # photos done side by side (they share one cutout model, used only for pale clothes)
PHOTO = re.compile(r"\.(jpe?g|png|webp)$", re.I)


class WhiteJob:
    def __init__(self, drive, path: List[str], data: Path, limit: int = 0, skip=("white background", "printed", "_bases"),
                 workers: int = WORKERS, on_done=None):
        self.id = uuid.uuid4().hex[:8]
        self.drive, self.path, self.data = drive, [p for p in path if p], Path(data)
        self.limit, self.skip, self.workers = max(0, int(limit or 0)), skip, max(1, int(workers))
        self.counts = {"photos": 0, "done": 0, "skipped": 0, "errors": 0}
        self.stage, self.error = "starting", None
        self.started, self.finished = time.time(), None
        self.log: List[dict] = []
        self.stop = threading.Event()
        self.on_done = on_done  # called with the job when it ends (the queue starts the next folder)
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"white-{self.id}")

    def status(self) -> dict:
        return {"id": self.id, "path": "/".join(self.path), "stage": self.stage, "counts": dict(self.counts),
                "error": self.error, "running": self.finished is None,
                "elapsed": round((self.finished or time.time()) - self.started), "log": self.log[-20:],
                "to": "/".join([ROOT] + self.path)}

    def _note(self, **kw):
        self.log.append({"t": round(time.time() - self.started, 1), **kw})
        del self.log[:-200]

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            self._go()
        except Exception as exc:  # shown in the tab
            self.error = str(exc)[:300]
        finally:
            self.finished = time.time()
            self.stage = "stopped" if self.stop.is_set() else ("failed" if self.error else "done")
            if self.on_done:
                try:
                    self.on_done(self)
                except Exception:
                    pass

    def _go(self):
        from .market import on_white

        self.stage = "reading Drive"
        files = [f for f in self.drive.list_tree(self.drive.folder_id(self.path), depth=7, limit=400000)
                 if PHOTO.search(f["name"]) and not any(w in p.lower() for p in f["path"] for w in self.skip)]
        if not files:
            raise RuntimeError("No photos in that folder (check the path, e.g. Cotton Mix/Boys)")
        self.counts["photos"] = len(files)
        try:  # what is already white: a stopped job starts again where it left off
            have = {("/".join(f["path"]), f["name"]) for f in self.drive.list_tree(
                self.drive.folder_id([ROOT] + self.path), depth=7, limit=400000)
                if str(f.get("modifiedTime") or REDO_BEFORE) >= REDO_BEFORE}  # older copies are made again
        except Exception:
            have = set()
        tmp = Path(tempfile.mkdtemp(prefix="white-", dir=self.data if self.data.exists() else None))
        todo = []
        for f in files:
            name = re.sub(r"\.(png|webp)$", ".jpg", f["name"], flags=re.I)
            if ("/".join(f["path"]), name) in have:
                self.counts["skipped"] += 1
            else:
                todo.append((f, name))
        if self.limit:
            todo = todo[:self.limit]
        lock = threading.Lock()

        def one(item):
            f, name = item
            if self.stop.is_set():
                return
            self.stage = "/".join(self.path + f["path"])
            local = tmp / f"{f['id']}_{name}"
            try:
                img = Image.open(io.BytesIO(self.drive.download(f["id"]))).convert("RGB")
                on_white(img).save(local, "JPEG", quality=95, subsampling=0)
                self.drive.upload(local, [ROOT] + self.path + list(f["path"]), name)
                with lock:
                    self.counts["done"] += 1
            except Exception as exc:
                with lock:
                    self.counts["errors"] += 1
                    self._note(file="/".join(f["path"] + [f["name"]]), error=str(exc)[:200])
            finally:
                local.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(one, todo))


class WhiteQueue:
    """Folders waiting for white copies, kept on disk so the work goes on after a server restart (each job
    skips the photos already done). One folder at a time; the next starts when one is done."""

    def __init__(self, drive, data: Path, jobs: dict):
        self.drive, self.data, self.jobs = drive, Path(data), jobs
        self.file = self.data / "white_queue.json"
        self.lock = threading.Lock()

    def paths(self) -> List[str]:
        try:
            return [p for p in json.loads(self.file.read_text()).get("paths", []) if p]
        except (OSError, ValueError):
            return []

    def _write(self, paths: List[str]):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"paths": paths}))
        tmp.replace(self.file)

    def start(self, paths: List[str], limit: int = 0) -> "WhiteJob":
        paths = [p for p in ("/".join(x.strip() for x in str(p).split("/") if x.strip()) for p in paths) if p]
        if not paths:
            raise ValueError("Name a folder of the photos folder, e.g. Cotton Mix/Boys")
        with self.lock:
            if any(j.finished is None for j in self.jobs.values()):
                raise ValueError("A white-background job is already going: wait for it or stop it")
            if not limit:  # a test run (limit) is not remembered
                self._write(paths)
            return self._next(paths[0], limit)

    def _next(self, path: str, limit: int = 0) -> "WhiteJob":
        job = WhiteJob(self.drive, path.split("/"), self.data, limit=limit, on_done=None if limit else self._done)
        self.jobs[job.id] = job.start()
        return job

    def _done(self, job: "WhiteJob"):
        if job.stage != "done":  # stopped or failed: the queue waits (a restart or Start carries on)
            return
        with self.lock:
            left = self.paths()
            if left and left[0] == "/".join(job.path):
                left = left[1:]
            self._write(left)
            if left:
                self._next(left[0])

    def stop(self):
        with self.lock:
            self._write([])
        for j in self.jobs.values():
            if j.finished is None:
                j.stop.set()

    def resume(self):
        """After a restart: carry on with the first folder still waiting."""
        left = self.paths()
        if left and not any(j.finished is None for j in self.jobs.values()):
            with self.lock:
                return self._next(left[0])
