"""Jag Studio on a server (Google Cloud): a password login, finished photos uploaded to Google Drive,
and an agent that turns "make 1000 Men models in half sleeve and hoodie" into batches.

Enabled by `jagimage serve --cloud` with these environment variables:
  JAGIMAGE_PASSWORD              the website password (required)
  JAGIMAGE_DATA                  folder for settings and the Drive sign-in (default: <out>/../jag-data)
  GOOGLE_OAUTH_CLIENT_ID/SECRET  a Google OAuth "Web application" client, for Drive (optional)
  JAGIMAGE_DRIVE_FOLDER          Drive folder the photos go into (default: kinsey models 2026)
  JAGIMAGE_AGENT_MODEL           Gemini text model for the agent (default: gemini-flash-latest)
"""

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

SESSION_DAYS = 30
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"  # only files and folders this app makes
FOLDER_MIME = "application/vnd.google-apps.folder"


# ---------------------------------------------------------------- login
class Login:
    """One shared password. A signed cookie keeps you signed in for SESSION_DAYS; repeated wrong
    passwords from one address are slowed down."""

    def __init__(self, password: str, secret: bytes):
        if not password or len(password) < 8:
            raise ValueError("JAGIMAGE_PASSWORD must be at least 8 characters")
        self.password = password.encode()
        self.secret = secret
        self.failures = {}  # ip -> (count, last time)

    def _sign(self, value: str) -> str:
        return hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    def cookie(self) -> str:
        expires = str(int(time.time()) + SESSION_DAYS * 86400)
        return f"jag_session={expires}.{self._sign(expires)}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age={SESSION_DAYS * 86400}"

    def valid(self, cookie_header: str) -> bool:
        for part in (cookie_header or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == "jag_session" and "." in value:
                expires, sig = value.split(".", 1)
                if hmac.compare_digest(sig, self._sign(expires)) and expires.isdigit() and int(expires) > time.time():
                    return True
        return False

    def try_password(self, ip: str, password: str) -> bool:
        count, last = self.failures.get(ip, (0, 0.0))
        if count >= 5 and time.time() - last < 30 * (count - 4):
            raise PermissionError("Too many wrong passwords. Wait a minute and try again.")
        if hmac.compare_digest(password.encode(), self.password):
            self.failures.pop(ip, None)
            return True
        self.failures[ip] = (count + 1, time.time())
        return False


# ---------------------------------------------------------------- Google Drive
class Drive:
    """Uploads finished photos into a Drive folder of the signed-in Google account (the one-time
    "Connect Google Drive" sign-in keeps a refresh token in the data folder). Plain HTTPS calls to the
    Drive v3 REST API; no extra libraries."""

    def __init__(self, client_id: str, client_secret: str, data: Path, folder: str = "kinsey models 2026",
                 opener: Callable = urllib.request.urlopen):
        self.client_id, self.client_secret = client_id, client_secret
        self.file = Path(data) / "drive.json"
        self.folder = folder
        self.opener = opener
        self.lock = threading.Lock()
        self.state = json.loads(self.file.read_text()) if self.file.exists() else {}
        self._access = (None, 0.0)

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def connected(self) -> bool:
        return bool(self.state.get("refresh_token"))

    def _save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.chmod(tmp, 0o600)
        tmp.replace(self.file)

    def auth_url(self, redirect_uri: str, state: str) -> str:
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
            "client_id": self.client_id, "redirect_uri": redirect_uri, "response_type": "code", "scope": DRIVE_SCOPE,
            "access_type": "offline", "prompt": "consent", "state": state})

    def _post_form(self, url: str, fields: dict) -> dict:
        req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.opener(req, timeout=30) as r:
            return json.loads(r.read())

    def finish(self, code: str, redirect_uri: str):
        tok = self._post_form("https://oauth2.googleapis.com/token", {
            "code": code, "client_id": self.client_id, "client_secret": self.client_secret,
            "redirect_uri": redirect_uri, "grant_type": "authorization_code"})
        if not tok.get("refresh_token"):
            raise RuntimeError("Google didn't return a refresh token; remove the app's access in your Google account and connect again")
        with self.lock:
            self.state = {"refresh_token": tok["refresh_token"], "folders": {}}
            self._access = (tok.get("access_token"), time.time() + int(tok.get("expires_in", 0)) - 60)
            self._save()

    def disconnect(self):
        with self.lock:
            self.state = {}
            self._access = (None, 0.0)
            self._save()

    def _token(self) -> str:
        token, expires = self._access
        if token and time.time() < expires:
            return token
        tok = self._post_form("https://oauth2.googleapis.com/token", {
            "refresh_token": self.state["refresh_token"], "client_id": self.client_id,
            "client_secret": self.client_secret, "grant_type": "refresh_token"})
        self._access = (tok["access_token"], time.time() + int(tok.get("expires_in", 3600)) - 60)
        return tok["access_token"]

    def _call(self, method: str, url: str, body: Optional[bytes] = None, ctype: str = "application/json") -> dict:
        req = urllib.request.Request(url, data=body, method=method,
                                     headers={"Authorization": f"Bearer {self._token()}", "Content-Type": ctype})
        with self.opener(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")

    def folder_id(self, parts: list) -> str:
        """The Drive folder for a path such as [batch, "Navy__Navy"] under the photos folder, made
        once and remembered."""
        path = [self.folder] + list(parts)
        key = "/".join(path)
        with self.lock:
            known = self.state.setdefault("folders", {})
            if key in known:
                return known[key]
            parent = "root"
            for i in range(len(path)):
                sub = "/".join(path[:i + 1])
                if sub not in known:
                    name = path[i].replace("'", "\\'")
                    q = f"name = '{name}' and mimeType = '{FOLDER_MIME}' and '{parent}' in parents and trashed = false"
                    found = self._call("GET", "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
                        {"q": q, "fields": "files(id)", "spaces": "drive"})).get("files") or []
                    known[sub] = found[0]["id"] if found else self._call(
                        "POST", "https://www.googleapis.com/drive/v3/files?fields=id",
                        json.dumps({"name": path[i], "mimeType": FOLDER_MIME, "parents": [parent]}).encode())["id"]
                parent = known[sub]
            self._save()
            return parent

    def upload(self, local: Path, parts: list, name: str) -> str:
        parent = self.folder_id(parts)
        boundary = "jag" + secrets.token_hex(8)
        meta = json.dumps({"name": name, "parents": [parent]}).encode()
        mime = "image/png" if local.suffix == ".png" else "image/jpeg"
        body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode() + meta +
                f"\r\n--{boundary}\r\nContent-Type: {mime}\r\n\r\n".encode() + local.read_bytes() + f"\r\n--{boundary}--\r\n".encode())
        return self._call("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id",
                          body, f"multipart/related; boundary={boundary}")["id"]


class DriveSync:
    """Every minute, uploads finished marketplace JPGs that aren't on Drive yet, as
    <photos folder>/<batch>/<Top__Bottom>/<model>_<pose>.jpg. Survives restarts: what is uploaded is
    listed in each batch folder's _drive.txt."""

    def __init__(self, drive: Drive, out_dir: Callable[[], str], every: float = 60):
        self.drive, self.out_dir, self.every = drive, out_dir, every
        self.status = {"uploaded": 0, "pending": 0, "last_error": None, "last_run": None}
        self.wake = threading.Event()
        self.lock = threading.Lock()

    def pending(self) -> list:
        rows = []
        for spec in Path(self.out_dir()).glob("*/spec.json"):
            batch = spec.parent
            done_file = batch / "_drive.txt"
            done = set(done_file.read_text().splitlines()) if done_file.exists() else set()
            for jpg in sorted(batch.glob("m[0-9][0-9][0-9][0-9]/*/*_1500x1500.jpg")):  # <model>/<Top__Bottom>/<pose>
                rel = jpg.relative_to(batch).as_posix()
                if rel not in done and not jpg.parent.name.startswith("_"):
                    rows.append((batch, rel, jpg))
        return rows

    def run_once(self) -> int:
        if not self.drive.connected:
            return 0
        with self.lock:
            todo = self.pending()
            self.status.update(pending=len(todo), last_run=round(time.time()))
            sent = 0
            for batch, rel, jpg in todo:
                model, pair = jpg.parent.parent.name, jpg.parent.name
                pose = jpg.name.rsplit("_", 1)[0]
                try:
                    self.drive.upload(jpg, [batch.name, pair], f"{model}_{pose}.jpg")
                except Exception as exc:  # network, quota, revoked access: try again next round
                    self.status["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
                    break
                with open(batch / "_drive.txt", "a") as f:
                    f.write(rel + "\n")
                sent += 1
                self.status["uploaded"] += 1
                self.status["pending"] = len(todo) - sent
            else:
                self.status["last_error"] = None
            return sent

    def start(self):
        def loop():
            while True:
                try:
                    self.run_once()
                except Exception as exc:
                    self.status["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
                self.wake.wait(self.every)
                self.wake.clear()
        threading.Thread(target=loop, daemon=True, name="drive-sync").start()


# ---------------------------------------------------------------- agent
AGENT_SYSTEM = """You plan catalogue photo batches for SPB Textile's Jag Studio. Turn the user's request
into batches. Each batch is one category with one or more top styles; all styles reuse the same models.
Allowed values:
- category: Men, Women, Kids_Boys, Kids_Girls
- tops: HalfSleeveTee, FullSleeveTee, SleevelessTee, Hoodie, Sweatshirt
- bottom: Shorts, Trackpants, Jeans (Men/Women default Trackpants, kids default Shorts)
- fabric: PureCotton (colours: Black, White, Light Pink, Red, Navy, Green, Lavender, Orange, Yellow) or
  CottonMix (Black, Blue, Maroon, Melange, White, Red, Yellow, Light Pink, Dark Pink, Navy Blue, Olive,
  Beige, Grey, Royal Blue, Military Green, Pink, Green)
- top_colours: colour names from that fabric; default all of them
- pairing: "match" (bottom in the same colour as the top, default) or "each" (every top with every bottom colour)
- shot: "outfit" (head to shoes, default) or "top" (head to above the knee, bottom not recoloured)
- models: number of different people (1-5000); poses: subset of front_main, back, threequarter, side,
  front_main_white, handpocket (default front_main and back)
Never invent other values. If something essential is unclear, ask in "reply" and return no batches.
"reply" is one or two short plain sentences for the user."""

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "batches": {"type": "array", "items": {"type": "object", "properties": {
            "category": {"type": "string"}, "tops": {"type": "array", "items": {"type": "string"}},
            "bottom": {"type": "string"}, "fabric": {"type": "string"},
            "top_colours": {"type": "array", "items": {"type": "string"}}, "pairing": {"type": "string"},
            "shot": {"type": "string"}, "models": {"type": "integer"},
            "poses": {"type": "array", "items": {"type": "string"}}},
            "required": ["category", "tops", "models"]}},
    },
    "required": ["reply", "batches"],
}


class Agent:
    """Turns a request in plain words into Studio batch requests with their cost. It only plans:
    nothing is made until the person presses Start on the plan."""

    def __init__(self, client=None, model: Optional[str] = None):
        self.client = client
        self.model = model or os.environ.get("JAGIMAGE_AGENT_MODEL", "gemini-flash-latest")

    def _ask(self, message: str) -> dict:
        from google.genai import types

        client = self.client
        if client is None:
            from google import genai

            client = genai.Client()
        resp = client.models.generate_content(
            model=self.model, contents=message,
            config=types.GenerateContentConfig(system_instruction=AGENT_SYSTEM, response_mime_type="application/json",
                                               response_schema=AGENT_SCHEMA, temperature=0.1))
        return json.loads(resp.text)

    def plan(self, message: str, ledger, base: dict) -> dict:
        """base: the engine settings to use (provider, gemini_model, image_size, batch, one_base ...)."""
        from . import library as L
        from . import pricing, studio

        ans = self._ask(message)
        bodies, rows, problems = [], [], []
        for b in ans.get("batches") or []:
            try:
                tops = [t for t in b.get("tops") or [] if t in studio.TOPS] or ["HalfSleeveTee"]
                category = b["category"]
                fabric = b.get("fabric") if b.get("fabric") in L.FABRICS else "PureCotton"
                colours = [c for c in (b.get("top_colours") or []) if c in L.COLOURS[fabric]] or list(L.COLOURS[fabric])
                bottom = b.get("bottom") if b.get("bottom") in studio.BOTTOMS else ("Shorts" if L.is_kids(category) else "Trackpants")
                poses = [p for p in (b.get("poses") or []) if p in L.POSES] or ["front_main", "back"]
                if "front_main" not in poses:
                    poses = ["front_main"] + poses
                body = {**base, "category": category, "top": tops[0], "tops": tops[1:], "bottom": bottom, "fabric": fabric,
                        "top_colours": colours, "pairing": b.get("pairing") if b.get("pairing") in ("match", "each") else "match",
                        "shot": b.get("shot") if b.get("shot") in studio.SHOTS else "outfit",
                        "models": max(1, min(5000, int(b.get("models") or 1))), "poses": poses}
                specs = _specs(dict(body), ledger)
                plans = [sp.plan() for sp in specs]
                cost = pricing.sum_estimates([pricing.estimate(sp) for sp in specs], sum(p["recoloured_images"] for p in plans))
                rows.append({"category": category, "styles": [sp.top for sp in specs], "models": body["models"],
                             "first_model": specs[0].first_model, "colours": len(colours), "poses": poses,
                             "ai_images": sum(p["ai_images"] for p in plans), "photos": sum(p["recoloured_images"] for p in plans),
                             "cost_inr": cost.get("with_retries_inr"), "disk_gb": round(sum(p["disk_gb"] for p in plans), 1)})
                body.pop("first_model", None)
                bodies.append(body)
            except (KeyError, TypeError, ValueError) as exc:
                problems.append(f"Skipped {b.get('category', 'a batch')} {', '.join(b.get('tops') or [])}: {exc}")
        total = {k: sum((r[k] or 0) for r in rows) for k in ("ai_images", "photos", "cost_inr")}
        return {"reply": ans.get("reply", ""), "batches": rows, "bodies": bodies, "total": total, "problems": problems}


def _specs(body: dict, ledger) -> list:
    from .server import specs_from

    return specs_from(body, ledger)


# ---------------------------------------------------------------- wiring
class Cloud:
    """Everything the server version adds, in one object the web server consults."""

    def __init__(self, data: Path, out_dir: Callable[[], str], password: Optional[str] = None,
                 drive: Optional[Drive] = None, agent: Optional[Agent] = None):
        self.data = Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        secret_file = self.data / "secret.key"
        if not secret_file.exists():
            secret_file.write_bytes(secrets.token_bytes(32))
            os.chmod(secret_file, 0o600)
        self.login = Login(password or os.environ.get("JAGIMAGE_PASSWORD", ""), secret_file.read_bytes())
        self.drive = drive or Drive(os.environ.get("GOOGLE_OAUTH_CLIENT_ID", ""), os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", ""),
                                    self.data, os.environ.get("JAGIMAGE_DRIVE_FOLDER", "kinsey models 2026"))
        self.sync = DriveSync(self.drive, out_dir)
        self.agent = agent or Agent()
        self.oauth_states = {}
        self.plans = {}  # plan id -> agent plan, until Start

    def drive_status(self) -> dict:
        return {"configured": self.drive.configured, "connected": self.drive.connected, "folder": self.drive.folder, **self.sync.status}
