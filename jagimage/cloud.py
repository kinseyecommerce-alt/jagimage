"""Jag Studio on a server (Google Cloud): a password login, finished photos uploaded to Google Drive,
and an agent that turns "make 1000 Men models in half sleeve and hoodie" into batches.

Enabled by `jagimage serve --cloud` with these environment variables:
  JAGIMAGE_PASSWORD              the first password of the user "admin" (required on the first start;
                                 later changed in Settings, where the admin also adds users)
  JAGIMAGE_RESET_ADMIN           set to 1 to put the admin's password back to JAGIMAGE_PASSWORD
  JAGIMAGE_DATA                  folder for settings and the Drive sign-in (default: <out>/../jag-data)
  GOOGLE_OAUTH_CLIENT_ID/SECRET  a Google OAuth "Web application" client, for Drive (optional)
  JAGIMAGE_DRIVE_FOLDER          Drive folder the photos go into (default: kinsey models 2026)
  JAGIMAGE_AGENT_MODEL           Gemini text model for the agent (default: gemini-flash-latest)
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .studio import DRIVE_LIST_LOCK

SESSION_DAYS = 30
# the app's own files and folders, plus reading (never changing) the print folders the shop points it at
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/drive.readonly"
READ_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER_MIME = "application/vnd.google-apps.folder"


# ---------------------------------------------------------------- users and login
USER_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
HASH_ROUNDS = 200_000


JPG_SIZE = re.compile(r"_\d+x\d+\.jpg$")  # the marketplace JPG (1500x1500, or a guide's size such as 1080x1440)

class Forbidden(Exception):
    """Signed in, but only an admin may do this."""


class Users:
    """Website accounts in users.json (data folder): name -> salted PBKDF2 hash, role ("admin" or "user")
    and a version that changes with the password, which signs out that user's old sessions.
    The first start makes the admin "admin" with JAGIMAGE_PASSWORD; after that the password is changed
    in Settings. JAGIMAGE_RESET_ADMIN=1 sets the admin's password back to JAGIMAGE_PASSWORD (forgotten)."""

    def __init__(self, data: Path, bootstrap_password: str = ""):
        self.file = Path(data) / "users.json"
        self.lock = threading.Lock()
        self.users = json.loads(self.file.read_text()) if self.file.exists() else {}
        if not self.users or os.environ.get("JAGIMAGE_RESET_ADMIN"):
            if not bootstrap_password or len(bootstrap_password) < 8:
                raise ValueError("JAGIMAGE_PASSWORD must be at least 8 characters")
            if "admin" in self.users:
                self.set_password("admin", bootstrap_password)
                self.users["admin"]["role"] = "admin"
            else:
                self.create("admin", bootstrap_password, "admin")

    def _save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.users, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.file)

    @staticmethod
    def _hash(password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), HASH_ROUNDS).hex()

    @staticmethod
    def _check_password(password: str):
        if not isinstance(password, str) or len(password) < 8:
            raise ValueError("The password needs at least 8 characters")

    def create(self, name: str, password: str, role: str = "user"):
        name = str(name or "").strip().lower()
        if not USER_NAME.match(name):
            raise ValueError("User name: 2-32 letters or digits (a-z, 0-9, dot, dash, underscore)")
        if role not in ("admin", "user"):
            raise ValueError("Role is admin or user")
        self._check_password(password)
        with self.lock:
            if name in self.users:
                raise ValueError(f"{name} already exists")
            salt = secrets.token_hex(16)
            # a new start number for every account, so a deleted and re-made account never accepts old sign-ins
            self.users[name] = {"hash": self._hash(password, salt), "salt": salt, "role": role, "ver": int(time.time() * 1000), "created": int(time.time())}
            self._save()

    def set_password(self, name: str, password: str):
        self._check_password(password)
        with self.lock:
            u = self.users[name]
            u["salt"] = secrets.token_hex(16)
            u["hash"] = self._hash(password, u["salt"])
            u["ver"] = u.get("ver", 1) + 1
            self._save()

    def delete(self, name: str):
        with self.lock:
            if name not in self.users:
                raise ValueError(f"No user {name}")
            if self.users[name]["role"] == "admin" and sum(u["role"] == "admin" for u in self.users.values()) == 1:
                raise ValueError("Keep at least one admin")
            del self.users[name]
            self._save()

    def check(self, name: str, password: str) -> bool:
        u = self.users.get(name)
        got = self._hash(password, u["salt"] if u else "00" * 16)  # same work for unknown names
        return bool(u) and hmac.compare_digest(got, u["hash"])

    def role(self, name: str) -> Optional[str]:
        u = self.users.get(name)
        return u and u["role"]

    def listing(self) -> list:
        return [{"name": n, "role": u["role"], "created": u.get("created")} for n, u in sorted(self.users.items())]


class Login:
    """Sign-in with a user name and password. A signed cookie keeps you signed in for SESSION_DAYS (until
    your password changes); repeated wrong passwords from one address are slowed down."""

    def __init__(self, users: Users, secret: bytes):
        self.users = users
        self.secret = secret
        self.failures = {}  # ip -> (count, last time)

    def _sign(self, value: str) -> str:
        return hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    def cookie(self, name: str) -> str:
        value = f"{int(time.time()) + SESSION_DAYS * 86400}.{name}.{self.users.users[name].get('ver', 1)}"
        return f"jag_session={value}.{self._sign(value)}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age={SESSION_DAYS * 86400}"

    def user(self, cookie_header: str) -> Optional[str]:
        """The signed-in user's name, or None."""
        for part in (cookie_header or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name != "jag_session" or value.count(".") < 3:
                continue
            body, sig = value.rsplit(".", 1)
            head, _, ver = body.rpartition(".")      # <expires>.<user name, may contain dots>.<version>
            expires, _, user = head.partition(".")
            u = self.users.users.get(user)
            if (hmac.compare_digest(sig, self._sign(body)) and expires.isdigit() and int(expires) > time.time()
                    and u and str(u.get("ver", 1)) == ver):
                return user
        return None

    def valid(self, cookie_header: str) -> bool:
        return self.user(cookie_header) is not None

    def try_login(self, ip: str, name: str, password: str) -> bool:
        if len(self.failures) > 5000:  # forget old addresses so the table can't grow without end
            cutoff = time.time() - 3600
            self.failures = {k: v for k, v in self.failures.items() if v[1] > cutoff}
        count, last = self.failures.get(ip, (0, 0.0))
        if count >= 5 and time.time() - last < 30 * (count - 4):
            raise PermissionError("Too many wrong passwords. Wait a minute and try again.")
        if self.users.check(str(name or "admin").strip().lower(), password):
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
        self.client_file = Path(data) / "drive_client.json"  # the OAuth client pasted in the website (if not in the environment)
        if not self.configured and self.client_file.exists():
            saved = json.loads(self.client_file.read_text())
            self.client_id, self.client_secret = saved.get("client_id", ""), saved.get("client_secret", "")
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

    def set_client(self, client_id: str, client_secret: str):
        """Keep the Google OAuth client the admin pasted in the website; a new client needs a new sign-in."""
        client_id, client_secret = str(client_id or "").strip(), str(client_secret or "").strip()
        if not client_id.endswith(".apps.googleusercontent.com") or len(client_secret) < 10:
            raise ValueError("Paste the Client ID (ends in .apps.googleusercontent.com) and the Client secret from Google Cloud")
        with self.lock:
            self.client_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.client_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"client_id": client_id, "client_secret": client_secret}))
            os.chmod(tmp, 0o600)
            tmp.replace(self.client_file)
            if client_id != self.client_id:
                self.state.pop("refresh_token", None)
                self._access = (None, 0.0)
                self._save()
            self.client_id, self.client_secret = client_id, client_secret

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
            self.state = {"refresh_token": tok["refresh_token"], "folders": {}, "scope": tok.get("scope", "")}
            self._access = (tok.get("access_token"), time.time() + int(tok.get("expires_in", 0)) - 60)
            self._save()

    @property
    def can_read(self) -> bool:
        """Connected with permission to read the shop's own folders (needed for print folders)."""
        return self.connected and READ_SCOPE in self.state.get("scope", "")

    def list_images(self, folder: str) -> list:
        """The pictures directly in a Drive folder: [{id, name, mimeType, modifiedTime}], by name."""
        files, page = [], None
        while True:
            q = {"q": f"'{folder}' in parents and trashed = false and mimeType contains 'image/'", "pageSize": 1000,
                 "fields": "nextPageToken, files(id, name, mimeType, modifiedTime)", "orderBy": "name",
                 "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}
            if page:
                q["pageToken"] = page
            r = self._call("GET", "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(q))
            files += r.get("files") or []
            page = r.get("nextPageToken")
            if not page:
                return files

    def download(self, file_id: str) -> bytes:
        req = urllib.request.Request(f"https://www.googleapis.com/drive/v3/files/{urllib.parse.quote(file_id)}?alt=media&supportsAllDrives=true",
                                     headers={"Authorization": f"Bearer {self._token()}"})
        with self.opener(req, timeout=120) as r:
            return r.read()

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
        """Upload a photo into its folder; a photo of the same name there (an earlier recolour) is replaced."""
        parent = self.folder_id(parts)
        q = f"name = '{name.replace(chr(39), chr(92) + chr(39))}' and mimeType != '{FOLDER_MIME}' and '{parent}' in parents and trashed = false"
        found = self._call("GET", "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
            {"q": q, "fields": "files(id)", "spaces": "drive"})).get("files") or []
        boundary = "jag" + secrets.token_hex(8)
        meta = json.dumps({"name": name} if found else {"name": name, "parents": [parent]}).encode()
        mime = "image/png" if local.suffix == ".png" else "image/jpeg"
        body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode() + meta +
                f"\r\n--{boundary}\r\nContent-Type: {mime}\r\n\r\n".encode() + local.read_bytes() + f"\r\n--{boundary}--\r\n".encode())
        if found:
            return self._call("PATCH", f"https://www.googleapis.com/upload/drive/v3/files/{found[0]['id']}?uploadType=multipart&fields=id",
                              body, f"multipart/related; boundary={boundary}")["id"]
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
            with DRIVE_LIST_LOCK:
                done = set(done_file.read_text().splitlines()) if done_file.exists() else set()
            for jpg in sorted(batch.glob("m[0-9][0-9][0-9][0-9]/*/*_*x*.jpg")):  # <model>/<Top__Bottom>/<pose>_<w>x<h>.jpg
                rel = jpg.relative_to(batch).as_posix()
                if rel not in done and not jpg.parent.name.startswith("_") and JPG_SIZE.search(jpg.name):
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
                with DRIVE_LIST_LOCK:
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
- shot: "top" (head to above the knee, bottom not recoloured) or "outfit" (head to shoes, both recoloured).
  Leave it out unless the user asks: Half Sleeve then gets "top", other tops "outfit". "full body", "head to
  toe" or "with pants/shorts recoloured" -> "outfit"; "top only", "above knee", "half body" -> "top"
- models: number of different people (1-5000); poses: subset of front_main, back, threequarter, side,
  front_main_white, handpocket (default front_main and back)
- "same model(s) in / can wear all top styles / all categories top / every top" means ONE batch whose tops
  list has all five: HalfSleeveTee, FullSleeveTee, SleevelessTee, Hoodie, Sweatshirt (a named style first),
  e.g. "10 mens half sleeve tshirt same model can wear all categories top" -> Men, all five tops, 10 models
- quality: "high" when the user asks for high or best quality, else "standard"
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
            "poses": {"type": "array", "items": {"type": "string"}}, "quality": {"type": "string"}},
            "required": ["category", "tops", "models"]}},
    },
    "required": ["reply", "batches"],
}


GUIDE_SYSTEM = """You read a clothing marketplace's guidelines for on-model catalogue photos (for example Myntra's
model photography guide) for SPB Textile's Jag Studio. The studio makes AI photos of a real-looking live model
wearing a plain, solid-colour garment (no print) on a plain studio backdrop; each photo is later recoloured to
the exact shade. Pull out only what the guide itself says; never invent rules it does not state.
- width, height: the final photo size in pixels it asks for (its stated size, or its minimum). If it gives
  only an aspect ratio, use 1500 px on the long side (e.g. 3:4 -> 1125 x 1500). 0 when it says nothing.
- background: the backdrop it asks for in plain words, with the hex code if it gives one (e.g. "pure white
  (#FFFFFF)", "light grey (#F2F2F2)"). Empty when it says nothing.
- framing: "full" when it wants full-length head-to-toe photos, "top" when it wants topwear shot from the head
  to the waist or knee, empty when it says nothing.
- poses: every on-model shot it asks for, in its order. name: a short English snake_case name such as front,
  back, side, three_quarter, walking, detail_closeup. direction: one plain sentence for the model and camera
  (body angle, head, arms, hands, expression) using only what the guide says. Leave out shots without a
  model (flat lay, ghost mannequin, fabric swatch, size chart): name those in not_applied.
- model: its directions about the model's look and manner (expression, makeup, hair, grooming, accessories,
  footwear, body language), as one short paragraph.
- avoid: what it forbids for the photo or the model, as a short comma-separated list.
- not_applied: its rules this app cannot follow by itself (file naming, file size, number of images,
  product-only shots, size charts, text overlays it allows, ...), briefly.
- marketplace: the marketplace's name, e.g. Myntra.
- summary: two to four short plain sentences for the shop owner saying what will be used."""

GUIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "marketplace": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"},
        "background": {"type": "string"}, "framing": {"type": "string"},
        "poses": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "direction": {"type": "string"}}, "required": ["name", "direction"]}},
        "model": {"type": "string"}, "avoid": {"type": "string"}, "not_applied": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["marketplace", "poses", "summary"],
}
GUIDE_TYPES = ".pdf, .docx, .txt, .md, .png, .jpg, .jpeg, .webp"

PRINT_CHECK_SYSTEM = """You are the print checker of a clothing studio. Image 1 is a catalogue photo of a model wearing a
T-shirt with a print placed on it by software; image 2 is the original print artwork. Judge only the print's
placement and look on the shirt, as a buyer on Myntra or Amazon would:
- centred on the chest (for a chest print), level, not riding onto the collar or neckband, not sliding under the
  armpits, not cut off by an arm or hand, a sensible size for the chest;
- the artwork itself intact: same design, colours and proportions as image 2, nothing missing or stretched;
- it looks printed on the fabric (follows its light and folds), not like a sticker floating on top.
Be strict: a print touching or overlapping the collar or neckband, cut by an arm, clearly off-centre, or covering
the whole front is not ok. You also get the software's measurements of the placement (shares of the chest width);
trust them for distances, and judge the look yourself. score: 1 (unusable) to 10 (perfect).
Then give a fix as a move and a resize: dx moves the print right (+) or left (-) and dy moves it down (+) or up (-),
both as a share of the chest width (0.05 = a small nudge, 0.15 = a lot); scale multiplies its size (0.9 = 10%
smaller). Use 0, 0, 1 when the placement is right. ok is true only when a buyer would accept the photo as it is.
Keep issues short and plain (e.g. "too high: touches the collar")."""

PRINT_CHECK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "score": {"type": "integer"},
                   "issues": {"type": "array", "items": {"type": "string"}},
                   "dx": {"type": "number"}, "dy": {"type": "number"}, "scale": {"type": "number"}},
    "required": ["ok", "issues", "dx", "dy", "scale"],
}

PRINT_PLAN_SYSTEM = """You set where a clothing studio's software places a print on T-shirt photos. Turn the shop's
words into a placement. place: "chest_large" (centred, big graphic), "chest_medium" (centred, medium) or
"left_chest" (small logo on the wearer's left chest). dx moves it right (+) or left (-) and dy down (+) or up (-),
as a share of the chest width (0.03 = "a little", 0.08 = "more", 0.15 = "a lot"); scale multiplies its size
(0.85 = "a bit smaller", 1.2 = "bigger"). Start from the current values given and change only what the words ask.
reply: one short plain sentence saying what you changed."""

PRINT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {"place": {"type": "string"}, "dx": {"type": "number"}, "dy": {"type": "number"},
                   "scale": {"type": "number"}, "reply": {"type": "string"}},
    "required": ["place", "dx", "dy", "scale", "reply"],
}
# a guide's shot names that mean one of the Studio's own poses (whose wording keeps e.g. a true rear view)
POSE_ALIASES = [("front", "front_main"), ("straight", "front_main"), ("back", "back"), ("rear", "back"),
                ("side", "side"), ("profile", "side"), ("three_quarter", "threequarter"), ("threequarter", "threequarter"),
                ("angle", "threequarter"), ("45", "threequarter"), ("pocket", "handpocket")]


def _on_white(im):
    from PIL import Image

    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.getchannel("A"))
    return bg


def drive_folder_id(link: str) -> str:
    """The folder id in a Google Drive folder link (or the id itself)."""
    link = link.strip()
    m = re.search(r"/folders/([A-Za-z0-9_-]{10,})", link) or re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", link) \
        or re.fullmatch(r"([A-Za-z0-9_-]{10,})", link)
    if not m:
        raise ValueError("Paste a Google Drive folder link, like https://drive.google.com/drive/folders/1AbC…")
    return m.group(1)


def guide_content(filename: str, data: bytes):
    """The guide's text, and up to six of its pictures (example photos, or the pages of a screenshot)."""
    import io
    import zipfile

    from PIL import Image

    def picture(raw):
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
        if min(im.size) < 120:  # icons and logos say nothing about the photo
            return None
        im.thumbnail((1280, 1280))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"

    ext = os.path.splitext(filename.lower())[1]
    text, pics = "", []
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:40])
        for page in reader.pages[:12]:
            for img in getattr(page, "images", [])[:3]:
                pics.append(picture(img.data))
    elif ext == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
            text = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))
            pics = [picture(z.read(n)) for n in z.namelist() if n.startswith("word/media/")][:12]
    elif ext in (".txt", ".md", ".csv"):
        text = data.decode("utf-8", "ignore")
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        pics = [picture(data)]
        if not pics[0]:
            raise ValueError("That picture can't be read")
    else:
        raise ValueError(f"Upload the guide as {GUIDE_TYPES}")
    text = re.sub(r"[ \t]+", " ", text).strip()[:40000]
    pics = [p for p in pics if p][:6]
    if not text and not pics:
        raise ValueError("Nothing readable in that file. If it is a scanned PDF, upload screenshots of its pages instead.")
    return text, pics


def clean_guide(ans: dict, filename: str) -> dict:
    """The AI's reading of a guide as Studio settings, every value checked."""
    from . import library as L

    def words(v, n):
        return " ".join(str(v or "").split())[:n]

    w, h = int(ans.get("width") or 0), int(ans.get("height") or 0)
    canvas = [w, h] if 200 <= w <= 6000 and 200 <= h <= 6000 else None
    framing = str(ans.get("framing") or "").lower()
    custom, poses = {}, []
    for p in (ans.get("poses") or [])[:8]:
        if not isinstance(p, dict):
            continue
        name = re.sub(r"[^a-z0-9_]+", "_", str(p.get("name") or "").lower()).strip("_")[:30]
        direction = words(p.get("direction"), 500).rstrip(".")
        if not name or not direction:
            continue
        ours = next((pose for word, pose in POSE_ALIASES if word in name), None)
        if ours in poses or name in poses:
            continue
        if ours:  # the Studio's own wording stays (it keeps e.g. a true rear view), the guide's direction is added
            custom[ours] = f"{L.POSES[ours]}; {direction}"
            poses.append(ours)
        else:
            custom[name] = direction
            poses.append(name)
    if "front_main" not in poses:
        poses.insert(0, "front_main")
    model, avoid = words(ans.get("model"), 600).rstrip("."), words(ans.get("avoid"), 300).rstrip(".")
    extra = (model + (". " if model and avoid else "") + (f"Avoid: {avoid}" if avoid else ""))[:1000]
    return {"marketplace": words(ans.get("marketplace"), 60) or os.path.splitext(filename)[0][:60],
            "canvas": canvas, "background": words(ans.get("background"), 200),
            "shot": {"full": "outfit", "top": "top"}.get(framing, ""), "poses": poses, "custom_poses": custom,
            "extra": extra, "not_applied": words(ans.get("not_applied"), 600), "summary": words(ans.get("summary"), 800),
            "file": filename[:120]}


class Agent:
    """Turns a request in plain words into Studio batch requests with their cost. It only plans:
    nothing is made until the person presses Start on the plan."""

    OPENAI_MODELS = ("gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini")  # first one the account can use

    def __init__(self, client=None, model: Optional[str] = None, openai_client=None):
        self.client, self.openai_client = client, openai_client
        self.model = model or os.environ.get("JAGIMAGE_AGENT_MODEL", "gemini-flash-latest")
        self.used = None  # "gemini" or "openai": which one answered last

    def _ask(self, message: str, system: str = AGENT_SYSTEM, schema: dict = AGENT_SCHEMA, images=()) -> dict:
        """Plans with whichever AI has a key: Gemini first, OpenAI when there is no Gemini key or Gemini
        can't be used (e.g. out of credits)."""
        gemini = self.client is not None or bool(os.environ.get("GEMINI_API_KEY"))
        openai = self.openai_client is not None or bool(os.environ.get("OPENAI_API_KEY"))
        if not gemini and not openai:
            raise RuntimeError("The agent needs an AI key: add a Gemini or OpenAI key in Settings → API keys")
        if gemini:
            try:
                ans = self._ask_gemini(message, system, schema, images)
                self.used = "gemini"
                return ans
            except Exception:
                if not openai:
                    raise
        ans = self._ask_openai(message, system, schema, images)
        self.used = "openai"
        return ans

    def _ask_gemini(self, message: str, system: str = AGENT_SYSTEM, schema: dict = AGENT_SCHEMA, images=()) -> dict:
        from google.genai import types

        client = self.client
        if client is None:
            from google import genai

            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model=self.model, contents=[types.Part.from_bytes(data=d, mime_type=m) for d, m in images] + [message] if images else message,
            config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                               response_schema=schema, temperature=0.1))
        return json.loads(resp.text)

    def _ask_openai(self, message: str, system: str = AGENT_SYSTEM, schema: dict = AGENT_SCHEMA, images=()) -> dict:
        client = self.openai_client
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        system = system + "\nAnswer with one JSON object only, shaped like this JSON schema:\n" + json.dumps(schema)
        content = message if not images else [{"type": "text", "text": message}] + [
            {"type": "image_url", "image_url": {"url": f"data:{m};base64," + base64.b64encode(d).decode()}} for d, m in images]
        models = [os.environ["JAGIMAGE_AGENT_OPENAI_MODEL"]] if os.environ.get("JAGIMAGE_AGENT_OPENAI_MODEL") else self.OPENAI_MODELS
        last = None
        for model in models:
            try:
                resp = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                                      messages=[{"role": "system", "content": system}, {"role": "user", "content": content}])
                return json.loads(resp.choices[0].message.content)
            except Exception as exc:  # a model this account can't use: try the next one
                last = exc
                if "model" not in str(exc).lower():
                    raise
        raise last

    def plan(self, message: str, ledger, base: dict) -> dict:
        """base: the engine settings to use (provider, gemini_model, image_size, batch, one_base ...)."""
        from . import library as L
        from . import pricing, studio

        ans = self._ask(message)
        bodies, rows, problems = [], [], []
        if not isinstance(ans, dict):
            ans = {}
        for b in ans.get("batches") or [] if isinstance(ans.get("batches"), list) else []:
            if not isinstance(b, dict):  # the AI answered something that isn't a batch
                problems.append("Skipped an unreadable part of the plan")
                continue
            listed = lambda key: [v for v in (b.get(key) or []) if isinstance(v, str)] if isinstance(b.get(key), list) else []
            try:
                tops = [t for t in listed("tops") if t in studio.TOPS] or ["HalfSleeveTee"]
                category = b["category"]
                if not isinstance(category, str) or category not in L.CATEGORIES:
                    raise ValueError(f"unknown category {category!r}")
                fabric = b.get("fabric") if b.get("fabric") in L.FABRICS else "PureCotton"
                colours = [c for c in listed("top_colours") if c in L.COLOURS[fabric]] or list(L.COLOURS[fabric])
                bottom = b.get("bottom") if b.get("bottom") in studio.BOTTOMS else ("Shorts" if L.is_kids(category) else "Trackpants")
                poses = [p for p in listed("poses") if p in L.POSES] or ["front_main", "back"]
                if "front_main" not in poses:
                    poses = ["front_main"] + poses
                engine = dict(base)
                if b.get("quality") == "high":  # "high quality": OpenAI high, or Gemini Pro
                    engine.update(quality="high") if engine.get("provider") == "openai" else engine.update(gemini_model="pro")
                body = {**engine, "category": category, "top": tops[0], "tops": tops[1:], "bottom": bottom, "fabric": fabric,
                        "top_colours": colours, "pairing": b.get("pairing") if b.get("pairing") in ("match", "each") else "match",
                        "shot": b.get("shot") if b.get("shot") in studio.SHOTS else studio.default_shot(tops[0]),
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
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                problems.append(f"Skipped {b.get('category', 'a batch')} {', '.join(listed('tops'))}: {exc}")
        total = {k: sum((r[k] or 0) for r in rows) for k in ("ai_images", "photos", "cost_inr")}
        return {"reply": str(ans.get("reply") or ""), "batches": rows, "bodies": bodies, "total": total, "problems": problems}

    def check_print(self, photo, art, measured: Optional[dict] = None) -> dict:
        """The print checker: is the print on this photo placed right? {ok, score, issues, dx, dy, scale}
        with the move and resize that would fix it (kept within jagimage.prints.LIMITS)."""
        import io

        from .prints import adjust

        def jpeg(im, side):
            im = im.convert("RGB") if im.mode != "RGBA" else _on_white(im)
            im.thumbnail((side, side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            return buf.getvalue(), "image/jpeg"

        facts = ("" if not measured else "\nMeasured by the software: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in measured.items())
                 + " (gaps, width and off-centre as shares of the chest width; hidden as a share of the print).")
        ans = self._ask("Check the print placement on image 1 (image 2 is the artwork)." + facts, PRINT_CHECK_SYSTEM,
                        PRINT_CHECK_SCHEMA, [jpeg(photo, 1024), jpeg(art, 512)])
        fix = adjust(ans.get("dx"), ans.get("dy"), ans.get("scale"))
        issues = [" ".join(str(i).split())[:160] for i in (ans.get("issues") or []) if str(i).strip()][:5]
        score = max(0, min(10, int(ans.get("score") or 0)))
        return {"ok": bool(ans.get("ok")), "score": score, "issues": issues, **fix}

    def plan_print(self, message: str, current: dict) -> dict:
        """The shop's words ("a bit lower and smaller") as a placement: {place, dx, dy, scale, reply}."""
        from .prints import PLACES, adjust

        ans = self._ask(f"Current placement: {json.dumps(current)}\nThe shop says: {message}", PRINT_PLAN_SYSTEM, PRINT_PLAN_SCHEMA)
        place = ans.get("place") if ans.get("place") in PLACES else current.get("place", "chest_large")
        return {"place": place, **adjust(ans.get("dx"), ans.get("dy"), ans.get("scale")),
                "reply": " ".join(str(ans.get("reply") or "").split())[:300]}

    def read_guide(self, filename: str, data: bytes) -> dict:
        """A marketplace's model photo guidelines (e.g. Myntra's) as Studio settings: photo size,
        backdrop, framing, the poses it asks for, and its directions for the model."""
        text, images = guide_content(filename, data)
        ans = self._ask(f"Guideline file: {filename}\n\n{text}" if text else f"Guideline file: {filename} (see the images)",
                        GUIDE_SYSTEM, GUIDE_SCHEMA, images)
        return clean_guide(ans, filename)


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
        self.users = Users(self.data, password or os.environ.get("JAGIMAGE_PASSWORD", ""))
        self.login = Login(self.users, secret_file.read_bytes())
        self.drive = drive or Drive(os.environ.get("GOOGLE_OAUTH_CLIENT_ID", ""), os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", ""),
                                    self.data, os.environ.get("JAGIMAGE_DRIVE_FOLDER", "kinsey models 2026"))
        self.sync = DriveSync(self.drive, out_dir)
        self.agent = agent or Agent()
        self.oauth_states = {}
        self.plans = {}  # plan id -> agent plan, until Start

    def drive_status(self) -> dict:
        return {"configured": self.drive.configured, "connected": self.drive.connected, "can_read": self.drive.can_read,
                "folder": self.drive.folder, **self.sync.status}

    # ---- print folders: which Drive folder holds the prints for a category, top style and colour
    def print_rows(self) -> list:
        f = self.data / "prints.json"
        return json.loads(f.read_text()) if f.exists() else []

    def set_print_rows(self, rows) -> list:
        from . import library as L
        from . import studio

        clean = []
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                raise ValueError("send a list of print folders")
            cat, top, colour = str(r.get("category", "")), str(r.get("top", "")), " ".join(str(r.get("colour") or "").split())
            if cat not in L.CATEGORIES or top not in studio.TOPS:
                raise ValueError(f"Pick a category and top style for every print folder ({cat or '?'} / {top or '?'})")
            folder = drive_folder_id(str(r.get("link", "")))
            clean.append({"category": cat, "top": top, "colour": colour[:40], "link": str(r.get("link", "")).strip()[:300], "folder": folder})
        if len(clean) > 500:
            raise ValueError("At most 500 print folders")
        f = self.data / "prints.json"
        f.write_text(json.dumps(clean, indent=1))
        return clean

    def print_files(self, category: str, top: str, colour: str) -> list:
        """[(print name, local PNG path)] for one shirt: the folder set for this colour, else the one for
        all colours of this category and top style. Downloaded once and kept until it changes on Drive."""
        rows = [r for r in self.print_rows() if r["category"] == category and r["top"] == top]
        row = next((r for r in rows if r["colour"].lower() == colour.lower()), None) or next((r for r in rows if not r["colour"]), None)
        if not row:
            return []
        if not self.drive.can_read:
            raise RuntimeError("Reconnect Google Drive (Settings → Google Drive → Connect) so the app may read your print folders")
        try:
            files = self.drive.list_images(row["folder"])
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Can't open the print folder for {category} {top} {colour or 'all colours'} (Drive says {exc.code}): "
                               "check the link, and that the folder is shared with the Google account connected here") from exc
        cache = self.data / "print_cache"
        cache.mkdir(exist_ok=True)
        out = []
        for f in files:
            local = cache / f"{f['id']}_{hashlib.sha1(f.get('modifiedTime', '').encode()).hexdigest()[:8]}{os.path.splitext(f['name'])[1].lower() or '.png'}"
            if not local.exists():
                tmp = local.with_suffix(".part")
                tmp.write_bytes(self.drive.download(f["id"]))
                tmp.replace(local)
            out.append((os.path.splitext(f["name"])[0], local))
        return out

    # ---- marketplace guides (e.g. Myntra's model photo guidelines), read once and kept for everyone
    def guides(self) -> list:
        d = self.data / "guides"
        rows = []
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            try:
                rows.append(json.loads(f.read_text()))
            except ValueError:
                continue
        return sorted(rows, key=lambda g: -g.get("saved", 0))

    def save_guide(self, guide: dict) -> dict:
        d = self.data / "guides"
        d.mkdir(exist_ok=True)
        guide = {**guide, "id": re.sub(r"[^a-z0-9]+", "-", guide["marketplace"].lower()).strip("-") or "guide", "saved": time.time()}
        (d / f"{guide['id']}.json").write_text(json.dumps(guide, indent=1))
        return guide

    def delete_guide(self, gid: str):
        f = self.data / "guides" / f"{re.sub(r'[^a-z0-9-]+', '', gid.lower())}.json"
        f.unlink(missing_ok=True)
