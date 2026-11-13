"""Jag Studio on a server (Google Cloud): a password login, finished photos uploaded to Google Drive,
and an agent that turns "make 1000 Men models in half sleeve and hoodie" into batches.

Enabled by `jagimage serve --cloud` with these environment variables:
  JAGIMAGE_PASSWORD              the first password of the user "admin" (required on the first start;
                                 later changed in Settings, where the admin also adds users)
  JAGIMAGE_RESET_ADMIN           set to 1 to put the admin's password back to JAGIMAGE_PASSWORD
  JAGIMAGE_AGENT_PASSWORD        while set (16+ characters), an admin account "claude" for an AI assistant
                                 with this password; remove the variable and the account is removed
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
ASSISTANT = "claude"  # the account JAGIMAGE_AGENT_PASSWORD switches on


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
        self._assistant(os.environ.get("JAGIMAGE_AGENT_PASSWORD", ""))

    def _assistant(self, password: str):
        """The "claude" account follows JAGIMAGE_AGENT_PASSWORD: made (or its password updated) while the
        variable is set, removed when it is gone. An account of that name made by hand is left alone."""
        u = self.users.get(ASSISTANT)
        if password and len(password) >= 16:
            if u is None:
                self.create(ASSISTANT, password, "admin")
                self.users[ASSISTANT]["from_env"] = True
                self._save()
            elif u.get("from_env") and not self.check(ASSISTANT, password):
                self.set_password(ASSISTANT, password)
        elif u is not None and u.get("from_env"):
            with self.lock:
                del self.users[ASSISTANT]
                self._save()

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

    def list_tree(self, folder: str, depth: int = 5, limit: int = 20000) -> list:
        """Every picture in a Drive folder and its subfolders (to depth levels): [{id, name, modifiedTime, path}],
        path being the subfolder names from the given folder down."""
        out, todo = [], [(folder, [])]
        while todo:
            fid, path = todo.pop(0)
            page = None
            while True:
                q = {"q": f"'{fid}' in parents and trashed = false and (mimeType contains 'image/' or "
                          "mimeType = 'application/vnd.google-apps.folder')", "pageSize": 1000,
                     "fields": "nextPageToken, files(id, name, mimeType, modifiedTime)", "orderBy": "name",
                     "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}
                if page:
                    q["pageToken"] = page
                r = self._call("GET", "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(q))
                for f in r.get("files") or []:
                    if f.get("mimeType") == "application/vnd.google-apps.folder":
                        if len(path) < depth:
                            todo.append((f["id"], path + [f["name"]]))
                    else:
                        out.append({"id": f["id"], "name": f["name"], "modifiedTime": f.get("modifiedTime", ""), "path": path})
                        if len(out) >= limit:
                            return out
                page = r.get("nextPageToken")
                if not page:
                    break
        return out

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
        try:
            with self.opener(req, timeout=120) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:  # keep Google's own explanation: "Bad Request" alone says nothing
            try:
                why = json.loads(exc.read() or b"{}").get("error", {}).get("message", "")
            except Exception:
                why = ""
            exc.msg = f"{exc.msg}: {why} [{method} {url.split('?')[0][-80:]}]" if why else exc.msg
            raise

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

    def fetch(self, parts: list, name: str) -> bytes:
        """A file this app uploaded to <photos folder>/<parts>/<name>."""
        parent = self.folder_id(parts)
        q = f"name = '{name.replace(chr(39), chr(92) + chr(39))}' and '{parent}' in parents and trashed = false"
        found = self._call("GET", "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
            {"q": q, "fields": "files(id)", "spaces": "drive"})).get("files") or []
        if not found:
            raise FileNotFoundError(f"{'/'.join(parts)}/{name} is not on Drive")
        return self.download(found[0]["id"])

    def top_items(self) -> list:
        """What is directly inside the photos folder: [{id, name, folder}]."""
        root = self.folder_id([])
        items, token = [], None
        while True:
            q = {"q": f"'{root}' in parents and trashed = false", "fields": "nextPageToken, files(id, name, mimeType)",
                 "pageSize": 1000, "spaces": "drive", **({"pageToken": token} if token else {})}
            r = self._call("GET", "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(q))
            items += [{"id": f["id"], "name": f["name"], "folder": f["mimeType"] == FOLDER_MIME} for f in r.get("files") or []]
            token = r.get("nextPageToken")
            if not token:
                return sorted(items, key=lambda f: f["name"])

    def trash(self, names: list, keep=()) -> list:
        """Moves the named items of the photos folder to Drive's trash (restorable there for 30 days); never the
        ones in `keep`. Returns the names trashed."""
        done, self.trash_refused = [], {}
        items = self.top_items()
        for path in [n for n in names if "/" in n]:  # a deeper folder: "Pure Cotton/Men/Navy/Co-ords"
            *parents, last = [p for p in path.split("/") if p]
            key = "/".join([self.folder] + parents)
            parent = self.state.get("folders", {}).get(key)
            if not parent:
                continue  # never made by this app here
            q = f"name = '{last.replace(chr(39), chr(92) + chr(39))}' and '{parent}' in parents and trashed = false"
            for f in self._call("GET", "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(
                    {"q": q, "fields": "files(id, name, mimeType)", "spaces": "drive"})).get("files") or []:
                items.append({"id": f["id"], "name": path, "folder": f["mimeType"] == FOLDER_MIME})
        for item in items:
            if item["name"] in names and item["name"] not in keep:
                try:
                    self._call("PATCH", f"https://www.googleapis.com/drive/v3/files/{item['id']}?fields=id",
                               json.dumps({"trashed": True}).encode())
                except urllib.error.HTTPError as exc:  # e.g. only the folder's owner may trash it: say why, go on
                    try:
                        why = json.loads(exc.read() or b"{}").get("error", {}).get("message", "")
                    except Exception:
                        why = ""
                    self.trash_refused[item["name"]] = f"{exc.code} {why}".strip()
                    continue
                done.append(item["name"])
        with self.lock:  # folders inside a trashed one are made again if something is uploaded there later
            known = self.state.setdefault("folders", {})
            for name in done:
                prefix = f"{self.folder}/{name}"
                for key in [k for k in known if k == prefix or k.startswith(prefix + "/")]:
                    del known[key]
            self._save()
        return done

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
    """Every minute, uploads finished marketplace JPGs that aren't on Drive yet: as
    <photos folder>/<batch>/<Top__Bottom>/<model>_<pose>.jpg, or for batches with drive_layout "catalogue" in
    the shop's folders (jagimage.catalogue), single tops cut on the way. With free_after_upload a JPG leaves the
    server once all its files are on Drive. Survives restarts: what is uploaded is listed in each batch folder's
    _drive.txt, what was taken off the server in _freed.txt."""

    def __init__(self, drive: Drive, out_dir: Callable[[], str], every: float = 60, workers: int = 8, chunk: int = 96):
        self.drive, self.out_dir, self.every = drive, out_dir, every
        self.workers, self.chunk = workers, chunk  # uploads at once; photos sent between frees of the server
        self.status = {"uploaded": 0, "pending": 0, "freed": 0, "last_error": None, "last_run": None}
        self.wake = threading.Event()
        self.lock = threading.Lock()

    @staticmethod
    def _batch(root: Path):
        from . import studio

        try:
            spec = studio.Spec(**json.loads((root / "spec.json").read_text()))
        except Exception:  # an old or broken spec: the plain batch layout
            return None
        return studio.Batch(spec, out=root.parent)

    def pending(self) -> list:
        """(batch folder, key, jpg, upload info) still to send; info None for the plain batch layout. In a batch
        with the QC agent, only photos that passed it (or that a person approved) are sent."""
        from . import catalogue, qcagent

        rows = []
        for spec in Path(self.out_dir()).glob("*/spec.json"):
            root = spec.parent
            done_file = root / "_drive.txt"
            with DRIVE_LIST_LOCK:
                done = set(done_file.read_text().splitlines()) if done_file.exists() else set()
            b = self._batch(root)
            placed = set()
            gated = b is not None and b.spec.qc_agent  # the QC agent decides what may go

            def ok(jpg):
                return not gated or qcagent.passed(jpg.parent / "qc.json", jpg.name.rsplit("_", 1)[0])
            if b is not None and b.spec.drive_layout == "catalogue":
                for f in catalogue.files(b):
                    placed.add(f["rel"])
                    if f["key"] not in done and ok(f["src"]):
                        rows.append((root, f["key"], f["src"], f))
            for jpg in sorted(root.glob("m[0-9][0-9][0-9][0-9]/*/*_*x*.jpg")):  # <model>/<Top__Bottom>/<pose>_<w>x<h>.jpg
                rel = jpg.relative_to(root).as_posix()
                if rel not in placed and rel not in done and not jpg.parent.name.startswith("_") and JPG_SIZE.search(jpg.name) and ok(jpg):
                    rows.append((root, rel, jpg, None))
        return rows

    def _send(self, root: Path, key: str, jpg: Path, info) -> None:
        if info is None:
            model, pair = jpg.parent.parent.name, jpg.parent.name
            self.drive.upload(jpg, [root.name, pair], f"{model}_{jpg.name.rsplit('_', 1)[0]}.jpg")
            return
        if not info["crop"]:
            self.drive.upload(jpg, info["parts"], info["name"])
            return
        import tempfile

        from . import catalogue

        with tempfile.TemporaryDirectory() as tmp:  # the single top is cut now, never stored on the server
            single = Path(tmp) / info["name"]
            single.write_bytes(catalogue.single_top(self._batch(root), info))
            self.drive.upload(single, info["parts"], info["name"])

    def _free(self, root: Path) -> int:
        """Takes a batch's JPGs off the server once every file made from them is on Drive."""
        b = self._batch(root)
        if b is None or not b.spec.free_after_upload:
            return 0
        from . import catalogue

        with DRIVE_LIST_LOCK:
            done_file = root / "_drive.txt"
            done = set(done_file.read_text().splitlines()) if done_file.exists() else set()
        keys: dict = {}
        if b.spec.drive_layout == "catalogue":
            for f in catalogue.files(b):
                keys.setdefault(f["rel"], []).append(f["key"])
        for jpg in root.glob("m[0-9][0-9][0-9][0-9]/*/*_*x*.jpg"):
            rel = jpg.relative_to(root).as_posix()
            keys.setdefault(rel, [rel])
        gone = [rel for rel, ks in keys.items() if all(k in done for k in ks) and (root / rel).exists()]
        if gone:
            with DRIVE_LIST_LOCK:
                with open(root / "_freed.txt", "a") as fh:
                    fh.writelines(f"{rel}\n" for rel in gone)
            for rel in gone:
                (root / rel).unlink(missing_ok=True)
        return len(gone)

    def run_once(self) -> int:
        if not self.drive.connected:
            return 0
        with self.lock:
            try:  # bases of finished batches leave first: a full disk would stop everything below
                self._archive_bases()
            except OSError as exc:
                self.status["bases_error"] = f"{type(exc).__name__}: {exc}"[:400]
            todo = self.pending()
            self.status.update(pending=len(todo), last_run=round(time.time()))
            sent, skipped, failed = 0, None, None
            from concurrent.futures import ThreadPoolExecutor

            def one(row):
                root, key, jpg, info = row
                try:
                    self._send(root, key, jpg, info)
                    return row, None
                except FileNotFoundError as exc:  # this photo (or its base) went meanwhile: skip it, send the rest
                    return row, ("skip", f"skipped: {type(exc).__name__}: {exc}"[:300])
                except Exception as exc:  # network, quota, revoked access: try again next round
                    return row, ("fail", f"{root.name}/{key}: {type(exc).__name__}: {exc}"[:400])
            # several uploads at once, and finished photos leave the server every chunk (not only after hours)
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for i in range(0, len(todo), self.chunk):
                    roots = set()
                    for (root, key, jpg, info), err in pool.map(one, todo[i:i + self.chunk]):
                        if err:
                            skipped, failed = (err[1], failed) if err[0] == "skip" else (skipped, err[1])
                            continue
                        with DRIVE_LIST_LOCK:
                            with open(root / "_drive.txt", "a") as f:
                                f.write(key + "\n")
                        sent += 1
                        roots.add(root)
                        self.status["uploaded"] += 1
                        self.status["pending"] = len(todo) - sent
                    for root in roots:
                        self.status["freed"] += self._free(root)
                    if failed:
                        break
            self.status["last_error"] = failed or skipped
            self._archive_bases(todo)
            return sent

    def _archive_bases(self, todo=None):
        """A finished batch's base photos go to Drive (<photos folder>/_bases/<batch>/...) and leave the server once
        every photo made from them, its fabric twins' too, is on Drive: the disk holds only batches in progress."""
        from . import qcagent, studio

        def waiting(r: Path) -> bool:
            """A finished photo still to be uploaded. Photos held for review don't count: they keep their JPG on the
            server, and a base that is needed again (to recolour them afresh) comes back from Drive."""
            gated = (b := self._batch(r)) is not None and b.spec.qc_agent
            return any(not gated or qcagent.passed(j.parent / "qc.json", j.name.rsplit("_", 1)[0])
                       for j in r.glob("m[0-9][0-9][0-9][0-9]/*/*_*x*.jpg"))
        roots = {spec.parent for spec in Path(self.out_dir()).glob("*/spec.json")}
        for root in sorted(roots):
            b = self._batch(root)
            if b is None or b.spec.base_from or not b.spec.free_after_upload or not (root / "_complete").exists():
                continue
            twins = [studio.fabric_twin(b.spec, f).name for f in b.spec.also_fabrics]
            if waiting(root) or any(waiting(root.parent / t) or not (root.parent / t / "_complete").exists() for t in twins):
                continue
            for png in sorted(root.glob("m[0-9][0-9][0-9][0-9]/_base/*/*.png")):
                if "_mask" in png.name or ".tmp." in png.name or studio.archived(png):
                    continue  # masks stay; a .tmp file is still being written
                try:
                    self.drive.upload(png, ["_bases", root.name, png.parent.parent.parent.name, png.parent.name], png.name)
                except FileNotFoundError:  # it went meanwhile (e.g. made again): nothing to keep
                    continue
                except Exception as exc:  # try again next round
                    self.status["bases_error"] = f"{png.relative_to(root.parent)}: {type(exc).__name__}: {exc}"[:400]
                    return
                marker = Path(str(png) + studio.ARCHIVED)
                try:
                    marker.write_text(json.dumps({"time": round(time.time())}))
                except OSError:  # a full disk: the base is safe on Drive, so free its space and write the marker then
                    png.unlink()
                    marker.write_text(json.dumps({"time": round(time.time())}))
                png.unlink(missing_ok=True)
                self.status["bases_archived"] = self.status.get("bases_archived", 0) + 1

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
def agent_system() -> str:
    """The planning instructions with the shop's colour lists as they are now (edited in Settings)."""
    from . import library as L

    parts = []
    for fabric, lists in L.CATEGORY_COLOURS.items():
        groups: dict = {}
        for cat in L.CATEGORIES:  # categories with the same list are named together
            groups.setdefault(tuple(lists[cat]), []).append(cat)
        parts.append(f"{fabric} (colours: " + "; ".join(f"{' and '.join(cats)}: {', '.join(names)}" for names, cats in groups.items()) + ")")
    return AGENT_SYSTEM.replace("{FABRIC_COLOURS}", " or\n  ".join(parts))


AGENT_SYSTEM = """You plan catalogue photo batches for SPB Textile's Jag Studio. Turn the user's request
into batches. Each batch is one category with one or more top styles; all styles reuse the same models.
Allowed values:
- category: Men, Women, Kids_Boys, Kids_Girls
- tops: HalfSleeveTee, FullSleeveTee, SleevelessTee, Hoodie, Sweatshirt, OversizedTee (oversized drop-shoulder tee:
  Men and Women only, never for kids), ZipHoodie (kids full-zip hoodie: Kids_Boys and Kids_Girls only, never adults)
- bottom: Shorts, Trackpants, Jeans (Men/Women default Trackpants, kids default Shorts)
- fabric: {FABRIC_COLOURS}
- top_colours: colour names from that fabric for the category; default all of them
- pairing: "match" (bottom in the same colour as the top, default) or "each" (every top with every bottom colour)
- shot: "top" (head to above the knee, bottom not recoloured) or "outfit" (head to shoes, both recoloured).
  Leave it out unless the user asks: Half Sleeve and Oversized then get "top", other tops "outfit". "full body", "head to
  toe" or "with pants/shorts recoloured" -> "outfit"; "top only", "above knee", "half body" -> "top"
- models: number of different people (1-5000); poses: subset of front_main, back, threequarter, side,
  front_main_white, handpocket (default front_main and back)
- "same model(s) in / can wear all top styles / all categories top / every top" means ONE batch whose tops
  list has every style: HalfSleeveTee, FullSleeveTee, SleevelessTee, Hoodie, Sweatshirt, plus OversizedTee for Men
  and Women or ZipHoodie for kids (a named style first), e.g. "10 mens half sleeve tshirt same model can wear all
  categories top" -> Men, all six adult tops, 10 models; for kids the six with ZipHoodie and without OversizedTee
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


GUIDE_SYSTEM = """You read a brand's or a clothing marketplace's guidelines for on-model catalogue photos (for example a
brand's image direction deck for Myntra, or Myntra's or Amazon's own model photography guide) for SPB Textile's Jag
Studio. The studio makes AI photos of a real-looking live model wearing a plain, solid-colour garment (no print) on
a plain studio backdrop; each photo is later recoloured to the exact shade. These are the default (list view)
catalogue photos: lifestyle, outdoor or location shots are not made here. Pull out only what the guide itself says;
never invent rules it does not state.
- width, height: the final photo size in pixels it asks for (its stated size, or its minimum). If it gives
  only an aspect ratio, use 1500 px on the long side (e.g. 3:4 -> 1125 x 1500). 0 when it says nothing.
- background: the studio backdrop it asks for in the default / list-view photos, in plain words, with the hex code
  if it gives one (e.g. "pure white (#FFFFFF)", "warm beige (#EFE6DA) seamless"). When it allows several, pick
  the first plain one it names. Never a lifestyle, outdoor or room setting. Empty when it says nothing.
- framing: "full" when it wants full-length head-to-toe photos, "top" when it wants topwear shot from the head
  to the waist or knee (cropped to the product), empty when it says nothing.
- poses: every on-model shot it asks for, in its order. name: a short English snake_case name such as front,
  back, side, three_quarter, walking, look_shot, detail_closeup. direction: one plain sentence for the model and
  camera (body angle, head, arms, hands, expression) using only what the guide says, including its pose and
  expression rules (e.g. relaxed and candid, not stiff, not smiling widely). Leave out shots without a model
  (flat lay, ghost mannequin, fabric swatch on its own, size chart): name those in not_applied.
- model: its directions about the model's look and manner that suit a model of any age (expression, attitude,
  makeup, hair, grooming, skin tone, accessories, footwear, body language, styling), as one short paragraph.
- adult_model: its rules that only fit adult models (age limits such as "under 25", height, shoulder, chest,
  waist and hip measurements, body type such as lean and athletic), one short paragraph, for men and women
  separately when it gives both (e.g. "Men: ... Women: ..."). Empty when it says nothing. These are never
  applied to kids.
- avoid: what it forbids for the photo or the model, as a short comma-separated list.
- not_applied: its rules this app cannot follow by itself (file naming, file size, number of images, lifestyle
  or outdoor shots, product-only shots, size charts, approved studios, text overlays it allows, ...), briefly.
- marketplace: the brand's name and the marketplace, e.g. "Roadster (Myntra)", or the marketplace alone.
- summary: two to four short plain sentences for the shop owner saying what will be used."""

GUIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "marketplace": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"},
        "background": {"type": "string"}, "framing": {"type": "string"},
        "poses": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "direction": {"type": "string"}}, "required": ["name", "direction"]}},
        "model": {"type": "string"}, "adult_model": {"type": "string"}, "avoid": {"type": "string"},
        "not_applied": {"type": "string"},
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
How print shops place a front print (judge against this, the way a print-shop owner would): every design starts
on the same line below the collar: about 2-2.5 inches on kids' and youth tees, 3-4 inches on adults' (roughly a
quarter of the chest width below the neckline on a child, a fifth on an adult). Its width follows the wearer:
about 5-6 inches on a 4-5 year old's tee (just under half the chest width), 8-9 inches youth, 10-12 inches adult
(about half the chest). Its shape decides the rest: wide text or a badge spans that width and is short; a square
graphic is a little narrower (it looks heavier); a tall graphic is held by its height so it never runs to the
belly or the hem. A small logo-like design belongs on the left chest, not in the centre. The design's visual
middle sits on the chest, never in the neck hollow or on the stomach.
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

    from .prints import PrintError, open_image

    def picture(raw):
        try:
            im = open_image(raw).convert("RGB")
        except PrintError:
            return None
        if min(im.size) < 120:  # icons and logos say nothing about the photo
            return None
        im.thumbnail((1280, 1280))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"

    ext = os.path.splitext(filename.lower())[1]
    text, pics = "", []
    try:
        if ext == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages[:40])[:400_000]
            for page in reader.pages[:12]:
                for img in getattr(page, "images", [])[:3]:
                    pics.append(picture(img.data))
        elif ext == ".docx":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                # sizes are checked before unpacking: a small .docx can unpack to gigabytes (a "zip bomb")
                doc = z.getinfo("word/document.xml")
                if doc.file_size > 20_000_000:
                    raise ValueError("That Word file's text is too large to read")
                xml = z.read(doc).decode("utf-8", "ignore")
                text = re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))
                media = [i for i in z.infolist() if i.filename.startswith("word/media/") and i.file_size <= 30_000_000][:12]
                pics = [picture(z.read(i)) for i in media]
        elif ext in (".txt", ".md", ".csv"):
            text = data.decode("utf-8", "ignore")
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            pics = [picture(data)]
            if not pics[0]:
                raise ValueError("That picture can't be read (or is too large)")
        else:
            raise ValueError(f"Upload the guide as {GUIDE_TYPES}")
    except ValueError:
        raise
    except Exception as exc:  # a damaged PDF or Word file
        raise ValueError(f"That file can't be read ({type(exc).__name__}): is it a damaged {ext or 'file'}?") from exc
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
            "extra": extra, "adult_extra": words(ans.get("adult_model"), 600).rstrip("."),
            "not_applied": words(ans.get("not_applied"), 600), "summary": words(ans.get("summary"), 800),
            "file": filename[:120]}


class Agent:
    """Turns a request in plain words into Studio batch requests with their cost. It only plans:
    nothing is made until the person presses Start on the plan."""

    OPENAI_MODELS = ("gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini")  # first one the account can use

    def __init__(self, client=None, model: Optional[str] = None, openai_client=None):
        self.client, self.openai_client = client, openai_client
        self.model = model or os.environ.get("JAGIMAGE_AGENT_MODEL", "gemini-flash-latest")
        self.used = None  # "gemini" or "openai": which one answered last

    def _ask(self, message: str, system: Optional[str] = None, schema: dict = AGENT_SCHEMA, images=()) -> dict:
        """Plans with whichever AI has a key: Gemini first, OpenAI when there is no Gemini key or Gemini
        can't be used (e.g. out of credits)."""
        system = system or agent_system()
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

    def _ask_gemini(self, message: str, system: str, schema: dict = AGENT_SCHEMA, images=()) -> dict:
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

    def _ask_openai(self, message: str, system: str, schema: dict = AGENT_SCHEMA, images=()) -> dict:
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
                tops = [t for t in tops if t in studio.tops_for(category)] or ["HalfSleeveTee"]  # adult- or kids-only styles
                fabric = b.get("fabric") if b.get("fabric") in L.FABRICS else "PureCotton"
                made = L.colours_for(fabric, category)  # the colours this category is made in
                colours = [c for c in listed("top_colours") if c in made] or list(made)
                auto_bottom = b.get("bottom") not in studio.BOTTOMS  # not asked for: each style its usual bottom
                bottom = studio.default_bottom(tops[0]) if auto_bottom else b["bottom"]
                poses = [p for p in listed("poses") if p in L.POSES] or ["front_main", "back"]
                if "front_main" not in poses:
                    poses = ["front_main"] + poses
                engine = dict(base)
                if b.get("quality") == "high":  # "high quality": OpenAI high, or Gemini Pro
                    engine.update(quality="high") if engine.get("provider") == "openai" else engine.update(gemini_model="pro")
                body = {**engine, "category": category, "top": tops[0], "tops": tops[1:], "bottom": bottom, "fabric": fabric,
                        "top_colours": colours, "pairing": b.get("pairing") if b.get("pairing") in ("match", "each") else "match",
                        "shot": b.get("shot") if b.get("shot") in studio.SHOTS else studio.default_shot(tops[0]),
                        "models": max(1, min(5000, int(b.get("models") or 1))), "poses": poses, "auto_bottom": auto_bottom}
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
        from .dropbox import Dropbox

        self.dropbox = Dropbox(self.data, os.environ.get("DROPBOX_APP_KEY", ""), os.environ.get("DROPBOX_APP_SECRET", ""))
        self.printruns = {}  # id -> PrintRun, this session
        self.white_jobs = {}  # id -> WhiteJob (white-background copies of the plain photos), this session
        from .whitejob import WhiteQueue

        self.white_queue = WhiteQueue(self.drive, self.data, self.white_jobs)
        if self.white_queue.paths():  # folders still waiting from before a restart: carry on in a minute

            def later():
                time.sleep(60)
                try:
                    if self.drive.can_read:
                        self.white_queue.resume()
                except Exception:
                    pass
            threading.Thread(target=later, daemon=True, name="white-resume").start()
        self.listing_jobs = {}  # id -> ListingJob, this session
        self.inventory = {}  # what is in the photos folder on Drive (admin check), filled by POST /api/drive/inventory
        self.oauth_states = {}
        self.plans = {}  # plan id -> agent plan, until Start

    def restore_base(self, path: Path) -> bytes:
        """A base photo the Drive sync moved to <photos folder>/_bases/<batch>/<model>/<shoot>/: its bytes."""
        path = Path(path)
        parts = ["_bases", path.parent.parent.parent.parent.name, path.parent.parent.parent.name, path.parent.name]
        return self.drive.fetch(parts, path.name)

    # ---- SKU pattern (Print placement) and the shop's listing facts (Listing tab), saved for everyone
    def shop(self) -> dict:
        from .printrun import DEFAULT_PATTERN

        f = self.data / "shop.json"
        saved = json.loads(f.read_text()) if f.exists() else {}
        return {"brand": saved.get("brand", "SPB"), "sku_pattern": saved.get("sku_pattern", DEFAULT_PATTERN),
                "facts": saved.get("facts", {})}

    def set_shop(self, body: dict) -> dict:
        from .printrun import make_sku, sku_fields

        cur = self.shop()
        if "brand" in body:
            cur["brand"] = " ".join(str(body["brand"] or "").split())[:40] or "SPB"
        if "sku_pattern" in body:
            pat = " ".join(str(body["sku_pattern"] or "").split())[:120]
            make_sku(pat, sku_fields(cur["brand"], {"category": "Men", "style": "HalfSleeveTee", "colour": "Red"}, "Test", 1))
            cur["sku_pattern"] = pat
        if isinstance(body.get("facts"), dict):
            cur["facts"] = {str(k)[:40]: " ".join(str(v).split())[:300] for k, v in body["facts"].items()}
        (self.data / "shop.json").write_text(json.dumps(cur, indent=1))
        return cur

    def ai(self, message, system, schema, images=()):
        return self.agent._ask(message, system, schema, list(images))

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
            # the link shown is rebuilt from the folder id: nothing else of what was pasted (a javascript: link) is kept
            clean.append({"category": cat, "top": top, "colour": colour[:40], "link": f"https://drive.google.com/drive/folders/{folder}", "folder": folder})
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
