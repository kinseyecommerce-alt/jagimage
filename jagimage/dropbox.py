"""Dropbox: the shop's own Dropbox account, connected once (OAuth, offline access) so the server can upload
finished photos into SKU folders and make links marketplaces can download images from. Plain HTTPS calls to
the Dropbox v2 API; no extra libraries."""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

AUTH = "https://www.dropbox.com/oauth2/authorize"
TOKEN = "https://api.dropboxapi.com/oauth2/token"
API = "https://api.dropboxapi.com/2"
CONTENT = "https://content.dropboxapi.com/2"
CHUNK = 8 * 1024 * 1024  # files above this go up in an upload session


def dropbox_path(*parts: str) -> str:
    """A Dropbox path from folder names: "/" + parts, each without slashes or characters Dropbox refuses."""
    bad = '\\/:?*"<>|'
    clean = ["".join("-" if ch in bad else ch for ch in str(p)).strip().strip(".") or "_" for p in parts if str(p).strip()]
    return "/" + "/".join(clean)


def _why(endpoint: str, exc) -> str:
    """Dropbox's own words for a refused call; a missing permission says which one to tick."""
    raw = exc.read()[:2000].decode("utf-8", "replace")
    try:
        msg = (json.loads(raw).get("user_message") or {}).get("text") or json.loads(raw).get("error_summary") or raw
    except ValueError:
        msg = raw
    m = re.search(r"required scope '([a-z._]+)'", msg)
    if m:
        return (f"Dropbox app is missing the permission {m.group(1)}: in the Dropbox App Console → Permissions tick it "
                "(and files.content.write, sharing.write), Submit, then Settings → Dropbox → Connect again")
    return f"Dropbox {endpoint}: {exc.code} {msg[:400]}"


class Dropbox:
    def __init__(self, data: Path, app_key: str = "", app_secret: str = "", root: str = "Jag Studio",
                 opener: Callable = urllib.request.urlopen):
        self.data = Path(data)
        self.client_file = self.data / "dropbox_client.json"
        self.file = self.data / "dropbox.json"
        self.app_key, self.app_secret = app_key, app_secret
        if not (self.app_key and self.app_secret) and self.client_file.exists():
            saved = json.loads(self.client_file.read_text())
            self.app_key, self.app_secret = saved.get("app_key", ""), saved.get("app_secret", "")
        self.state = json.loads(self.file.read_text()) if self.file.exists() else {}
        self.root = self.state.get("root") or root
        self.opener = opener
        self.lock = threading.Lock()
        self._access = (None, 0.0)

    @property
    def configured(self) -> bool:
        return bool(self.app_key and self.app_secret)

    @property
    def connected(self) -> bool:
        return bool(self.state.get("refresh_token"))

    def status(self) -> dict:
        return {"configured": self.configured, "connected": self.connected, "root": self.root,
                "account": self.state.get("account", "")}

    def _write(self, path: Path, obj: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj))
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    def set_client(self, app_key: str, app_secret: str, root: str = ""):
        app_key, app_secret = str(app_key or "").strip(), str(app_secret or "").strip()
        if len(app_key) < 8 or len(app_secret) < 8 or " " in app_key + app_secret:
            raise ValueError("Paste the App key and App secret from your Dropbox app's Settings page")
        with self.lock:
            self._write(self.client_file, {"app_key": app_key, "app_secret": app_secret})
            if app_key != self.app_key:
                self.state.pop("refresh_token", None)
                self._access = (None, 0.0)
            self.app_key, self.app_secret = app_key, app_secret
            if str(root or "").strip():
                self.set_root(root)
            self._write(self.file, self.state)

    def set_root(self, root: str):
        root = " ".join(str(root or "").replace("/", " ").split())[:80]
        if not root:
            raise ValueError("Name the Dropbox folder the photos go into")
        self.root = self.state["root"] = root
        self._write(self.file, self.state)

    def auth_url(self, redirect_uri: str, state: str) -> str:
        return AUTH + "?" + urllib.parse.urlencode({"client_id": self.app_key, "redirect_uri": redirect_uri,
                                                     "response_type": "code", "token_access_type": "offline",
                                                     "state": state})

    def _form(self, url: str, fields: dict) -> dict:
        req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.opener(req, timeout=30) as r:
            return json.loads(r.read())

    def finish(self, code: str, redirect_uri: str):
        tok = self._form(TOKEN, {"code": code, "grant_type": "authorization_code", "client_id": self.app_key,
                                 "client_secret": self.app_secret, "redirect_uri": redirect_uri})
        if not tok.get("refresh_token"):
            raise RuntimeError("Dropbox gave no refresh token: connect again")
        with self.lock:
            self.state["refresh_token"] = tok["refresh_token"]
            self._access = (tok.get("access_token"), time.time() + int(tok.get("expires_in", 14400)) - 60)
            try:
                self.state["account"] = self._rpc("users/get_current_account", None).get("email", "")
            except Exception:
                pass
            self._write(self.file, self.state)

    def disconnect(self):
        with self.lock:
            self.state.pop("refresh_token", None)
            self.state.pop("account", None)
            self._access = (None, 0.0)
            self._write(self.file, self.state)

    def _token(self) -> str:
        token, expires = self._access
        if token and time.time() < expires:
            return token
        if not self.connected:
            raise RuntimeError("Dropbox isn't connected: Settings → Dropbox → Connect")
        tok = self._form(TOKEN, {"refresh_token": self.state["refresh_token"], "grant_type": "refresh_token",
                                 "client_id": self.app_key, "client_secret": self.app_secret})
        self._access = (tok["access_token"], time.time() + int(tok.get("expires_in", 14400)) - 60)
        return tok["access_token"]

    def _rpc(self, endpoint: str, args) -> dict:
        req = urllib.request.Request(f"{API}/{endpoint}", data=json.dumps(args).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"})
        try:
            with self.opener(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(_why(endpoint, exc)) from exc

    def _content(self, endpoint: str, args: dict, data: bytes) -> dict:
        req = urllib.request.Request(f"{CONTENT}/{endpoint}", data=data, method="POST", headers={
            "Authorization": f"Bearer {self._token()}", "Content-Type": "application/octet-stream",
            "Dropbox-API-Arg": json.dumps(args, ensure_ascii=True)})
        try:
            with self.opener(req, timeout=300) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(_why(endpoint, exc)) from exc

    def upload(self, data: bytes, *parts: str) -> str:
        """Puts the bytes at <root>/<parts...> (the last part is the file name), replacing a file already there.
        Returns the Dropbox path."""
        path = dropbox_path(self.root, *parts)
        if len(data) <= CHUNK:
            self._content("files/upload", {"path": path, "mode": "overwrite", "mute": True}, data)
            return path
        sess = self._content("files/upload_session/start", {"close": False}, data[:CHUNK])["session_id"]
        off = CHUNK
        while len(data) - off > CHUNK:
            self._content("files/upload_session/append_v2", {"cursor": {"session_id": sess, "offset": off}, "close": False},
                          data[off:off + CHUNK])
            off += CHUNK
        self._content("files/upload_session/finish", {"cursor": {"session_id": sess, "offset": off},
                                                      "commit": {"path": path, "mode": "overwrite", "mute": True}}, data[off:])
        return path

    def delete(self, *parts: str) -> str:
        """Removes a file or folder under the app's folder (<root>/<parts...>); nothing outside it."""
        if not parts:
            raise ValueError("name what to delete inside the Dropbox folder")
        path = dropbox_path(self.root, *parts)
        try:
            self._rpc("files/delete_v2", {"path": path})
        except RuntimeError as exc:
            if "not_found" not in str(exc):
                raise
        return path

    def link(self, path: str) -> str:
        """A link anyone can download the file from (marketplaces fetch listing images this way)."""
        try:
            url = self._rpc("sharing/create_shared_link_with_settings", {"path": path})["url"]
        except RuntimeError as exc:
            if "shared_link_already_exists" not in str(exc):
                raise
            links = self._rpc("sharing/list_shared_links", {"path": path, "direct_only": True}).get("links") or []
            if not links:
                raise
            url = links[0]["url"]
        # a direct file instead of Dropbox's preview page
        url = url.replace("?dl=0", "?raw=1").replace("&dl=0", "&raw=1")
        return url if "raw=1" in url else url + ("&" if "?" in url else "?") + "raw=1"
