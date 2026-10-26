import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace

import pytest

from jagimage import studio
from jagimage.cloud import Agent, Cloud, Drive, DriveSync, Login, Users
from jagimage.server import serve
from test_studio import fake_generate

PASSWORD = "tirupur-2026"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def call(url, body=None, cookie=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", **({"Cookie": cookie} if cookie else {})})
    try:
        r = urllib.request.build_opener(NoRedirect).open(req, timeout=10)
        return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


# ---------------------------------------------------------------- fake Google
class FakeGoogle:
    """Just enough of oauth2.googleapis.com and the Drive v3 API for the uploader."""

    def __init__(self, fail_uploads=0):
        self.folders, self.files, self.fail_uploads, self.tokens = {}, [], fail_uploads, 0

    def __call__(self, req, timeout=None):
        url, data = req.full_url, req.data or b""
        if url.startswith("https://oauth2.googleapis.com/token"):
            form = urllib.parse.parse_qs(data.decode())
            self.tokens += 1
            out = {"access_token": f"tok{self.tokens}", "expires_in": 3600}
            if form["grant_type"] == ["authorization_code"]:
                out["refresh_token"] = "refresh-1"
            return self._resp(out)
        assert req.headers["Authorization"].startswith("Bearer tok")
        if url.startswith("https://www.googleapis.com/drive/v3/files?q="):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["q"][0]
            name, parent = re.search(r"name = '(.*?)'", q).group(1), re.search(r"'([^']+)' in parents", q).group(1)
            hit = [i for i, (n, p) in self.folders.items() if n == name and p == parent]
            return self._resp({"files": [{"id": h} for h in hit]})
        if url.startswith("https://www.googleapis.com/drive/v3/files?fields=id"):
            meta = json.loads(data)
            fid = f"folder{len(self.folders) + 1}"
            self.folders[fid] = (meta["name"], meta["parents"][0])
            return self._resp({"id": fid})
        if url.startswith("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"):
            if self.fail_uploads:
                self.fail_uploads -= 1
                raise urllib.error.URLError("network down")
            meta = json.loads(re.search(rb"\r\n\r\n(\{.*?\})\r\n--", data, re.S).group(1))
            self.files.append((meta["name"], meta["parents"][0], len(data)))
            return self._resp({"id": f"file{len(self.files)}"})
        raise AssertionError(url)

    @staticmethod
    def _resp(obj):
        return SimpleNamespace(read=lambda: json.dumps(obj).encode(), __enter__=lambda s: s, __exit__=lambda s, *a: None)


class Resp:
    def __init__(self, obj):
        self.obj = obj

    def read(self):
        return json.dumps(self.obj).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


FakeGoogle._resp = staticmethod(lambda obj: Resp(obj))


def path_of(g, fid):
    names = []
    while fid in g.folders:
        name, fid = g.folders[fid]
        names.append(name)
    return "/".join(reversed(names))


# ---------------------------------------------------------------- tests
def test_login_needs_a_real_password_and_slows_down_guessing(tmp_path):
    with pytest.raises(ValueError):
        Users(tmp_path / "a", "short")
    users = Users(tmp_path / "b", PASSWORD)
    lg = Login(users, b"k" * 32)
    assert not lg.valid("") and not lg.valid("jag_session=123.admin.1.abc")
    cookie = lg.cookie("admin").split(";")[0]
    assert lg.user(cookie) == "admin" and not Login(users, b"x" * 32).valid(cookie)  # signed with the server's secret
    for _ in range(5):
        assert not lg.try_login("1.2.3.4", "admin", "guess")
    with pytest.raises(PermissionError):
        lg.try_login("1.2.3.4", "admin", PASSWORD)
    assert lg.try_login("5.6.7.8", "admin", PASSWORD)  # other addresses aren't blocked
    users.set_password("admin", "new-password-1")
    assert not lg.valid(cookie)  # old sessions end when the password changes
    assert Users(tmp_path / "b", "ignored-now").check("admin", "new-password-1")  # the env password only starts things


def signed(base, name="admin", password=PASSWORD):
    code, headers, _ = call(base + "/login", {"username": name, "password": password})
    assert code == 200, (name, code)
    return headers["Set-Cookie"].split(";")[0]


def test_admin_adds_users_who_cannot_change_settings(tmp_path):
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD, drive=Drive("", "", tmp_path / "data"))
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        admin = signed(base)
        assert json.loads(call(base + "/api/me", cookie=admin)[2]) == {"user": "admin", "role": "admin"}
        assert call(base + "/api/users", {"username": "Ravi", "password": "short"}, cookie=admin)[0] == 400
        code, _, body = call(base + "/api/users", {"username": "Ravi", "password": "ravi-pass-1"}, cookie=admin)
        assert code == 200 and [u["name"] for u in json.loads(body)["users"]] == ["admin", "ravi"]
        assert call(base + "/api/users", {"username": "ravi", "password": "ravi-pass-1"}, cookie=admin)[0] == 400  # taken
        ravi = signed(base, "ravi", "ravi-pass-1")
        assert json.loads(call(base + "/api/me", cookie=ravi)[2])["role"] == "user"
        # a user makes photos, but keys, users, settings and Drive are for admins
        assert call(base + "/api/options", cookie=ravi)[0] == 200
        for path, body in [("/api/keys", {"gemini": "k"}), ("/api/users", {"username": "x2", "password": "12345678"}),
                           ("/api/settings", {"out": "/tmp"}), ("/api/drive/client", {}), ("/api/users/admin/delete", {})]:
            assert call(base + path, body, cookie=ravi)[0] == 403, path
        assert call(base + "/api/users", cookie=ravi)[0] == 403
        # own password: needs the current one; afterwards the old one fails
        assert call(base + "/api/account/password", {"current": "nope", "new": "ravi-pass-2"}, cookie=ravi)[0] == 400
        code, headers, _ = call(base + "/api/account/password", {"current": "ravi-pass-1", "new": "ravi-pass-2"}, cookie=ravi)
        assert code == 200 and call(base + "/api/me", cookie=headers["Set-Cookie"].split(";")[0])[0] == 200  # stays signed in
        assert call(base + "/api/me", cookie=ravi)[0] == 401  # the old session elsewhere is signed out
        assert call(base + "/login", {"username": "ravi", "password": "ravi-pass-1"})[0] == 403
        ravi = signed(base, "ravi", "ravi-pass-2")
        # admin resets a password, deletes the user; the last admin stays
        assert call(base + "/api/users/ravi/password", {"password": "reset-pass-3"}, cookie=admin)[0] == 200
        assert call(base + "/api/me", cookie=ravi)[0] == 401
        assert call(base + "/api/users/admin/delete", {}, cookie=admin)[0] == 400
        assert call(base + "/api/users/ravi/delete", {}, cookie=admin)[0] == 200
        assert call(base + "/login", {"username": "ravi", "password": "reset-pass-3"})[0] == 403
        # admin keys: save, then remove
        assert json.loads(call(base + "/api/keys", {"openai": "sk-test", "remember": True}, cookie=admin)[2])["openai"] is True
        assert json.loads(call(base + "/api/keys", {"clear": ["openai"]}, cookie=admin)[2])["openai"] is False
        # Drive: the admin pastes the OAuth client in the website, then Connect goes to Google
        assert call(base + "/api/drive/client", {"client_id": "x", "client_secret": "y"}, cookie=admin)[0] == 400
        code, _, body = call(base + "/api/drive/client", {"client_id": "123-abc.apps.googleusercontent.com", "client_secret": "GOCSPX-secret-1"}, cookie=admin)
        assert code == 200 and json.loads(body)["configured"] is True
        code, headers, _ = call(base + "/drive/connect", cookie=admin)
        assert code == 302 and headers["Location"].startswith("https://accounts.google.com/") and "123-abc.apps" in headers["Location"]
        assert Drive("", "", tmp_path / "data").configured  # kept for the next start
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        httpd.shutdown()


def test_server_version_asks_for_the_password(tmp_path):
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD, drive=Drive("", "", tmp_path / "data"))
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        code, headers, _ = call(base + "/")
        assert code == 302 and headers["Location"] == "/login"
        assert call(base + "/api/options")[0] == 401 and call(base + "/api/jobs", {})[0] == 401
        assert call(base + "/files/x/y.jpg")[0] == 302
        assert call(base + "/login", {"password": "nope"})[0] == 403
        code, headers, _ = call(base + "/login", {"password": PASSWORD})  # user name left out: admin
        cookie = headers["Set-Cookie"].split(";")[0]
        assert code == 200 and "HttpOnly" in headers["Set-Cookie"]
        code, _, body = call(base + "/api/options", cookie=cookie)
        assert code == 200 and json.loads(body)["cloud"] is True
        code, _, page = call(base + "/recolor/", cookie=cookie)
        assert code == 200 and b"Jag Recolor" in page
        assert call(base + "/recolor/samples/men-half-sleeve.jpg", cookie=cookie)[0] == 200
        assert call(base + "/recolor/../../jagimage/cloud.py", cookie=cookie)[0] in (403, 404)
        drive = json.loads(call(base + "/api/drive", cookie=cookie)[2])
        assert drive["configured"] is False and drive["connected"] is False
    finally:
        httpd.shutdown()


def test_drive_connect_then_photos_upload_once_into_the_right_folders(tmp_path):
    g = FakeGoogle(fail_uploads=1)
    drive = Drive("cid", "secret", tmp_path / "data", "kinsey models 2026", opener=g)
    url = drive.auth_url("https://jag.example/drive/callback", "st8")
    assert "drive.file" in urllib.parse.unquote(url) and "access_type=offline" in url
    drive.finish("code-1", "https://jag.example/drive/callback")
    assert drive.connected and json.loads((tmp_path / "data" / "drive.json").read_text())["refresh_token"] == "refresh-1"
    # a finished batch on disk
    spec = studio.Spec(top_colours=["Red", "Navy"], models=1, poses=["front_main", "back"], keep_png=False)
    studio.Batch(spec, out=tmp_path / "out", generate=fake_generate, workers=1).run()
    sync = DriveSync(drive, lambda: str(tmp_path / "out"))
    assert len(sync.pending()) == 4  # 2 colours x front and back
    assert sync.run_once() == 0 and "network down" in sync.status["last_error"]  # retried next round
    assert sync.run_once() == 4 and sync.status["last_error"] is None
    assert sync.run_once() == 0  # nothing twice, also after a restart:
    assert DriveSync(Drive("cid", "secret", tmp_path / "data", "kinsey models 2026", opener=g), lambda: str(tmp_path / "out")).run_once() == 0
    names = sorted((path_of(g, parent), name) for name, parent, _ in g.files)
    assert names[0] == (f"kinsey models 2026/{spec.name}/Navy__Navy", "m0001_back.jpg")
    assert len({p for p, _ in names}) == 2 and len([f for f in g.folders.values() if f[0] == "kinsey models 2026"]) == 1


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.models = answer, self
        self.asked = None

    def generate_content(self, model, contents, config):
        self.asked = (model, contents)
        return SimpleNamespace(text=json.dumps(self.answer))


def test_agent_plans_batches_with_cost_and_runs_them_after_start(tmp_path):
    llm = FakeLLM({"reply": "Two styles for 3 Men models.", "batches": [
        {"category": "Men", "tops": ["HalfSleeveTee", "Hoodie", "Tuxedo"], "models": 3, "top_colours": ["Red", "Navy", "Gold"]},
        {"category": "Aliens", "tops": ["HalfSleeveTee"], "models": 2}]})
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD,
                  drive=Drive("", "", tmp_path / "data"), agent=Agent(client=llm, model="gemini-test"))
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cookie = call(base + "/login", {"password": PASSWORD})[1]["Set-Cookie"].split(";")[0]
    try:
        code, _, body = call(base + "/api/agent", {"message": "3 men in half sleeve and hoodie, red and navy",
                                                   "provider": "gemini", "one_base": True}, cookie=cookie)
        plan = json.loads(body)
        assert code == 200 and llm.asked[0] == "gemini-test"
        assert [b["styles"] for b in plan["batches"]] == [["HalfSleeveTee", "Hoodie"]]  # "Tuxedo" isn't a style
        b = plan["batches"][0]
        assert b["colours"] == 2 and b["ai_images"] == 3 * 2 * 2 and b["photos"] == 3 * 2 * 2 * 2 and b["cost_inr"] > 0
        assert plan["problems"] and "Aliens" in plan["problems"][0]  # unknown category: reported, not made
        assert "bodies" not in plan  # the plan stays on the server until Start
        code, _, body = call(base + "/api/agent/start", {"plan_id": plan["plan_id"]}, cookie=cookie)
        assert code == 200
        for _ in range(900):
            runs = json.loads(call(base + "/api/agent/runs", cookie=cookie)[2])["runs"]
            if not runs[0]["running"]:
                break
            time.sleep(0.1)
        assert runs[0]["done"] == 1 and runs[0]["error"] is None
        jobs = json.loads(call(base + "/api/jobs", cookie=cookie)[2])["jobs"]
        assert jobs[0]["counts"]["recoloured"] == 24 and jobs[0]["counts"]["errors"] == 0
        assert call(base + "/api/agent/start", {"plan_id": plan["plan_id"]}, cookie=cookie)[0] == 400  # used once
    finally:
        httpd.shutdown()
