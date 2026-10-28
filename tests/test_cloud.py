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
        self.folders, self.files, self.fail_uploads, self.tokens, self.replaced = {}, [], fail_uploads, 0, []

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
            if "mimeType != " in q:  # a photo of this name already in the folder?
                return self._resp({"files": [{"id": f"file{i + 1}"} for i, f in enumerate(self.files) if f[0] == name and f[1] == parent]})
            hit = [i for i, (n, p) in self.folders.items() if n == name and p == parent]
            return self._resp({"files": [{"id": h} for h in hit]})
        if url.startswith("https://www.googleapis.com/drive/v3/files?fields=id"):
            meta = json.loads(data)
            fid = f"folder{len(self.folders) + 1}"
            self.folders[fid] = (meta["name"], meta["parents"][0])
            return self._resp({"id": fid})
        if url.startswith("https://www.googleapis.com/upload/drive/v3/files/") and req.get_method() == "PATCH":
            self.replaced.append(url.split("/files/")[1].split("?")[0])
            return self._resp({"id": self.replaced[-1]})
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
        assert call(base + "/api/users", {"username": "ravi.k", "password": "ravi-pass-9"}, cookie=admin)[0] == 200
        dotted = signed(base, "Ravi.K", "ravi-pass-9")  # a dot in the name, typed in capitals
        assert json.loads(call(base + "/api/me", cookie=dotted)[2])["user"] == "ravi.k"
        assert call(base + "/api/users/ravi.k/delete", {}, cookie=admin)[0] == 200
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
    # recolouring again remakes the photos; Drive gets them as replacements, not duplicates
    redo = studio.Batch(spec, out=tmp_path / "out", generate=fake_generate, workers=1, redo=True)
    redo.run()
    assert redo.counts["recoloured"] == 4 and redo.counts["recolour_skipped"] == 0
    assert len(sync.pending()) == 4 and sync.run_once() == 4
    assert len(g.files) == 4 and sorted(g.replaced) == ["file1", "file2", "file3", "file4"]


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.models = answer, self
        self.asked = None

    def generate_content(self, model, contents, config):
        self.asked = (model, contents)
        return SimpleNamespace(text=json.dumps(self.answer))


def test_agent_plans_batches_with_cost_and_runs_them_after_start(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
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
        assert jobs[0]["name"].endswith("_top_only")  # Half Sleeve starts as Top only unless asked for full body
        assert call(base + "/api/agent/start", {"plan_id": plan["plan_id"]}, cookie=cookie)[0] == 400  # used once
    finally:
        httpd.shutdown()


class FakeOpenAIChat:
    """client.chat.completions.create, answering like gpt-*-mini; the first model "doesn't exist"."""

    def __init__(self, answer):
        self.answer, self.models = answer, []
        self.chat = SimpleNamespace(completions=self)

    def create(self, model, messages, response_format):
        self.models.append(model)
        if len(self.models) == 1:
            raise RuntimeError(f"The model `{model}` does not exist or you do not have access to it.")
        assert response_format == {"type": "json_object"} and "JSON schema" in messages[0]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.answer)))])


def test_agent_uses_the_openai_key_when_there_is_no_gemini_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chat = FakeOpenAIChat({"reply": "10 Men models in every top style.", "batches": [
        {"category": "Men", "tops": ["HalfSleeveTee", "FullSleeveTee", "SleevelessTee", "Hoodie", "Sweatshirt"],
         "models": 10, "top_colours": ["Red"], "poses": ["front_main"], "quality": "high"}]})
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD,
                  drive=Drive("", "", tmp_path / "data"), agent=Agent(openai_client=chat))
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cookie = call(base + "/login", {"password": PASSWORD})[1]["Set-Cookie"].split(";")[0]
    try:
        # the Studio still has Gemini selected, but only an OpenAI key is saved
        code, _, body = call(base + "/api/agent", {"message": "10 mens half sleeve, same model in all tops, high quality",
                                                   "provider": "gemini", "gemini_model": "flash", "batch": True}, cookie=cookie)
        plan = json.loads(body)
        assert code == 200, plan
        assert len(chat.models) == 2 and cloud.agent.used == "openai"
        assert plan["batches"][0]["styles"] == ["HalfSleeveTee", "FullSleeveTee", "SleevelessTee", "Hoodie", "Sweatshirt"]
        assert "OpenAI" in plan["reply"]
        bodies = cloud.plans[plan["plan_id"]]["bodies"]
        assert bodies[0]["provider"] == "openai" and bodies[0]["quality"] == "high" and bodies[0]["batch"] is True  # OpenAI has batch mode too
        monkeypatch.delenv("OPENAI_API_KEY")
        code, _, body = call(base + "/api/agent", {"message": "10 men", "provider": "gemini"}, cookie=cookie)
        assert code == 400 and "API key" in json.loads(body)["error"]
    finally:
        httpd.shutdown()


MYNTRA = {"marketplace": "Myntra", "width": 1080, "height": 1440, "background": "light grey (#F2F2F2)", "framing": "full",
          "poses": [{"name": "front", "direction": "Facing the camera, arms relaxed, neutral smile"},
                    {"name": "back_view", "direction": "Back to the camera, head straight"},
                    {"name": "walking", "direction": "Mid-stride walking towards the camera"},
                    {"name": "front_close", "direction": "a second front shot"}],
          "model": "Natural makeup, hair tied back, no jewellery", "avoid": "props, sunglasses",
          "not_applied": "Name files by style code", "summary": "Full-length photos at 1080 x 1440 on light grey."}


def test_guide_files_are_read_into_text_and_pictures(tmp_path):
    import io
    import zipfile

    from PIL import Image

    from jagimage.cloud import guide_content
    text, pics = guide_content("myntra.txt", b"Image size 1080 x 1440. Light grey background.")
    assert "1080" in text and pics == []
    buf = io.BytesIO()
    Image.new("RGB", (600, 800), "#cccccc").save(buf, "PNG")
    doc = io.BytesIO()
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr("word/document.xml", "<w:document><w:p><w:t>Front pose</w:t></w:p><w:p><w:t>Back pose</w:t></w:p></w:document>")
        z.writestr("word/media/image1.png", buf.getvalue())
    text, pics = guide_content("guide.docx", doc.getvalue())
    assert "Front pose\nBack pose" in text and len(pics) == 1 and pics[0][1] == "image/jpeg"
    text, pics = guide_content("page.png", buf.getvalue())
    assert text == "" and len(pics) == 1
    pdf = io.BytesIO()
    Image.new("RGB", (600, 800), "#cccccc").save(pdf, "PDF")
    text, pics = guide_content("scan.pdf", pdf.getvalue())  # a scanned page: its picture goes to the AI
    assert len(pics) == 1
    with pytest.raises(ValueError, match="Upload the guide as"):
        guide_content("guide.exe", b"MZ")


def test_guide_answer_becomes_checked_studio_settings():
    from jagimage import library as L
    from jagimage.cloud import clean_guide
    g = clean_guide(MYNTRA, "myntra.pdf")
    assert g["canvas"] == [1080, 1440] and g["shot"] == "outfit" and g["background"] == "light grey (#F2F2F2)"
    assert g["poses"] == ["front_main", "back", "walking"]  # the second front shot is the same pose
    assert g["custom_poses"]["back"].startswith(L.POSES["back"]) and "head straight" in g["custom_poses"]["back"]
    assert g["custom_poses"]["walking"] == "Mid-stride walking towards the camera"
    assert g["extra"] == "Natural makeup, hair tied back, no jewellery. Avoid: props, sunglasses"
    bad = clean_guide({"marketplace": "", "width": 99999, "height": 5, "framing": "sideways", "poses": "x", "summary": ""}, "x.txt")
    assert bad["canvas"] is None and bad["shot"] == "" and bad["poses"] == ["front_main"] and bad["marketplace"] == "x"
    spec = studio.Spec(category="Women", top="HalfSleeveTee", bottom="Trackpants", top_colours=["Red"], poses=g["poses"],
                       custom_poses=g["custom_poses"], extra=g["extra"], canvas=g["canvas"], background=g["background"],
                       guide=g["marketplace"], shot="outfit")
    text = studio.prompt(spec, 1, spec.bases()[0], "walking", "front")
    assert "Plain seamless light grey (#F2F2F2) studio background" in text and "#EDEDED" not in text and "Mid-stride" in text
    assert spec.name.endswith("_Myntra") and spec.canvas == (1080, 1440)


def test_guide_upload_is_read_saved_and_listed(tmp_path, monkeypatch):
    import base64
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    llm = FakeLLM(MYNTRA)
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD,
                  drive=Drive("", "", tmp_path / "data"), agent=Agent(client=llm, model="gemini-test"))
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cookie = call(base + "/login", {"password": PASSWORD})[1]["Set-Cookie"].split(";")[0]
    try:
        data = base64.b64encode(b"Myntra model guidelines: 1080 x 1440, light grey, front, back and walking.").decode()
        code, _, body = call(base + "/api/guide", {"filename": "myntra.txt", "data": data}, cookie=cookie)
        g = json.loads(body)["guide"]
        assert code == 200 and g["id"] == "myntra" and g["canvas"] == [1080, 1440]
        assert "1080 x 1440" in str(llm.asked[1])  # the AI got the guide's text
        assert [x["id"] for x in json.loads(call(base + "/api/guides", cookie=cookie)[2])["guides"]] == ["myntra"]
        assert call(base + "/api/guide", {"filename": "x.exe", "data": data}, cookie=cookie)[0] == 400
        assert call(base + "/api/guide", {"filename": "x.txt", "data": "not base64!"}, cookie=cookie)[0] == 400
        code, _, body = call(base + "/api/guides/myntra/delete", {}, cookie=cookie)
        assert code == 200 and json.loads(body)["guides"] == []
        # a guide batch: its own folder, photo size and backdrop, and its photos still go to the dashboard
        spec = {"category": "Men", "top": "FullSleeveTee", "bottom": "Trackpants", "top_colours": ["Navy"], "models": 1,
                "poses": ["front_main"], "canvas": [1080, 1440], "background": "light grey (#F2F2F2)", "guide": "Myntra"}
        code, _, body = call(base + "/api/jobs", spec, cookie=cookie)
        name = json.loads(body)["name"]
        assert code == 200 and name.endswith("_Myntra")
        for _ in range(300):
            if not json.loads(call(base + "/api/jobs", cookie=cookie)[2])["jobs"][0]["running"]:
                break
            time.sleep(0.1)
        assert list((tmp_path / "out" / name).glob("m0001/*/front_main_1080x1440.jpg"))
        from jagimage.cloud import JPG_SIZE
        assert JPG_SIZE.search("front_main_1080x1440.jpg") and not JPG_SIZE.search("front_main_qc.jpg")
    finally:
        httpd.shutdown()


def test_guide_screenshot_goes_to_openai_as_a_picture(monkeypatch):
    import io

    from PIL import Image
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    chat = FakeOpenAIChat(MYNTRA)
    seen = []
    real = chat.create
    chat.create = lambda model, messages, response_format: (seen.append(messages), real(model, messages, response_format))[1]
    buf = io.BytesIO()
    Image.new("RGB", (600, 800), "#cccccc").save(buf, "PNG")
    g = Agent(openai_client=chat).read_guide("page1.png", buf.getvalue())
    parts = seen[-1][1]["content"]
    assert g["marketplace"] == "Myntra" and parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_print_folders_are_saved_and_their_prints_downloaded_once(tmp_path):
    from jagimage.cloud import drive_folder_id
    assert drive_folder_id("https://drive.google.com/drive/folders/1wfJz1wv8BYT5iN-1a7mzYWhlCrOkzwvr?usp=sharing") == "1wfJz1wv8BYT5iN-1a7mzYWhlCrOkzwvr"
    with pytest.raises(ValueError, match="Drive folder link"):
        drive_folder_id("https://example.com/x")
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD, drive=Drive("", "", tmp_path / "data"))
    rows = cloud.set_print_rows([
        {"category": "Men", "top": "HalfSleeveTee", "colour": "", "link": "https://drive.google.com/drive/folders/AAAAAAAAAAAA"},
        {"category": "Men", "top": "HalfSleeveTee", "colour": "Black", "link": "https://drive.google.com/drive/folders/BBBBBBBBBBBB"}])
    assert [r["folder"] for r in rows] == ["AAAAAAAAAAAA", "BBBBBBBBBBBB"]
    with pytest.raises(ValueError, match="category and top style"):
        cloud.set_print_rows([{"category": "Aliens", "top": "HalfSleeveTee", "link": "AAAAAAAAAAAA"}])
    downloads = []
    cloud.drive = SimpleNamespace(can_read=True, list_images=lambda folder: [{"id": folder + "-1", "name": "237.png", "modifiedTime": "t1"}],
                                  download=lambda fid: (downloads.append(fid), b"png-bytes")[1])
    black = cloud.print_files("Men", "HalfSleeveTee", "Black")
    navy = cloud.print_files("Men", "HalfSleeveTee", "Navy")  # no folder of its own: the all-colours one
    assert black[0][0] == "237" and black[0][1].read_bytes() == b"png-bytes" and "BBBB" in black[0][1].name
    assert "AAAA" in navy[0][1].name and cloud.print_files("Women", "Hoodie", "Red") == []
    cloud.print_files("Men", "HalfSleeveTee", "Black")
    assert downloads == ["BBBBBBBBBBBB-1", "AAAAAAAAAAAA-1"]  # kept: not downloaded again
    cloud.drive.can_read = False
    with pytest.raises(RuntimeError, match="Reconnect Google Drive"):
        cloud.print_files("Men", "HalfSleeveTee", "Black")


def test_print_agents_check_photos_and_turn_words_into_a_placement(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    llm = FakeLLM({"ok": False, "score": 4, "issues": ["too high", ""], "dx": 0, "dy": 0.9, "scale": 1.0})
    agent = Agent(client=llm, model="gemini-test")
    res = agent.check_print(Image.new("RGB", (800, 1200), "navy"), Image.new("RGBA", (300, 300), (255, 0, 0, 255)))
    assert res == {"ok": False, "score": 4, "issues": ["too high"], "dx": 0.0, "dy": 0.4, "scale": 1.0}  # move kept within limits
    assert len(llm.asked[1]) == 3  # photo, artwork, then the question
    llm.answer = {"place": "nowhere", "dx": 0, "dy": 0.05, "scale": 0.85, "reply": "Moved it a little lower and made it smaller."}
    plan = agent.plan_print("a bit lower and smaller", {"place": "chest_medium", "dx": 0, "dy": 0, "scale": 1})
    assert plan == {"place": "chest_medium", "dx": 0.0, "dy": 0.05, "scale": 0.85, "reply": "Moved it a little lower and made it smaller."}
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD, drive=Drive("", "", tmp_path / "data"), agent=agent)
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cookie = call(base + "/login", {"password": PASSWORD})[1]["Set-Cookie"].split(";")[0]
    try:
        code, _, body = call(base + "/api/prints/plan", {"message": "a bit lower and smaller", "current": {"place": "chest_large"}}, cookie=cookie)
        assert code == 200 and json.loads(body)["scale"] == 0.85
        assert call(base + "/api/prints/plan", {"message": " "}, cookie=cookie)[0] == 400
    finally:
        httpd.shutdown()


def test_print_preview_shows_one_photo_and_the_checkers_verdict(tmp_path, monkeypatch):
    import base64
    import io

    from PIL import Image
    from test_studio import make_print
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    llm = FakeLLM({"ok": False, "issues": ["too high"], "dx": 0, "dy": 0.05, "scale": 0.9})
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD,
                  drive=Drive("", "", tmp_path / "data"), agent=Agent(client=llm, model="gemini-test"))
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cookie = call(base + "/login", {"password": PASSWORD})[1]["Set-Cookie"].split(";")[0]
    try:
        s = studio.Spec(category="Men", top="HalfSleeveTee", bottom="Trackpants", top_colours=["Navy", "White"], poses=["front_main"])
        studio.Batch(s, out=tmp_path / "out", generate=fake_generate).run()
        art = base64.b64encode(make_print(tmp_path / "p.png").read_bytes()).decode()
        code, _, body = call(base + "/api/prints/preview", {"batch": s.name, "colour": "White", "art": art, "dy": 0.05}, cookie=cookie)
        r = json.loads(body)
        assert code == 200 and r["colours"] == ["Navy", "White"] and r["colour"] == "White" and r["check"] is None and r["dy"] == 0.05
        im = Image.open(io.BytesIO(base64.b64decode(r["image"].split(",", 1)[1])))
        assert im.format == "JPEG" and max(im.size) <= 1200
        assert not list((tmp_path / "out" / s.name).rglob("*print*"))  # a preview saves nothing
        code, _, body = call(base + "/api/prints/preview", {"batch": s.name, "art": art, "check": True}, cookie=cookie)
        check = json.loads(body)["check"]
        assert {k: check[k] for k in ("ok", "issues", "dx", "dy", "scale")} == {"ok": False, "issues": ["too high"], "dx": 0.0, "dy": 0.05, "scale": 0.9}
        assert "collar gap" in str(llm.asked[1][-1]) and check["measured"]["collar_gap"] > 0.07  # the AI is told the measurements
        # no file uploaded: the prints come from the print folders
        cloud.print_files = lambda cat, top, colour: [("Logo", make_print(tmp_path / "q.png")), ("237", make_print(tmp_path / "r.png"))]
        code, _, body = call(base + "/api/prints/preview", {"batch": s.name, "print": "237"}, cookie=cookie)
        assert json.loads(body)["prints"] == ["Logo", "237"] and json.loads(body)["print"] == "237"
        cloud.print_files = lambda cat, top, colour: []
        code, _, body = call(base + "/api/prints/preview", {"batch": s.name}, cookie=cookie)
        assert code == 400 and "upload a print file" in json.loads(body)["error"]
        assert call(base + "/api/prints/preview", {"batch": "../etc"}, cookie=cookie)[0] in (400, 404)
    finally:
        httpd.shutdown()
