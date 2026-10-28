"""Attacks found in a security review of the website and desktop versions; each must stay closed."""
import http.client
import json
import threading

from jagimage.cloud import Agent, Cloud, Drive, Users
from jagimage.server import serve
from test_cloud import PASSWORD, FakeLLM, call
from test_studio import fake_generate


def start(tmp_path, cloud=True, agent=None):
    out = tmp_path / "studio"
    c = Cloud(tmp_path / "jag-data", lambda: str(out), password=PASSWORD, drive=Drive("", "", tmp_path / "jag-data"),
              agent=agent) if cloud else None
    app, httpd = serve(port=0, out=str(out), engine=lambda spec: fake_generate, cloud=c)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return c, httpd, httpd.server_address[1]


def raw(port, method, path, body=None, headers=None):
    """A request exactly as written (no URL clean-up), like an attacker's."""
    con = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    con.request(method, path, body=body, headers=headers or {})
    r = con.getresponse()
    return r.status, r.read()


def cookie_for(port, name="admin", password=PASSWORD):
    con = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    con.request("POST", "/login", json.dumps({"username": name, "password": password}), {"Content-Type": "application/json"})
    r = con.getresponse()
    r.read()
    return r.getheader("Set-Cookie").split(";")[0]


def test_files_route_never_leaves_a_batch_folder(tmp_path):
    c, httpd, port = start(tmp_path)
    try:
        ck = {"Cookie": cookie_for(port)}
        for path in ["/files/../jag-data/secret.key", "/files/%2e%2e/jag-data/users.json", "/files/%2Fetc/passwd",
                     "/thumbs/../jag-data/secret.key", "/files/x/..%2F..%2Fjag-data%2Fkeys.json", "/files/./jag-data/secret.key"]:
            status, body = raw(port, "GET", path, headers=ck)
            assert status == 404 and b"root:" not in body and len(body) < 200, (path, status)
    finally:
        httpd.shutdown()


def test_batches_cannot_write_outside_the_photos_folder(tmp_path):
    c, httpd, port = start(tmp_path)
    base = f"http://127.0.0.1:{port}"
    try:
        admin = cookie_for(port)
        assert call(base + "/api/users", {"username": "bob", "password": "bob-pass-1"}, cookie=admin)[0] == 200
        bob = cookie_for(port, "bob", "bob-pass-1")
        job = {"top_colours": ["Red"], "models": 1, "poses": ["front_main"]}
        for name in ["../jag-data/evil", "/tmp/evil", "a/b", ".."]:
            code, _, body = call(base + "/api/jobs", {**job, "name": name}, cookie=bob)
            assert code == 400 and "batch name" in json.loads(body)["error"], name
        assert call(base + "/api/jobs", {**job, "export": str(tmp_path / "anywhere")}, cookie=bob)[0] == 403
        assert call(base + "/api/jobs", {**job, "canvas": [100000, 100000]}, cookie=bob)[0] == 400
        assert not (tmp_path / "jag-data" / "evil").exists() and not (tmp_path / "anywhere").exists()
    finally:
        httpd.shutdown()


def test_old_sign_in_is_dead_after_the_account_is_deleted_and_made_again(tmp_path):
    c, httpd, port = start(tmp_path)
    base = f"http://127.0.0.1:{port}"
    try:
        admin = cookie_for(port)
        call(base + "/api/users", {"username": "carol", "password": "carol-pass-1"}, cookie=admin)
        old = cookie_for(port, "carol", "carol-pass-1")
        call(base + "/api/users/carol/delete", {}, cookie=admin)
        call(base + "/api/users", {"username": "carol", "password": "carol-pass-2", "role": "admin"}, cookie=admin)
        assert call(base + "/api/me", cookie=old)[0] == 401
    finally:
        httpd.shutdown()


def test_faked_forwarded_for_does_not_dodge_the_password_limit(tmp_path):
    c, httpd, port = start(tmp_path)
    try:
        hdr = lambda xff: {"Content-Type": "application/json", "X-Forwarded-For": xff}
        for i in range(6):  # wrong passwords, each claiming a different address before the proxy's real one
            raw(port, "POST", "/login", json.dumps({"password": "wrong"}), hdr(f"9.9.9.{i}, 1.1.1.1"))
        status, _ = raw(port, "POST", "/login", json.dumps({"password": PASSWORD}), hdr("8.8.8.8, 1.1.1.1"))
        assert status == 429  # the real client (last entry, added by the proxy) is still slowed down
        status, _ = raw(port, "POST", "/login", json.dumps({"password": PASSWORD}), hdr("1.1.1.1, 2.2.2.2"))
        assert status == 200  # and someone else isn't locked out by a faked first entry
    finally:
        httpd.shutdown()


def test_requests_from_other_websites_are_refused(tmp_path):
    _, httpd, port = start(tmp_path, cloud=False)  # the desktop app: no password, so this is its only guard
    try:
        status, _ = raw(port, "POST", "/api/keys", "{\"runpod_url\": \"https://attacker.example\"}",
                        {"Content-Type": "text/plain", "Origin": "https://evil.example"})
        assert status == 400  # a cross-site form or text POST can't be sent as JSON without the browser asking
        status, _ = raw(port, "GET", "/api/options", headers={"Host": "rebind.evil.example:8765"})
        assert status == 403  # a website that points its own name at 127.0.0.1
        assert raw(port, "GET", "/api/options", headers={"Host": f"127.0.0.1:{port}"})[0] == 200
    finally:
        httpd.shutdown()
    c, httpd, port = start(tmp_path / "web")
    try:  # signing someone in from another website (login CSRF) needs JSON too
        assert raw(port, "POST", "/login", json.dumps({"password": PASSWORD}), {"Content-Type": "text/plain"})[0] == 400
    finally:
        httpd.shutdown()


def test_odd_request_bodies_get_an_answer_not_a_dropped_connection(tmp_path):
    llm = FakeLLM({"reply": 5, "batches": [7, {"category": ["Men"], "tops": [1, None], "models": "x"}, None]})
    c, httpd, port = start(tmp_path, agent=Agent(client=llm))
    try:
        ck = cookie_for(port)
        for path in ["/login", "/api/users", "/api/agent", "/api/jobs"]:
            for body in ["[]", "null", "\"x\"", "1", "{bad json"]:
                status, _ = raw(port, "POST", path, body, {"Content-Type": "application/json", "Cookie": ck})
                assert status in (400, 401, 403), (path, body, status)
        import os
        os.environ["GEMINI_API_KEY"] = "g-test"
        try:  # the AI answers nonsense: the plan reports it instead of crashing
            status, body = raw(port, "POST", "/api/agent", json.dumps({"message": "x", "provider": "gemini"}),
                               {"Content-Type": "application/json", "Cookie": ck})
        finally:
            os.environ.pop("GEMINI_API_KEY")
        plan = json.loads(body)
        assert status == 200 and plan["batches"] == [] and plan["problems"], plan
    finally:
        httpd.shutdown()


def test_user_names_with_dots_stay_signed_in(tmp_path):
    users = Users(tmp_path, PASSWORD)
    users.create("j.doe", "jdoe-pass-1")
    from jagimage.cloud import Login
    lg = Login(users, b"k" * 32)
    assert lg.user(lg.cookie("j.doe").split(";")[0]) == "j.doe"


def test_huge_pictures_zip_bombs_and_bad_values_are_refused(tmp_path):
    import io
    import math
    import zipfile

    import pytest
    from PIL import Image

    from jagimage.cloud import Cloud, Drive, guide_content
    from jagimage.prints import PrintError, adjust, open_image
    from jagimage import studio
    # a tiny file that decodes to a huge picture is refused before it is decoded
    buf = io.BytesIO()
    Image.new("L", (13000, 13000)).save(buf, "PNG")
    assert len(buf.getvalue()) < 500_000
    with pytest.raises(PrintError, match="too large"):
        open_image(buf.getvalue())
    with pytest.raises(ValueError, match="too large"):
        guide_content("page.png", buf.getvalue())
    # a .docx whose text unpacks to a gigabyte is refused before unpacking
    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"<w:p>" + b"a" * 30_000_000)
    with pytest.raises(ValueError, match="too large"):
        guide_content("bomb.docx", bomb.getvalue())
    # damaged files are a plain "can't be read", not a server error
    for name, data in [("a.pdf", b"%PDF-1.4 garbage"), ("a.docx", b"PK\x03\x04garbage")]:
        with pytest.raises(ValueError, match="can't be read"):
            guide_content(name, data)
    # NaN can't slip past the placement limits; print_check must be a real true/false
    assert adjust(math.nan, float("inf"), math.nan) == {"dx": 0.0, "dy": 0.0, "scale": 1.0}  # not a number: the default
    assert adjust("x", None, "y") == {"dx": 0.0, "dy": 0.0, "scale": 1.0}
    with pytest.raises(ValueError, match="print_check"):
        studio.Spec(top_colours=["Navy"], print_check="false")
    # a pasted print folder link is rebuilt from its folder id: a javascript: link can't be stored
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password="tirupur-2026-x", drive=Drive("", "", tmp_path / "data"))
    rows = cloud.set_print_rows([{"category": "Men", "top": "HalfSleeveTee", "link": "javascript:alert(document.cookie)//?id=AAAAAAAAAAAA"}])
    assert rows[0]["link"] == "https://drive.google.com/drive/folders/AAAAAAAAAAAA"
