import io
import json
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from jagimage import faces, studio
from jagimage.server import serve


def fake_generate(prompt, refs):
    """A figure whose top and bottom are in the base colours the prompt asks for."""
    top, bottom = re.findall(r"top [a-z ]+ \((#[0-9A-F]{6})\)(?:, bottom [a-z ]+ \((#[0-9A-F]{6})\))?", prompt)[0]
    img = Image.new("RGB", (300, 400), "#EDEDED")
    d = ImageDraw.Draw(img)
    d.ellipse((115, 30, 185, 110), fill="#C99B80")
    d.ellipse((112, 22, 188, 70), fill="#111111")
    d.polygon([(90, 120), (210, 120), (240, 200), (215, 210), (205, 250), (95, 250), (85, 210), (60, 200)], fill=top)
    d.rectangle((95, 250, 205, 330), fill=bottom or "#5A7FB0")
    d.rectangle((100, 360, 200, 380), fill="#FAFAFA")
    arr = np.asarray(img).astype(np.float32) * np.linspace(0.85, 1.08, 300)[None, :, None]
    out = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)).save(out, "PNG")
    return out.getvalue()


def test_300_models_are_all_different():
    for category in faces.FEATURES:
        assert len({faces.describe(category, i) for i in range(1, 301)}) == 300
        assert faces.describe(category, 7) == faces.describe(category, 7)  # stable, so batches can resume


def test_plan_bases_and_pairs():
    s = studio.Spec(top_colours=["Yellow", "Navy", "White"], bottom_colours=["Black", "Light Pink"], pairing="each", models=300)
    assert len(s.pairs()) == 6 and s.bases()[0] in s.bases() and len(s.bases()) == 4
    assert s.plan() == {"models": 300, "bases_per_model": 4, "colour_pairs": 6, "poses": 1,
                        "ai_images": 1200, "recoloured_images": 1800}
    same = studio.Spec(top_colours=["Yellow", "Navy"], models=300)  # bottom = top colour
    assert same.pairs() == [("Yellow", "Yellow"), ("Navy", "Navy")] and sorted(same.bases()) == [("dark", "dark"), ("light", "light")]
    jeans = studio.Spec(bottom="Jeans", top_colours=["Yellow", "Navy", "White"], models=2)
    assert jeans.pairs() == [("Yellow", None), ("Navy", None), ("White", None)] and len(jeans.bases()) == 2
    assert studio.colour_hex("#3a7d44", "PureCotton") == "#3A7D44" and studio.colour_hex("Maroon", "PureCotton") == "#490D14"
    with pytest.raises(ValueError):
        studio.Spec(top_colours=["Not a colour"])


def test_prompt_asks_for_a_real_person_in_both_base_colours():
    s = studio.Spec(category="Women", top="HalfSleeveTee", bottom="Trackpants", top_colours=["Red"], bottom_colours=["White"], pairing="each")
    p = studio.prompt(s, 12, ("dark", "light"), "front_main", "none")
    assert "real, living human model (not a mannequin" in p and "mannequin, dummy" in p.split("Avoid:")[1]
    assert "bottle green (#1E4636)" in p and "lilac (#C4A3EE)" in p and faces.describe("Women", 12) in p
    assert "trackpants" in p and "Joggers end in a plain straight hem" in p
    assert studio.prompt(s, 12, ("dark", "light"), "front_main", "face").startswith("Use the reference image ONLY for identity")


def test_batch_makes_bases_once_and_recolours_every_pair(tmp_path):
    s = studio.Spec(top_colours=["Yellow", "Navy"], bottom_colours=["Black", "Light Pink"], pairing="each", models=2)
    calls = []
    gen = lambda p, refs: (calls.append(refs), fake_generate(p, refs))[1]
    b = studio.Batch(s, out=tmp_path, generate=gen, workers=2)
    counts = b.run()
    assert counts["ai_done"] == 8 and counts["recoloured"] == 8 and counts["errors"] == 0
    # per model: the first base is made from the text, every other base uses it as the face reference
    assert sum(1 for r in calls if not r) == 2 and sum(1 for r in calls if r and r[0].endswith("front_main.png")) == 6
    rows = b.results()
    assert len(rows) == 8 and all(r["jpg"] for r in rows)
    qc = json.loads((tmp_path / s.name / "m0001" / "Yellow__Light_Pink" / "qc.json").read_text())["front_main"]
    assert qc["top"]["deltaE"] < 3 and qc["bottom"]["deltaE"] < 3
    assert Image.open(tmp_path / s.name / "m0001" / "Yellow__Light_Pink" / "front_main_1500x1500.jpg").size == (1500, 1500)
    again = studio.Batch(s, out=tmp_path, generate=gen).run()  # resumable: nothing is made twice
    assert again["ai_done"] == 0 and again["ai_skipped"] == 8 and again["recolour_skipped"] == 8


def test_finished_photos_are_copied_to_a_drive_folder(tmp_path):
    s = studio.Spec(top_colours=["Red"], models=2)
    drive = tmp_path / "My Drive" / "Jag Studio"
    studio.Batch(s, out=tmp_path / "work", generate=fake_generate, export=str(drive), export_what="both").run()
    got = sorted(p.relative_to(drive).as_posix() for p in drive.rglob("*.*"))
    assert got == [f"{s.name}/Red__Red/m000{m}_front_main.{e}" for m in (1, 2) for e in ("jpg", "png")]


def test_wrong_colours_from_the_ai_are_retried(tmp_path):
    s = studio.Spec(top_colours=["Navy"], bottom_colours=["Yellow"], pairing="each", models=1)
    tries = []

    def gen(p, refs):
        tries.append(1)
        return fake_generate(p.replace("#C4A3EE", "#EDEDED"), refs) if len(tries) == 1 else fake_generate(p, refs)

    counts = studio.Batch(s, out=tmp_path, generate=gen, retries=2).run()
    assert len(tries) == 2 and counts["ai_done"] == 1 and counts["errors"] == 0


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_studio_server_runs_a_batch(tmp_path):
    app, httpd = serve(port=0, out=str(tmp_path), engine=lambda spec: fake_generate)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        code, page = _get(base + "/")
        assert code == 200 and b"Jag Studio" in page
        opts = json.loads(_get(base + "/api/options")[1])
        assert "Kids_Girls" in opts["categories"] and "Navy" in opts["palettes"]["PureCotton"]
        body = {"category": "Kids_Girls", "top": "HalfSleeveTee", "bottom": "Shorts", "top_colours": ["Red", "White"],
                "bottom_colours": [], "pairing": "match", "models": 3}
        code, plan = _post(base + "/api/plan", body)
        assert code == 200 and plan["ai_images"] == 6 and plan["recoloured_images"] == 6
        assert _post(base + "/api/plan", {**body, "top_colours": []})[0] == 400
        code, job = _post(base + "/api/jobs", body)
        assert code == 200
        for _ in range(100):
            st = json.loads(_get(f"{base}/api/jobs/{job['id']}")[1])
            if not st["running"]:
                break
            time.sleep(0.1)
        assert st["counts"]["recoloured"] == 6 and st["counts"]["errors"] == 0
        res = json.loads(_get(f"{base}/api/batches/{st['name']}/results")[1])
        assert res["total"] == 6
        assert _get(f"{base}/thumbs/{st['name']}/{res['rows'][0]['png']}")[0] == 200
        z = zipfile.ZipFile(io.BytesIO(_get(f"{base}/api/batches/{st['name']}/zip?what=jpg")[1]))
        assert len(z.namelist()) == 6 and all(n.endswith(".jpg") for n in z.namelist())
        with pytest.raises(urllib.error.HTTPError):
            _get(f"{base}/files/{st['name']}/..%2F..%2Fetc%2Fpasswd")
    finally:
        httpd.shutdown()
