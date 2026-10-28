import io
import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from types import SimpleNamespace

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


def test_every_model_number_has_its_own_face():
    for category in faces.FEATURES:
        n = faces.capacity(category)
        assert n > 500000  # 1,000 new models a day for over a year
        sample = list(range(1, 20001)) + list(range(n - 5000, n + 1))
        assert len({tuple(faces.look(category, i).values()) for i in sample}) == len(sample)


def test_ledger_never_gives_a_face_twice(tmp_path):
    led = studio.ModelLedger(tmp_path)
    assert led.next_free("Men") == 1
    a = studio.Spec(category="Men", top_colours=["Red"], models=300, first_model=1, name="a")
    led.reserve(a)
    assert led.next_free("Men") == 301 and led.next_free("Women") == 1
    assert led.clash(studio.Spec(category="Men", top_colours=["Red"], models=5, first_model=299, name="b"))["batch"] == "a"
    assert led.clash(studio.Spec(category="Men", top_colours=["Red"], models=5, first_model=301, name="b")) is None
    assert led.clash(a) is None  # re-running the same batch continues it
    (tmp_path / "old").mkdir()  # a batch from before the ledger: read from its spec.json
    (tmp_path / "old" / "spec.json").write_text(json.dumps({"category": "Men", "first_model": 301, "models": 10, "name": "old"}))
    assert led.next_free("Men") == 311


def test_ledger_survives_batches_starting_together(tmp_path):
    led = studio.ModelLedger(tmp_path)
    specs = [studio.Spec(category=c, top_colours=["Red"], models=3, first_model=1, name=f"b{i}")
             for i, c in enumerate(["Men", "Women", "Kids_Boys", "Kids_Girls"] * 5)]
    ts = [threading.Thread(target=led.reserve, args=(s,)) for s in specs]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(json.loads((tmp_path / "_models.json").read_text())) == 20


def test_top_only_shot_is_above_the_knee_and_recolours_only_the_top(tmp_path):
    s = studio.Spec(top_colours=["Red", "White"], shot="top", bottom="Shorts", bottom_colour="Black", models=1)
    assert s.pairs() == [("Red", None), ("White", None)] and s.name.endswith("_top_only")
    p = studio.prompt(s, 1, s.bases()[0], "front_main", "none")
    assert "just above the knees" in p and "Full body head to shoes" not in p and "plain black regular-fit knit shorts" in p
    assert "sneakers" not in p.split("Avoid:")[0] and "feet or shoes in frame" in p.split("Avoid:")[1]
    full = studio.prompt(studio.Spec(top_colours=["Red"]), 1, ("dark", "dark"), "front_main", "none")
    assert "Full body head to shoes" in full and "sneakers" in full
    counts = studio.Batch(s, out=tmp_path, generate=fake_generate).run()
    assert counts["ai_done"] == 2 and counts["recoloured"] == 2 and counts["errors"] == 0


def test_plan_bases_and_pairs():
    s = studio.Spec(top_colours=["Yellow", "Navy", "White"], bottom_colours=["Black", "Light Pink"], pairing="each", models=300)
    assert len(s.pairs()) == 6 and s.bases()[0] in s.bases() and len(s.bases()) == 4
    assert s.plan() == {"models": 300, "bases_per_model": 4, "colour_pairs": 6, "poses": 1,
                        "ai_images": 1200, "recoloured_images": 1800, "disk_gb": 13.1}
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
    ref = studio.prompt(s, 12, ("dark", "light"), "front_main", "face")
    assert ref.startswith("Use the reference image ONLY for identity")
    # with a reference photo the hair comes from the photo only, never from a written hairstyle that may differ
    assert faces.hairstyle("Women", 12) not in ref and "hairstyle exactly as in the reference photo" in ref


def test_custom_poses_and_extra_instructions():
    s = studio.Spec(top_colours=["Red"], custom_poses={"Sitting Wave": "sitting cross-legged,   waving"},
                    poses=["front_main", "sitting_wave"], extra="big laughing expression", models=1)
    assert s.custom_poses == {"sitting_wave": "sitting cross-legged, waving"}
    p = studio.prompt(s, 1, ("dark", "dark"), "sitting_wave", "front")
    assert "sitting cross-legged, waving." in p and "Also: big laughing expression." in p
    edited = studio.Spec(top_colours=["Red"], custom_poses={"front_main": "standing with hands on hips"})
    assert "standing with hands on hips" in studio.prompt(edited, 1, ("dark", "dark"), "front_main", "none")
    with pytest.raises(ValueError):
        studio.Spec(top_colours=["Red"], poses=["front_main", "dancing"])  # not defined
    with pytest.raises(ValueError):
        studio.Spec(top_colours=["Red"], custom_poses={"empty": "  "})


def test_cost_estimate():
    from jagimage import pricing
    s = studio.Spec(top_colours=["Black", "White", "Red", "Navy", "Yellow"], models=300, poses=["front_main", "back"], gemini_model="pro")
    c = pricing.estimate(s)  # 300 models x 2 bases x 2 poses = 1200 Gemini 3 Pro 2K images
    assert s.plan()["ai_images"] == 1200 and c["per_image_usd"] == 0.137
    assert c["with_retries_usd"] == round(1200 * 0.137 * 1.1, 2) and c["per_photo_usd"] == round(c["with_retries_usd"] / 3000, 4)
    assert pricing.estimate(studio.Spec(top_colours=["Red"], provider="openai", quality="max"))["per_image_usd"] is None
    flash = pricing.estimate(studio.Spec(top_colours=["Red"]))  # Flash is the default: $0.101 + input at 2K
    assert flash["per_image_usd"] == 0.102 and flash["model"] == "gemini-3.1-flash-image"


def test_hands_free_pose():
    s = studio.Spec(top_colours=["Red"], poses=["front_main", "back", "handpocket"])
    front = studio.prompt(s, 1, ("dark", "dark"), "front_main", "none")
    assert studio.HANDS_FREE in front and "hands in pockets" in front.split("Avoid:")[1]
    assert studio.HANDS_FREE in studio.prompt(s, 1, ("dark", "dark"), "back", "front")
    pocket = studio.prompt(s, 1, ("dark", "dark"), "handpocket", "front")
    assert studio.HANDS_FREE not in pocket and "hands in pockets" not in pocket  # that pose needs the pocket
    off = studio.Spec(top_colours=["Red"], hands_free=False)
    assert studio.HANDS_FREE not in studio.prompt(off, 1, ("dark", "dark"), "front_main", "none")


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


def test_seam_between_top_and_bottom_keeps_no_base_colour(tmp_path):
    """Pixels where the top meets the bottom are garment: each garment's rim check must not leave
    them unpainted in the base colours (a dotted mint/lilac line along the hem)."""
    spec = studio.Spec(top_colours=["Red"], pairing="match", models=1, poses=["front_main"], same_models=True)
    def gen(prompt, refs):  # with the dull grey-lilac shadow a real hem casts on the bottom
        img = Image.open(io.BytesIO(fake_generate(prompt, refs))).convert("RGB")
        ImageDraw.Draw(img).rectangle((96, 248, 204, 250), fill="#6D6783")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    b = studio.Batch(spec, out=tmp_path, generate=gen, workers=1)
    b.run()
    out = np.asarray(Image.open(next(tmp_path.glob("*/m0001/Red__Red/front_main.png"))).convert("RGB"))
    base = np.asarray(Image.open(next(tmp_path.glob("*/m0001/_base/*/front_main.png"))).convert("RGB"))
    seam = (slice(244, 257), slice(100, 200))  # the hem: top above, bottom below
    unpainted = np.abs(out[seam].astype(int) - base[seam].astype(int)).max(-1) < 15
    assert unpainted.sum() == 0


def test_jpg_only_saves_disk_and_still_resumes(tmp_path):
    s = studio.Spec(top_colours=["Red", "White"], models=1, keep_png=False)
    b = studio.Batch(s, out=tmp_path, generate=fake_generate)
    assert b.run()["recoloured"] == 2
    rows = b.results()
    assert len(rows) == 2 and all(r["png"] is None and r["jpg"] for r in rows)
    assert not list((tmp_path / s.name).glob("m0001/Red__Red/*.png"))
    assert studio.Batch(s, out=tmp_path, generate=fake_generate).run()["recolour_skipped"] == 2
    big = studio.Spec(top_colours=["Red", "White"], models=1000, poses=["front_main", "back"], keep_png=False).plan()
    assert big["ai_images"] == 4000 and big["disk_gb"] == 17.4  # 4,000 bases x 4.1 MB + 4,000 JPGs x 0.25 MB


def test_finished_photos_are_copied_to_a_drive_folder(tmp_path):
    s = studio.Spec(top_colours=["Red"], models=2)
    drive = tmp_path / "My Drive" / "Jag Studio"
    studio.Batch(s, out=tmp_path / "work", generate=fake_generate, export=str(drive), export_what="both").run()
    got = sorted(p.relative_to(drive).as_posix() for p in drive.rglob("*.*"))
    assert got == [f"{s.name}/Red__Red/m000{m}_front_main.{e}" for m in (1, 2) for e in ("jpg", "png")]


def test_adding_colours_later_keeps_the_same_face(tmp_path):
    first = studio.Spec(top_colours=["Navy"], bottom_colours=["Black"], pairing="each", models=1)
    studio.Batch(first, out=tmp_path, generate=fake_generate).run()
    calls = []
    gen = lambda p, refs: (calls.append((p, refs)), fake_generate(p, refs))[1]
    more = studio.Spec(top_colours=["Navy"], bottom_colours=["Yellow"], pairing="each", models=1)  # needs a new base
    assert more.bases() == [("dark", "light")]
    studio.Batch(more, out=tmp_path, generate=gen).run()
    (prompt, refs), = calls
    assert refs == [str(tmp_path / first.name / "m0001" / "_base" / "dark-dark" / "front_main.png")]
    assert prompt.startswith("Use the reference image ONLY for identity")


def test_out_of_credits_stops_the_whole_batch(tmp_path):
    calls = []

    def broke(p, refs):
        calls.append(1)
        raise RuntimeError("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted.")

    b = studio.Batch(studio.Spec(top_colours=["Navy"], models=20), out=tmp_path, generate=broke, workers=2)
    b.run()
    assert "out of credits" in b.fatal and len(calls) <= 2 and b.counts["errors"] <= 2
    assert studio.account_problem(RuntimeError("429 RESOURCE_EXHAUSTED rate limit")) is None
    assert studio.is_rate_limit(RuntimeError("429 RESOURCE_EXHAUSTED rate limit"))


def test_wrong_colours_from_the_ai_are_retried(tmp_path):
    s = studio.Spec(top_colours=["Navy"], bottom_colours=["Yellow"], pairing="each", models=1)
    tries = []

    def gen(p, refs):
        tries.append(1)
        return fake_generate(p.replace("#C4A3EE", "#EDEDED"), refs) if len(tries) == 1 else fake_generate(p, refs)

    counts = studio.Batch(s, out=tmp_path, generate=gen, retries=2).run()
    assert len(tries) == 2 and counts["ai_done"] == 1 and counts["errors"] == 0


def test_photo_off_the_studio_backdrop_is_made_again(tmp_path):
    """Right base colours but the model outdoors (Flash once ignored the studio backdrop): rejected."""
    s = studio.Spec(top_colours=["Navy"], models=1)
    tries = []

    def gen(p, refs):
        tries.append(1)
        data = fake_generate(p, refs)
        if len(tries) == 1:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            d = ImageDraw.Draw(img)
            d.rectangle((0, 0, 299, 20), fill="#6E8B3D")       # trees along the top
            d.rectangle((0, 0, 25, 280), fill="#5C7A32")       # and down the sides
            d.rectangle((274, 0, 299, 280), fill="#7A5C3A")
            buf = io.BytesIO()
            img.save(buf, "PNG")
            data = buf.getvalue()
        return data

    counts = studio.Batch(s, out=tmp_path, generate=gen, retries=2).run()
    assert len(tries) == 2 and counts["ai_done"] == 1 and counts["errors"] == 0


def test_collage_and_washed_out_base_photos_are_made_again(tmp_path):
    """Seen with Flash: a 4-panel product infographic, and lilac trousers so pale they were almost
    white (their edge can't be told from the backdrop). Both are rejected and asked for again."""
    s = studio.Spec(top_colours=["White"], models=1)  # light colours: mint top, lilac bottom base
    tries = []

    def gen(p, refs):
        tries.append(1)
        if len(tries) == 1:  # four small figures side by side
            one = Image.open(io.BytesIO(fake_generate(p, refs))).convert("RGB").resize((150, 200))
            img = Image.new("RGB", (300, 400), "#EDEDED")
            for x, y in ((0, 0), (150, 0), (0, 200), (150, 200)):
                img.paste(one, (x, y))
        elif len(tries) == 2:  # washed-out lilac bottom
            img = Image.open(io.BytesIO(fake_generate(p.replace("#C4A3EE", "#D6CCE2"), refs))).convert("RGB")
        else:
            return fake_generate(p, refs)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    counts = studio.Batch(s, out=tmp_path, generate=gen, retries=3).run()
    assert len(tries) == 3 and counts["ai_done"] == 1 and counts["errors"] == 0


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
        for _ in range(600):
            st = json.loads(_get(f"{base}/api/jobs/{job['id']}")[1])
            if not st["running"]:
                break
            time.sleep(0.1)
        assert st["counts"]["recoloured"] == 6 and st["counts"]["errors"] == 0
        res = json.loads(_get(f"{base}/api/batches/{st['name']}/results")[1])
        assert res["total"] == 6
        assert _get(f"{base}/thumbs/{st['name']}/{res['rows'][0]['png']}")[0] == 200
        pose = {**body, "custom_poses": {"wave": "waving at the camera"}, "poses": ["front_main", "wave"]}
        code, pr = _post(base + "/api/prompt", {**pose, "preview_pose": "wave", "preview_model": 9, "tops": ["Hoodie"]})  # as the page sends it
        assert code == 200 and "waving at the camera" in pr["prompt"] and pr["model"] == 9 and pr["reference_images"]
        code, pv = _post(base + "/api/preview", {**pose, "preview_pose": "wave", "preview_model": 9, "tops": ["Hoodie"]})
        assert code == 200
        for _ in range(600):
            pv = json.loads(_get(f"{base}/api/preview/{pv['id']}")[1])
            if pv["done"]:
                break
            time.sleep(0.1)
        assert pv["counts"]["ai_done"] == 2 and pv["result"].endswith("wave.png") and not pv["error"]
        assert _get(f"{base}/thumbs/_previews/{pv['result']}")[0] == 200
        code, plan2 = _post(base + "/api/plan", body)  # the next batch gets new faces automatically
        assert plan2["first_model"] == 4 and plan2["auto"] and not plan2["clash"]
        code, clash = _post(base + "/api/jobs", {**body, "first_model": 2, "name": "other"})
        assert code == 400 and "already in batch" in clash["error"]
        code, again = _post(base + "/api/jobs", {**body, "first_model": 2, "name": "other", "stage": "redo"})
        assert code == 200  # recolouring again makes no faces, so models in another batch are fine
        for _ in range(300):
            if not json.loads(_get(f"{base}/api/jobs/{again['id']}")[1])["running"]:
                break
            time.sleep(0.1)
        dash = json.loads(_get(base + "/api/dashboard")[1])
        row = next(b for b in dash["batches"] if b["name"] == st["name"])
        assert row["photos"] == 6 and row["checked"] == 6 and row["passed"] + row["flagged"] == 6 and row["ai_images"] == st["counts"]["ai_done"]
        assert dash["totals"]["photos"] >= 6 and not dash["running"]
        code, redo = _post(f"{base}/api/batches/{st['name']}/redo", {})  # the Dashboard's "Recolour again"
        assert code == 200 and redo["stage"] == "recolor"
        for _ in range(300):
            st2 = json.loads(_get(f"{base}/api/jobs/{redo['id']}")[1])
            if not st2["running"]:
                break
            time.sleep(0.1)
        assert st2["counts"]["recoloured"] == 6 and st2["counts"]["errors"] == 0
        z = zipfile.ZipFile(io.BytesIO(_get(f"{base}/api/batches/{st['name']}/zip?what=jpg")[1]))
        assert len(z.namelist()) == 6 and all(n.endswith(".jpg") for n in z.namelist())
        with pytest.raises(urllib.error.HTTPError):
            _get(f"{base}/files/{st['name']}/..%2F..%2Fetc%2Fpasswd")
    finally:
        httpd.shutdown()


class FakeGenai:
    """Stands in for google-genai's client: Files API and Batch API, run on fake_generate."""

    def __init__(self, polls_before_done=1):
        self.store, self.jobs, self.created, self.polls = {}, {}, 0, polls_before_done
        outer = self

        class Files:
            def upload(self, file, config=None):
                name = f"files/{len(outer.store)}"
                outer.store[name] = open(file, "rb").read()
                return SimpleNamespace(name=name, uri=f"https://fake/{name}", mime_type="image/png")

            def download(self, file):
                return outer.store[file]

        class Batches:
            def create(self, model, src, config=None):
                import base64
                outer.created += 1
                lines = []
                for line in outer.store[src].decode().splitlines():
                    row = json.loads(line)
                    parts = row["request"]["contents"][0]["parts"]
                    text = parts[-1]["text"]
                    refs = [p["fileData"]["fileUri"] for p in parts[:-1]]
                    png = base64.b64encode(fake_generate(text, refs)).decode()
                    lines.append(json.dumps({"key": row["key"], "response": {"candidates": [
                        {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": png}}]}}]}}))
                out = f"files/out{outer.created}"
                outer.store[out] = "\n".join(lines).encode()
                name = f"batches/{outer.created}"
                outer.jobs[name] = {"left": outer.polls, "out": out, "n": len(lines)}
                return SimpleNamespace(name=name)

            def get(self, name):
                j = outer.jobs[name]
                done = j["left"] <= 0
                j["left"] -= 1
                return SimpleNamespace(name=name, state=SimpleNamespace(name="JOB_STATE_SUCCEEDED" if done else "JOB_STATE_RUNNING"),
                                       dest=SimpleNamespace(file_name=j["out"]), error=None)

        self.files, self.batches = Files(), Batches()


def batch_engine(client):
    from jagimage.providers import GeminiProvider
    provider = GeminiProvider(client=client, model="gemini-3-pro-image")

    def generate(p, refs):
        raise AssertionError("batch mode must not make single calls")

    generate.run_batch = lambda jobs, **kw: provider.run_batch(jobs, poll=0, **kw)
    return generate


def test_batch_mode_makes_everything_in_rounds(tmp_path):
    client = FakeGenai()
    s = studio.Spec(top_colours=["Navy", "Yellow"], models=3, poses=["front_main", "back"], batch=True)
    assert s.bases() and len(s.bases()) == 2
    counts = studio.Batch(s, out=tmp_path, generate=batch_engine(client)).run()
    # round 1: 3 first fronts; round 2: 3 other fronts + 3 first-base backs; round 3: 3 other backs
    assert client.created == 3 and [j["n"] for j in client.jobs.values()] == [3, 6, 3]
    assert counts["ai_done"] == 12 and counts["recoloured"] == 12 and counts["errors"] == 0
    assert not json.loads((tmp_path / s.name / "_batch_jobs.json").read_text())  # all collected
    again = studio.Batch(s, out=tmp_path, generate=batch_engine(client)).run()
    assert client.created == 3 and again["ai_skipped"] == 12  # nothing re-sent


def test_batch_job_is_collected_after_a_restart(tmp_path):
    import threading as th
    from jagimage.providers import GeminiProvider
    client = FakeGenai(polls_before_done=5)
    provider = GeminiProvider(client=client, model="gemini-3-pro-image")
    state = tmp_path / "_batch_jobs.json"
    stop = th.Event()
    stop.set()  # the app stops while Google is still working on the job
    jobs = [("m0001|dark-dark|front_main", studio.prompt(studio.Spec(top_colours=["Navy"]), 1, ("dark", "dark"), "front_main", "none"), [])]
    assert provider.run_batch(jobs, stop=stop, state_file=state, poll=0) == {}
    assert list(json.loads(state.read_text())) == ["batches/1"]
    got = provider.run_batch(jobs, state_file=state, poll=0)  # next start: same job, not a new one
    assert client.created == 1 and got["m0001|dark-dark|front_main"][:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads(state.read_text()) == {}


def test_big_rounds_are_split_into_jobs_under_the_quota(tmp_path):
    from jagimage.providers import GeminiProvider
    client = FakeGenai()
    provider = GeminiProvider(client=client, model="gemini-3-pro-image")
    sp = studio.Spec(top_colours=["Navy"])
    jobs = [(f"m{i:04d}|dark-dark|front_main", studio.prompt(sp, i, ("dark", "dark"), "front_main", "none"), []) for i in range(1, 8)]
    got = provider.run_batch(jobs, state_file=tmp_path / "s.json", poll=0, chunk=3)
    assert client.created == 3 and [j["n"] for j in client.jobs.values()] == [3, 3, 1] and len(got) == 7


def test_batch_mode_is_half_price_and_gemini_only():
    from jagimage import pricing
    full = pricing.estimate(studio.Spec(top_colours=["Navy"], models=100))
    half = pricing.estimate(studio.Spec(top_colours=["Navy"], models=100, batch=True))
    assert half["per_image_usd"] == round(full["per_image_usd"] / 2, 4) and half["batch"]
    oa = pricing.estimate(studio.Spec(top_colours=["Navy"], models=100, provider="openai", quality="medium"))
    oa_half = pricing.estimate(studio.Spec(top_colours=["Navy"], models=100, provider="openai", quality="medium", batch=True))
    assert oa_half["per_image_usd"] == round(oa["per_image_usd"] / 2, 4)
    with pytest.raises(ValueError):
        studio.Spec(top_colours=["Navy"], provider="runpod", batch=True)


class FakeOpenAIBatches:
    """client.files / client.batches like OpenAI's Batch API, answering image requests with fake_generate."""

    def __init__(self, polls_before_done=1, fail_key=None):
        self.store, self.batches_made, self.uploads, self.fail_key = {}, [], [], fail_key
        self.polls, outer = polls_before_done, self

        class Files:
            def create(self, file, purpose):
                fid = f"file-{len(outer.store) + 1}"
                data = file[1].read() if isinstance(file, tuple) else file.read()
                outer.store[fid] = data
                outer.uploads.append(purpose)
                return SimpleNamespace(id=fid)

            def content(self, fid):
                return SimpleNamespace(text=outer.store[fid].decode())

        class Batches:
            def create(self, input_file_id, endpoint, completion_window, metadata=None):
                assert completion_window == "24h"
                ok, bad = [], []
                for line in outer.store[input_file_id].decode().splitlines():
                    row = json.loads(line)
                    assert row["url"] == endpoint and row["method"] == "POST"
                    body = row["body"]
                    assert body["output_format"] == "png" and ("images" in body) == (endpoint == "/v1/images/edits")
                    refs = [i["file_id"] for i in body.get("images", [])]
                    if row["custom_id"] == outer.fail_key:
                        bad.append(json.dumps({"custom_id": row["custom_id"], "response": {"status_code": 400, "body": {
                            "error": {"message": "Your request was rejected by the safety system."}}}}))
                        continue
                    png = base64.b64encode(fake_generate(body["prompt"], refs)).decode()
                    ok.append(json.dumps({"custom_id": row["custom_id"], "response": {"status_code": 200, "body": {"data": [{"b64_json": png}]}}}))
                bid = f"batch_{len(outer.batches_made) + 1}"
                outer.store[bid + "_out"] = "\n".join(ok).encode()
                outer.store[bid + "_err"] = "\n".join(bad).encode()
                outer.batches_made.append({"id": bid, "endpoint": endpoint, "n": len(ok) + len(bad), "left": outer.polls})
                return SimpleNamespace(id=bid)

            def retrieve(self, bid):
                b = next(x for x in outer.batches_made if x["id"] == bid)
                done = b["left"] <= 0
                b["left"] -= 1
                return SimpleNamespace(id=bid, status="completed" if done else "in_progress", errors=None,
                                       output_file_id=bid + "_out" if done else None, error_file_id=bid + "_err" if done else None)

        self.files, self.batches = Files(), Batches()


def openai_batch_engine(client):
    from jagimage.providers import OpenAIProvider
    provider = OpenAIProvider(client=client, model="gpt-image-test", quality="medium")

    def generate(p, refs):
        raise AssertionError("batch mode must not make single calls")

    generate.run_batch = lambda jobs, **kw: provider.run_batch(jobs, poll=0, **kw)
    return generate


def test_openai_batch_mode_makes_everything_in_rounds(tmp_path):
    client = FakeOpenAIBatches()
    s = studio.Spec(top_colours=["Navy", "Yellow"], models=2, poses=["front_main", "back"], provider="openai", batch=True)
    counts = studio.Batch(s, out=tmp_path, generate=openai_batch_engine(client)).run()
    assert counts["ai_done"] == 8 and counts["recoloured"] == 8 and counts["errors"] == 0
    # first fronts have no reference photo (generations); everything after copies one (edits)
    assert client.batches_made[0]["endpoint"] == "/v1/images/generations"
    assert all(b["endpoint"] == "/v1/images/edits" for b in client.batches_made[1:])
    assert "vision" in client.uploads and "batch" in client.uploads
    assert not json.loads((tmp_path / s.name / "_batch_jobs.json").read_text())  # all collected


def test_openai_batch_is_collected_after_a_restart_and_reports_refusals(tmp_path):
    import threading as th
    from jagimage.providers import OpenAIProvider
    keys = ["m0001|dark-dark|front_main", "m0002|dark-dark|front_main"]
    client = FakeOpenAIBatches(polls_before_done=5, fail_key=keys[1])
    provider = OpenAIProvider(client=client, model="gpt-image-test", quality="medium")
    state = tmp_path / "_batch_jobs.json"
    stop = th.Event()
    stop.set()  # the app stops while OpenAI is still working on the batch
    text = studio.prompt(studio.Spec(top_colours=["Navy"]), 1, ("dark", "dark"), "front_main", "none")
    jobs = [(k, text, []) for k in keys]
    assert provider.run_batch(jobs, stop=stop, state_file=state, poll=0) == {}
    assert list(json.loads(state.read_text())) == ["openai:batch_1"]
    got = provider.run_batch(jobs, state_file=state, poll=0)  # next start: same batch, not a new one
    assert len(client.batches_made) == 1 and got[keys[0]][:8] == b"\x89PNG\r\n\x1a\n"
    assert isinstance(got[keys[1]], RuntimeError) and "safety system" in str(got[keys[1]])
    assert json.loads(state.read_text()) == {}


def test_same_models_in_another_style_keep_their_face(tmp_path):
    tee = studio.Spec(category="Men", top="HalfSleeveTee", bottom="Trackpants", top_colours=["Navy"], models=2)
    studio.Batch(tee, out=tmp_path, generate=fake_generate).run()
    face = tmp_path / "_faces" / "Men" / "m0001.png"
    assert face.exists()  # the first front ever made for model 1 is now his face for every style
    calls = []
    gen = lambda p, refs: (calls.append((p, refs)), fake_generate(p, refs))[1]
    hoodie = studio.Spec(category="Men", top="Hoodie", bottom="Trackpants", top_colours=["Navy"], models=2, same_models=True)
    assert studio.ModelLedger(tmp_path).clash(hoodie) is None
    counts = studio.Batch(hoodie, out=tmp_path, generate=gen).run()
    assert counts["ai_done"] == 2 and counts["recoloured"] == 2
    fronts = [(p, r) for p, r in calls]
    assert sorted(r for _, r in fronts) == [[str(face)], [str(tmp_path / "_faces" / "Men" / "m0002.png")]]  # models run in parallel
    assert all(p.startswith("Use the reference image ONLY for identity") and "pullover hoodie" in p for p, _ in fronts)


def test_server_runs_the_same_models_in_several_styles(tmp_path):
    app, httpd = serve(port=0, out=str(tmp_path), engine=lambda spec: fake_generate)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        body = {"category": "Men", "top": "HalfSleeveTee", "bottom": "Trackpants", "top_colours": ["Navy", "White"],
                "models": 2, "tops": ["FullSleeveTee", "SleevelessTee", "Hoodie", "Sweatshirt"]}
        code, plan = _post(base + "/api/plan", dict(body))
        one = _post(base + "/api/plan", {k: v for k, v in body.items() if k != "tops"})[1]
        assert code == 200 and plan["styles"] == 5 and plan["ai_images"] == 5 * one["ai_images"]
        assert abs(plan["cost"]["with_retries_usd"] - 5 * one["cost"]["with_retries_usd"]) < 0.05
        code, job = _post(base + "/api/jobs", dict(body))
        assert code == 200 and len(job["names"]) == 5
        for _ in range(600):
            st = json.loads(_get(f"{base}/api/jobs/{job['id']}")[1])
            if not st["running"]:
                break
            time.sleep(0.1)
        assert st["counts"]["recoloured"] == 5 * 2 * 2 and st["counts"]["errors"] == 0
        assert sorted(p.name for p in (tmp_path / "_faces" / "Men").glob("*.png")) == ["m0001.png", "m0002.png"]
        nxt = _post(base + "/api/plan", {k: v for k, v in body.items() if k != "tops"})[1]
        assert nxt["first_model"] == 3  # new batches still get new faces
    finally:
        httpd.shutdown()


def test_one_base_per_model_halves_the_ai_images(tmp_path):
    colours = ["Black", "White", "Light Pink", "Red", "Navy", "Green", "Lavender", "Orange", "Yellow"]
    two = studio.Spec(top_colours=colours, models=1000, poses=["front_main", "back"])
    one = studio.Spec(top_colours=colours, models=1000, poses=["front_main", "back"], one_base=True)
    assert two.plan()["ai_images"] == 4000 and one.plan()["ai_images"] == 2000 and one.bases() == [("light", "light")]
    small = studio.Spec(top_colours=["Navy", "Black", "White"], models=1, one_base=True)
    counts = studio.Batch(small, out=tmp_path, generate=fake_generate).run()
    assert counts["ai_done"] == 1 and counts["recoloured"] == 3 and counts["errors"] == 0
    for c in ["Navy", "Black", "White"]:
        qc = json.loads((tmp_path / small.name / "m0001" / f"{c}__{c}" / "qc.json").read_text())["front_main"]
        assert qc["top"]["deltaE"] < 3 and qc["bottom"]["deltaE"] < 3


def test_every_photo_gets_a_quality_check_and_bleeding_is_flagged(tmp_path):
    from jagimage.recolor import quality_check
    spec = studio.Spec(top_colours=["Red", "Navy"], models=1, poses=["front_main"], keep_png=False)
    b = studio.Batch(spec, out=tmp_path, generate=fake_generate, workers=1)
    b.run()
    for qc in b.root.glob("m0001/*/qc.json"):
        check = json.loads(qc.read_text())["front_main"]["check"]
        assert check["passed"] and check["bleed_px"] <= 10 and check["base_left_px"] == 0, (qc, check)
    assert b.counts["qc_flagged"] == 0
    # a recolour that spills onto the backdrop, and one that leaves the base colour in, are caught
    before = Image.open(io.BytesIO(fake_generate("top mint (#8FD3A8), bottom lilac (#C4A3EE)", [])))
    mask = np.zeros((400, 300), np.float32)
    mask[125:245, 100:200] = 1
    spilled = np.asarray(before).copy()
    spilled[125:245, 100:200] = (205, 0, 26)
    spilled[150:200, 203:212] = (205, 0, 26)  # 5-12 px beside the shirt
    check = quality_check(before, Image.fromarray(spilled), [mask], ["#8FD3A8"], ["#CD001A"])
    assert not check["passed"] and check["bleed_px"] > 60 and "bleeds" in check["problems"][0]
    untouched = quality_check(before, before, [mask], ["#8FD3A8"], ["#CD001A"])
    assert not untouched["passed"] and untouched["base_left_px"] > 40 and untouched["bleed_px"] == 0
    assert quality_check(before, before, [mask], ["#8FD3A8"], ["#6CC24A"])["base_left_px"] == 0  # green on mint: its own hue


def test_masks_from_older_code_are_made_again(tmp_path):
    from jagimage.recolor import MASK_VERSION
    spec = studio.Spec(top_colours=["Red"], models=1, poses=["front_main"], keep_png=False)
    b = studio.Batch(spec, out=tmp_path, generate=fake_generate, workers=1)
    b.run()
    src = b.base_path(1, spec.bases()[0], "front_main")
    ver, tm = src.parent / "front_main_mask_version.txt", src.parent / "front_main_top_mask.png"
    assert ver.read_text() == str(MASK_VERSION)
    ver.unlink()                         # as left by a version without mask versions
    tm.write_bytes(b"")                  # and an outdated mask that must not be used
    redo = studio.Batch(spec, out=tmp_path, generate=fake_generate, workers=1, redo=True)
    redo.run(stage="recolor")
    assert redo.counts["recoloured"] == 1 and redo.counts["errors"] == 0
    assert ver.read_text() == str(MASK_VERSION) and tm.stat().st_size > 0


def test_top_only_prompts_never_ask_for_a_full_body():
    import re
    spec = studio.Spec(category="Women", shot="top", top_colours=["Red"], models=1,
                       poses=["front_main", "back", "side", "threequarter", "handpocket"])
    for pose in spec.poses:
        for ref in ("none", "face", "front", "face+front"):
            text = studio.prompt(spec, 1, spec.bases()[0], pose, ref).split(" Avoid: ")[0]
            assert "just above the knees" in text, (pose, ref)
            assert not re.search(r"full[- ]body|one leg|head to shoes|sneaker", text, re.I), (pose, ref)
    full = studio.Spec(category="Women", top_colours=["Red"], models=1, poses=["front_main"])
    assert "Full body head to shoes" in studio.prompt(full, 1, full.bases()[0], "front_main", "none")


def test_openai_gets_plain_childrens_wording_gemini_unchanged():
    for cat in ("Kids_Girls", "Kids_Boys"):
        for i in range(1, 30):  # every face variant
            oa = studio.Spec(category=cat, top_colours=["Red"], models=1, first_model=i, poses=["front_main"], provider="openai")
            text = studio.prompt(oa, i, oa.bases()[0], "front_main", "none")
            assert text.startswith("Wholesome, family-friendly children's clothing catalogue photo")
            for word in ("chubby", "lips", "rosy", "slim adult body", "mature proportions", "makeup on child", "long legs", "toddler proportions"):
                assert word not in text, (cat, i, word)
        ge = studio.Spec(category=cat, top_colours=["Red"], models=1, poses=["front_main"], provider="gemini")
        assert "toddler proportions" in studio.prompt(ge, 1, ge.bases()[0], "front_main", "none")
    men = studio.Spec(category="Men", top_colours=["Red"], models=1, poses=["front_main"], provider="openai")
    assert not studio.prompt(men, 1, men.bases()[0], "front_main", "none").startswith("Wholesome")


def test_half_sleeve_starts_as_top_only():
    from jagimage.studio import default_shot
    assert default_shot("HalfSleeveTee") == "top" and default_shot("Hoodie") == "outfit"


def test_close_up_pose_from_a_guide_is_not_framed_head_to_toe():
    sp = studio.Spec(category="Men", top="HalfSleeveTee", bottom="Trackpants", top_colours=["Red"], poses=["front_main", "detail_closeup"],
                     custom_poses={"detail_closeup": "Close-up of the neckline showing fabric detail"})
    close = studio.prompt(sp, 1, sp.bases()[0], "detail_closeup", "face+front")
    assert studio.CLOSE_FRAMING in close and studio.FULL_FRAMING not in close and "full-body photo" not in close
    assert studio.FULL_FRAMING in studio.prompt(sp, 1, sp.bases()[0], "front_main", "none")


def make_print(path, size=(400, 300), bg="white"):
    from PIL import Image, ImageDraw
    art = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(art)
    d.rectangle((40, 40, size[0] - 40, size[1] - 40), fill="#D23C1E")
    d.ellipse((150, 100, 250, 200), fill=bg)  # a hole: the background colour inside means no ink there too
    art.save(path)
    return path


def test_print_is_placed_on_the_chest_and_its_white_background_is_not_ink(tmp_path):
    from PIL import Image
    from jagimage.prints import artwork, chest, place_print
    from jagimage.recolor import garment_mask, rgb_to_lab
    import numpy as np
    from test_gemini_bases import fake_photo
    photo = fake_photo("#1E4636")
    mask, _ = garment_mask(rgb_to_lab(np.asarray(photo)), "#1E4636")
    art = Image.open(make_print(tmp_path / "p.png"))
    a = np.asarray(artwork(art))
    assert a[5, 5, 3] == 0 and a[150, 200, 3] == 0 and a[60, 60, 3] == 255  # background and hole: no ink
    cx, neck, w = chest(mask)
    assert abs(cx - 150) < 8 and 115 <= neck <= 125 and 100 < w < 140
    out = np.asarray(place_print(photo, mask, art)).astype(int)
    before = np.asarray(photo).astype(int)
    changed = np.abs(out - before).sum(-1) > 30
    ys, xs = np.nonzero(changed)
    assert changed.sum() > 500 and ys.min() > neck and abs(xs.mean() - cx) < 6  # centred, below the neckline
    assert not changed[mask < 0.5].any()  # never on skin, hair or the backdrop
    red = out[changed]
    assert (red[:, 0] > red[:, 1] + 60).mean() > 0.9  # the print's own red, not mixed with the shirt


def test_prints_stage_prints_every_front_photo_of_a_batch(tmp_path):
    s = studio.Spec(top_colours=["Navy", "White"], models=1, poses=["front_main", "back"], keep_png=False)
    studio.Batch(s, out=tmp_path, generate=fake_generate).run()
    arts = [("Logo 1", make_print(tmp_path / "a.png")), ("237", make_print(tmp_path / "b.png", bg="black"))]
    asked = []
    prints = lambda cat, top, colour: (asked.append((cat, top, colour)), arts if colour == "Navy" else [])[1]
    counts = studio.Batch(s, out=tmp_path, prints=prints).run("prints")
    assert counts["printed"] == 2 and counts["errors"] == 0 and sorted(asked) == [(s.category, s.top, "Navy"), (s.category, s.top, "White")]
    navy = tmp_path / s.name / "m0001" / "Navy__Navy"
    assert (navy / "front_main_print_Logo_1_1500x1500.jpg").exists() and (navy / "front_main_print_237_1500x1500.jpg").exists()
    assert not list(navy.glob("back_print*"))  # front pose only
    rows = studio.Batch(s, out=tmp_path).results()
    plain = next(r for r in rows if r["top"] == "Navy" and r["pose"] == "front_main")
    assert plain["jpg"].endswith("front_main_1500x1500.jpg")  # the plain photo stays itself
    assert any(r["pose"] == "front_main_print_237" for r in rows)
    again = studio.Batch(s, out=tmp_path, prints=prints).run("prints")
    assert again["printed"] == 0  # already printed: nothing redone


def test_print_adjustment_moves_and_resizes_and_stays_within_limits(tmp_path):
    import numpy as np
    from PIL import Image
    from jagimage.prints import adjust, place_print
    from jagimage.recolor import garment_mask, rgb_to_lab
    from test_gemini_bases import fake_photo
    assert adjust(5, -5, 9) == {"dx": 0.3, "dy": -0.15, "scale": 1.6} and adjust(None, None, None) == {"dx": 0, "dy": 0, "scale": 1}
    photo = fake_photo("#1E4636")
    mask, _ = garment_mask(rgb_to_lab(np.asarray(photo)), "#1E4636")
    art = Image.open(make_print(tmp_path / "p.png"))

    def box(**kw):
        diff = np.abs(np.asarray(place_print(photo, mask, art, **kw)).astype(int) - np.asarray(photo).astype(int)).sum(-1) > 30
        ys, xs = np.nonzero(diff)
        return ys.min(), xs.mean(), diff.sum()
    y0, x0, n0 = box()
    y1, x1, n1 = box(dy=0.1, dx=0.1, scale=0.8)
    assert y1 > y0 + 5 and x1 > x0 + 5 and n1 < n0  # lower, to the right, smaller


def test_print_checker_fixes_the_placement_or_flags_it(tmp_path):
    s = studio.Spec(top_colours=["Navy"], models=1, poses=["front_main"], keep_png=False)
    studio.Batch(s, out=tmp_path, generate=fake_generate).run()
    arts = [("ok", make_print(tmp_path / "a.png")), ("stubborn", make_print(tmp_path / "b.png", size=(410, 300))),
            ("offline", make_print(tmp_path / "c.png", size=(380, 300)))]
    calls = []

    def checker(photo, art, measured):
        assert set(measured) == {"collar_gap", "hem_gap", "width", "off_centre", "hidden"}
        calls.append(art.size)
        if art.size == (380, 300):
            raise RuntimeError("no connection")
        if art.size == (410, 300):
            return {"ok": False, "issues": ["too high: touches the collar"], "dx": 0, "dy": 0.05, "scale": 1}
        first = sum(1 for c in calls if c == art.size) == 1
        return {"ok": not first, "issues": ["too high"] if first else [], "dx": 0, "dy": 0.08 if first else 0, "scale": 1}
    counts = studio.Batch(s, out=tmp_path, prints=lambda *a: arts, checker=checker).run("prints")
    assert counts["printed"] == 3 and counts["print_fixed"] == 2 and counts["print_flagged"] == 2
    qc = json.loads((tmp_path / s.name / "m0001" / "Navy__Navy" / "prints_qc.json").read_text())
    assert qc["front_main_print_ok"]["ok"] and qc["front_main_print_ok"]["adjust"]["dy"] == 0.08
    assert len(qc["front_main_print_stubborn"]["rounds"]) == 3 and qc["front_main_print_stubborn"]["adjust"]["dy"] == 0.1
    assert "no connection" in qc["front_main_print_offline"]["error"]
    rows = {r["pose"]: r for r in studio.Batch(s, out=tmp_path).results()}
    assert rows["front_main_print_ok"]["print_check"] == {"ok": True, "fixed": True, "error": None, "issues": []}
    assert rows["front_main_print_stubborn"]["print_check"]["issues"] == ["too high: touches the collar"]


def test_firm_rules_catch_a_print_on_the_collar_even_when_the_ai_says_ok(tmp_path):
    import numpy as np
    from PIL import Image
    from jagimage.prints import measure, rule_check
    from jagimage.recolor import garment_mask, rgb_to_lab
    from test_gemini_bases import fake_photo
    photo = fake_photo("#1E4636")
    mask, _ = garment_mask(rgb_to_lab(np.asarray(photo)), "#1E4636")
    art = Image.open(make_print(tmp_path / "p.png"))
    good = measure(mask, art)
    assert rule_check(good) == ([], {"dx": 0.0, "dy": 0.0, "scale": 1.0})
    high = measure(mask, art, dy=-0.12, scale=1.4)
    issues, fix = rule_check(high)
    assert "too high: touches the collar" in issues and "too big for the chest" in issues and fix["dy"] > 0.1 and fix["scale"] < 0.8
    pleased = lambda photo, art, measured: {"ok": True, "issues": [], "dx": 0, "dy": 0, "scale": 1}
    verdict = studio.judge_print(photo, art, mask, "chest_large", {"dx": 0, "dy": -0.12, "scale": 1.4}, pleased)
    assert not verdict["ok"] and verdict["dy"] == fix["dy"]
    # the batch's checker then moves it down and shrinks it until the rules pass
    s = studio.Spec(top_colours=["Navy"], models=1, poses=["front_main"], keep_png=False, print_dy=-0.12, print_scale=1.4)
    studio.Batch(s, out=tmp_path, generate=fake_generate).run()
    counts = studio.Batch(s, out=tmp_path, prints=lambda *a: [("p", tmp_path / "p.png")], checker=pleased).run("prints")
    qc = json.loads((tmp_path / s.name / "m0001" / "Navy__Navy" / "prints_qc.json").read_text())["front_main_print_p"]
    assert counts["print_fixed"] == 1 and qc["ok"] and qc["adjust"]["dy"] >= 0 and qc["adjust"]["scale"] < 1.2  # back below the collar


def test_oversized_drop_shoulder_tee_is_for_men_and_women_only(tmp_path):
    from jagimage.server import specs_from
    men = studio.Spec(category="Men", top="OversizedTee", bottom="Trackpants", top_colours=["Black"])
    assert studio.default_shot("OversizedTee") == "top"
    text = studio.prompt(men, 1, men.bases()[0], "front_main", "none")
    assert "DROP SHOULDERS" in text and "oversized" in text and "set-in shoulder seam at the natural shoulder" not in text
    studio.Spec(category="Women", top="OversizedTee", bottom="Trackpants", top_colours=["Black"])
    with pytest.raises(ValueError, match="Men and Women only"):
        studio.Spec(category="Kids_Boys", top="OversizedTee", bottom="Shorts", top_colours=["Navy"])
    # "every style" for kids leaves it out; for adults it is one of the styles with the same models
    ledger = studio.ModelLedger(tmp_path)
    kids = specs_from({"category": "Kids_Girls", "top": "HalfSleeveTee", "bottom": "Shorts", "top_colours": ["Navy"],
                       "tops": list(studio.TOPS)}, ledger)
    assert "OversizedTee" not in [s.top for s in kids] and len(kids) == 5
    adults = specs_from({"category": "Men", "top": "HalfSleeveTee", "bottom": "Trackpants", "top_colours": ["Navy"],
                         "tops": list(studio.TOPS)}, ledger)
    assert [s.top for s in adults][-1] == "OversizedTee" and len(adults) == 6


def test_top_only_trim_keeps_dark_folds_far_above_the_hem(tmp_path):
    """A navy bottom and a bottle-green top's dark side fold look alike: the trim must not follow the
    fold up to the sleeve (that part of the top was left in bottle green), only clear the bottom at the hem."""
    import numpy as np

    from jagimage.recolor import rgb_to_lab

    h, w = 300, 200
    rgb = np.full((h, w, 3), 230, np.uint8)
    rgb[40:200, 40:160] = (30, 70, 54)        # the top in bottle green
    rgb[200:300, 60:140] = (34, 40, 66)       # navy trackpants below the hem
    rgb[60:200, 95:105] = (28, 32, 40)        # a dark, nearly colourless fold running down to the hem
    top = np.zeros((h, w), np.float32)
    top[40:204, 40:160] = 1                    # the mask's shadow growth ran a little into the trackpants
    spec = studio.Spec(category="Men", top="OversizedTee", bottom="Trackpants", top_colours=["Navy"], bottom_colour="Navy", shot="top")
    out = studio.Batch(spec, out=tmp_path)._trim_fixed_bottom(rgb_to_lab(rgb), top, ("dark", None))
    assert out[60:150, 95:105].min() > 0.5    # the fold far above the hem stays part of the top
    assert out[202:204, 70:90].max() < 0.5    # the trackpants just below the hem are taken out


def test_face_test_changes_only_the_face_of_a_finished_base(tmp_path, monkeypatch):
    from jagimage import providers

    seen = []
    monkeypatch.setattr(providers.OpenAIProvider, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(providers.OpenAIProvider, "generate", lambda self, prompt, refs: seen.append((prompt, refs)) or b"png")
    app, httpd = serve(port=0, out=str(tmp_path), engine=lambda spec: fake_generate)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        code, job = _post(base + "/api/jobs", {"category": "Men", "top": "HalfSleeveTee", "bottom": "Trackpants",
                                              "top_colours": ["Navy"], "models": 1})
        for _ in range(600):
            st = json.loads(_get(f"{base}/api/jobs/{job['id']}")[1])
            if not st["running"]:
                break
            time.sleep(0.1)
        m = studio.Spec(**json.loads((tmp_path / st["name"] / "spec.json").read_text())).first_model
        assert _post(base + "/api/facetest", {"batch": "nope", "model": m})[0] == 400
        code, ft = _post(base + "/api/facetest", {"batch": st["name"], "model": m, "faces": [201, 202]})
        assert code == 200, ft
        for _ in range(100):
            ft = json.loads(_get(base + "/api/facetest")[1])
            if not ft["running"]:
                break
            time.sleep(0.05)
        assert len(ft["done"]) == 2 and not ft["errors"]
        assert all("Change ONLY the face" in p and r[0].endswith("front_main.png") for p, r in seen)
        assert _get(f"{base}/files/{ft['done'][0]}")[0] == 200
    finally:
        httpd.shutdown()
