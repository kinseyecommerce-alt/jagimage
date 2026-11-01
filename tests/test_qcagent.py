import json

import numpy as np
from PIL import Image

from jagimage import qcagent, studio
from jagimage.cloud import DriveSync
from test_studio import fake_generate


class Eyes:
    """A stand-in for the vision check: fails the first `fail` questions about base photos, passes the rest."""

    def __init__(self, fail=0, photo_ok=True):
        self.fail, self.photo_ok, self.asked = fail, photo_ok, []

    def __call__(self, message, image):
        assert image[:2] == b"\xff\xd8"  # shown as a JPEG
        self.asked.append(message)
        if "recoloured" in message:
            return {"ok": self.photo_ok, "problems": [] if self.photo_ok else ["grey strip along the left sleeve"]}
        if self.fail:
            self.fail -= 1
            return {"ok": False, "problems": ["the top is tucked into the pants"]}
        return {"ok": True, "problems": []}


class FakeDrive:
    connected = True

    def __init__(self):
        self.files = []

    def upload(self, local, parts, name):
        self.files.append("/".join(parts + [name]))


def spec(**kw):
    return studio.Spec(**{"category": "Kids_Girls", "top": "Hoodie", "bottom": "Trackpants", "fabric": "CottonMix", "models": 1,
                          "colour_pairs": [["Red", "Red"]], "poses": ["front_main", "back"], "one_base": True,
                          "keep_png": False, "drive_layout": "catalogue", "free_after_upload": True, "qc_agent": True, **kw})


def test_base_points_ask_for_what_recolouring_cant_fix():
    pts = " ".join(qcagent.base_points(spec(), "front_main"))
    for want in ("untucked", "NO drawstring", "inside of the hood", "hair bows", "five fingers", "face"):
        assert want in pts
    assert "face" not in " ".join(qcagent.base_points(spec(), "back")).replace("surface", "")
    assert "hood" not in " ".join(qcagent.base_points(spec(top="HalfSleeveTee", category="Kids_Boys"), "front_main"))


def test_a_rejected_base_is_made_again(tmp_path):
    eyes = Eyes(fail=1)
    calls = []

    def gen(prompt, refs):
        calls.append(prompt)
        return fake_generate(prompt, refs)
    b = studio.Batch(spec(), out=tmp_path, generate=gen, workers=1, inspector=eyes)
    counts = b.run()
    assert counts["base_rejected"] == 1 and counts["errors"] == 0 and counts["recoloured"] == 2
    shoots = [p for p in calls if p != studio.RETOUCH_BOTTOMS]
    assert len(shoots) == 3  # the front twice (first one rejected), the back once
    qc = json.loads((b.base_path(1, ("light", "light"), "front_main").parent / "front_main_base_qc.json").read_text())
    assert qc["ok"] is True


def test_only_passed_photos_go_to_drive_and_a_person_can_approve_the_rest(tmp_path):
    b = studio.Batch(spec(), out=tmp_path, generate=fake_generate, workers=1, inspector=Eyes())
    b.run()
    qc_file = next(b.root.glob("m0001/Red__Red/qc.json"))
    rows = json.loads(qc_file.read_text())
    rows["back"]["check"].update(passed=False, held=True, problems=["grey strip"])  # as if the agent held it
    qc_file.write_text(json.dumps(rows))
    drive = FakeDrive()
    sync = DriveSync(drive, lambda: str(tmp_path))
    sync.run_once()
    assert drive.files and all("back" not in f for f in drive.files)
    assert (qc_file.parent / "back_2000x2000.jpg").exists()  # held: stays on the server, not uploaded
    assert studio.approve(b.root, ["m0001/Red__Red/back_2000x2000.jpg", "../../etc/passwd.jpg"]) == 1
    drive.files.clear()
    sync.run_once()
    assert sorted(drive.files) == ["Girls/Red/Co-ords/Hoodies set/m0001_back.jpg", "Girls/Red/Hoodies/m0001_back.jpg"]


def test_bleed_is_put_back_and_leftover_base_colour_painted():
    before = Image.new("RGB", (40, 40), (200, 200, 200))
    after = Image.new("RGB", (40, 40), (200, 200, 200))
    after.paste((220, 30, 30), (10, 10, 30, 30))      # the garment, recoloured red
    after.paste((210, 120, 120), (5, 10, 10, 30))     # red spilled onto the backdrop beside it
    bleed = np.zeros((40, 40), bool)
    bleed[10:30, 5:10] = True
    left = np.zeros((40, 40), bool)
    left[12:14, 12:14] = True
    got = []
    out = qcagent.fix_photo(before, after, {"_bleed": bleed, "_left": left},
                            lambda extra: (got.append(extra), after)[1])
    assert got and got[0][12:14, 12:14].all()  # the leftover pixels joined the garment for another recolour
    px = np.asarray(out)
    assert abs(int(px[20, 6, 1]) - 200) < 10       # backdrop got its own grey back
    assert tuple(px[20, 20]) == (220, 30, 30)      # the garment is untouched


def test_a_flagged_photo_the_agent_rejects_is_held(tmp_path, monkeypatch):
    real = studio.Batch._recolour_one
    from jagimage import recolor

    orig = recolor.quality_check

    def always_flag(*a, **k):  # every recolour still counts problems after fixing
        out = orig(*a, **k)
        out["problems"] = ["colour bleeds onto skin or background (500 px)"]
        out["passed"] = False
        return out
    monkeypatch.setattr(recolor, "quality_check", always_flag)
    b = studio.Batch(spec(poses=["front_main"]), out=tmp_path, generate=fake_generate, workers=1, inspector=Eyes(photo_ok=False))
    counts = b.run()
    assert counts["qc_held"] == 1 and real is studio.Batch._recolour_one
    check = json.loads(next(b.root.glob("m0001/Red__Red/qc.json")).read_text())["front_main"]["check"]
    assert check["held"] and not check["passed"] and check["agent"]["problems"] == ["grey strip along the left sleeve"]
    assert not qcagent.passed(next(b.root.glob("m0001/Red__Red/qc.json")), "front_main")
    b2 = studio.Batch(spec(poses=["front_main"]), out=tmp_path, generate=fake_generate, workers=1, inspector=Eyes(photo_ok=True), redo=True)
    assert b2.run(stage="recolor")["qc_held"] == 0  # the agent sees nothing wrong: it passes
    assert qcagent.passed(next(b.root.glob("m0001/Red__Red/qc.json")), "front_main")


def test_qc_stage_rechecks_existing_bases(tmp_path):
    studio.Batch(spec(qc_agent=False), out=tmp_path, generate=fake_generate, workers=1).run()
    calls = []

    def gen(prompt, refs):
        calls.append(prompt)
        return fake_generate(prompt, refs)
    b = studio.Batch(spec(), out=tmp_path, generate=gen, workers=1, inspector=Eyes(fail=1), redo=True)
    counts = b.run(stage="qc")
    assert counts["base_rejected"] == 1 and counts["recoloured"] == 2
    assert len([p for p in calls if p != studio.RETOUCH_BOTTOMS]) == 2  # the rejected front and its back, made again


def test_a_real_fault_is_fixed_by_an_ai_edit_and_then_passes(tmp_path, monkeypatch):
    from jagimage import recolor

    orig = recolor.quality_check

    def always_flag(*a, **k):
        out = orig(*a, **k)
        out["problems"], out["passed"] = ["colour bleeds onto skin or background (500 px)"], False
        return out
    monkeypatch.setattr(recolor, "quality_check", always_flag)

    class Twice(Eyes):  # the first look at the photo finds the fault, the look after the edit passes
        def __call__(self, message, image):
            if "recoloured" in message:
                self.asked.append(message)
                ok = sum("recoloured" in m for m in self.asked) > 1
                return {"ok": ok, "problems": [] if ok else ["grey strip along the left sleeve"]}
            return super().__call__(message, image)
    edits = []

    def gen(prompt, refs):
        if prompt.startswith("Edit image 1, a clothing catalogue photo"):
            edits.append(prompt)
            with Image.open(refs[0]) as im:
                w, h = im.size
            return _png(Image.new("RGB", (w // 2, h // 2), (255, 0, 0)))  # any size: it is fitted back
        return fake_generate(prompt, refs)
    b = studio.Batch(spec(poses=["front_main"]), out=tmp_path, generate=gen, workers=1, inspector=Twice())
    counts = b.run()
    assert counts["qc_ai_fixed"] == 1 and counts["qc_held"] == 0
    assert "grey strip along the left sleeve" in edits[0] and "the top red (#" in edits[0].lower()
    qc_file = next(b.root.glob("m0001/Red__Red/qc.json"))
    check = json.loads(qc_file.read_text())["front_main"]["check"]
    assert check["passed"] and not check["held"] and any(f.startswith("AI retouch") for f in check["fixed"])
    assert qcagent.passed(qc_file, "front_main")
    px = np.asarray(Image.open(qc_file.parent / "front_main_2000x2000.jpg").convert("RGB"))
    assert px[30:60, 30:60].std() < 3 and px[30:60, 30:60].mean() > 200  # the backdrop corner wasn't touched by the edit


def _png(img):
    import io

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_rejected_bases_come_back_when_no_new_one_can_be_made(tmp_path):
    studio.Batch(spec(qc_agent=False), out=tmp_path, generate=fake_generate, workers=1).run()
    front = studio.Batch(spec(), out=tmp_path).base_path(1, ("light", "light"), "front_main")
    before = front.read_bytes()

    def broke(prompt, refs):
        raise RuntimeError("402 RESOURCE_EXHAUSTED")
    b = studio.Batch(spec(), out=tmp_path, generate=broke, workers=1, inspector=Eyes(fail=1), redo=True, retries=1)
    b.run(stage="qc")
    assert front.read_bytes() == before and (front.parent / "front_main_top_mask.png").exists()
    assert not list(front.parent.glob("*.rejected"))


def test_drive_upload_skips_a_photo_whose_base_went_and_sends_the_rest(tmp_path):
    b = studio.Batch(spec(qc_agent=False), out=tmp_path, generate=fake_generate, workers=1)
    b.run()
    b.base_path(1, ("light", "light"), "back").unlink()
    drive = FakeDrive()
    DriveSync(drive, lambda: str(tmp_path)).run_once()
    assert "Girls/Red/Hoodies/m0001_front.jpg" in drive.files and "Girls/Red/Hoodies/m0001_back.jpg" not in drive.files
    assert "Girls/Red/Co-ords/Hoodies set/m0001_back.jpg" in drive.files
