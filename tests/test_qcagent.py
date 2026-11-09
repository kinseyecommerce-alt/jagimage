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
    photos = [f for f in drive.files if not f.startswith("_bases/")]
    assert photos and all("back" not in f for f in photos)
    assert any(f.startswith("_bases/") for f in drive.files)  # a held photo doesn't keep the bases on the server
    assert (qc_file.parent / "back_2000x2000.jpg").exists()  # held: stays on the server, not uploaded
    assert studio.approve(b.root, ["m0001/Red__Red/back_2000x2000.jpg", "../../etc/passwd.jpg"]) == 1
    drive.files.clear()
    sync.run_once()
    assert sorted(f for f in drive.files if not f.startswith("_bases/")) == [
        "Cotton Mix/Girls/Red/Co-ords/Hoodies set/m0001_back.jpg", "Cotton Mix/Girls/Red/Hoodies/m0001_back.jpg"]


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


def test_flagged_photos_get_the_agents_own_retouch_without_any_ai_edit(tmp_path, monkeypatch):
    from jagimage import recolor

    orig, n = recolor.quality_check, []

    def flag_first_two(*a, **k):  # the first checks count bleed; after the retouch it is gone
        out = orig(*a, **k)
        n.append(1)
        if len(n) <= 2:
            out["problems"], out["passed"] = ["colour bleeds onto skin or background (500 px)"], False
        return out
    monkeypatch.setattr(recolor, "quality_check", flag_first_two)
    calls = []

    def gen(prompt, refs):
        calls.append(prompt)
        return fake_generate(prompt, refs)
    eyes = Eyes()
    b = studio.Batch(spec(poses=["front_main"]), out=tmp_path, generate=gen, workers=1, inspector=eyes)
    counts = b.run()
    assert counts["qc_retouched"] == 1 and counts["qc_held"] == 0
    assert not any(p.startswith("Edit image 1, a clothing") for p in calls)  # no AI edit: the retouch is local
    assert not any("recoloured" in m for m in eyes.asked)  # fixed before the vision check was needed
    qc_file = next(b.root.glob("m0001/Red__Red/qc.json"))
    check = json.loads(qc_file.read_text())["front_main"]["check"]
    assert check["passed"] and not check["held"] and any(f.startswith("retouch") for f in check["fixed"])


def test_local_fix_gives_a_sleeve_lit_by_the_pants_the_sleeves_colour_and_cleans_bleed():
    from scipy import ndimage

    from jagimage.recolor import garment_mask, recolor_image, rgb_to_lab

    mint, lav, grey = np.array([170, 225, 205.]), np.array([205, 180, 235.]), np.array([236, 236, 238.])
    img = np.tile(grey, (240, 200, 1))
    img[20:200, 40:150] = lav
    img[0:120, 50:150] = mint
    img[0:190, 20:70] = mint * 0.9
    xs = np.linspace(0, 1, 30)[None, :, None]
    img[110:190, 40:70] = (mint * 0.85) * (1 - 0.8 * xs) + (lav * 0.8) * 0.8 * xs  # the sleeve, lit purple by the pants
    img = np.pad(img, ((40, 160), (200, 200), (0, 0)), mode="edge")
    img[:, :200] = img[:, -200:] = grey
    img[-160:] = grey
    before = Image.fromarray(np.clip(ndimage.gaussian_filter(img, (0.6, 0.6, 0)), 0, 255).astype(np.uint8))
    lab = rgb_to_lab(np.asarray(before))
    top, _ = garment_mask(lab, "#AAE1CD")
    bottom, _ = garment_mask(lab, "#CDB4EB")  # the colour split: part of the sleeve went to the pants

    def paint(ms):
        o, _, _ = recolor_image(before, "#AAE1CD", "#FFFFFF", mask=ms[0], others=ms[1])
        o, _, _ = recolor_image(o, "#CDB4EB", "#000000", mask=ms[1], others=ms[0])
        return o
    after = np.asarray(paint([top, bottom]).convert("RGB")).copy()
    out, ms = qcagent.local_fix(before, Image.fromarray(after), [top, bottom], ["#FFFFFF", "#000000"], paint)
    px = np.asarray(out).astype(int)
    sleeve = px[152:228, 222:268].reshape(-1, 3)
    assert (sleeve.mean(axis=1) > 150).mean() > 0.97      # the whole sleeve is white, no black or grey on it


def test_local_fix_gives_bled_backdrop_its_own_pixels_back_and_evens_a_wrong_shade():
    grey = (236, 236, 238)
    before = Image.new("RGB", (300, 400), grey)
    before.paste((170, 225, 205), (100, 100, 200, 300))   # the mint garment
    after = before.copy()
    after.paste((200, 30, 30), (100, 100, 200, 300))      # recoloured red...
    after.paste((150, 150, 150), (130, 150, 160, 180))    # ...with a grey patch left in it
    after.paste((150, 40, 40), (204, 150, 208, 220))      # ...and red bled onto the backdrop beside it
    mask = np.zeros((400, 300), np.float32)
    mask[100:300, 100:200] = 1
    out, ms = qcagent.local_fix(before, after, [mask], ["#C81E1E"], lambda ms: after)
    px = np.asarray(out).astype(int)
    assert abs(px[185, 206] - np.array(grey)).max() < 12  # the backdrop is grey again
    r, g, b = px[165, 145]
    assert r > g + 60 and r > b + 60                      # the grey patch is red now, with its own lightness
    assert tuple(px[250, 150]) == (200, 30, 30)           # the rest of the garment is untouched


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
    back = b.base_path(1, ("light", "light"), "back")
    back.unlink()  # the base and its masks went (a base archived to Drive keeps its masks: then the cut still works)
    for m in back.parent.glob("back_*mask.png"):
        m.unlink()
    drive = FakeDrive()
    DriveSync(drive, lambda: str(tmp_path)).run_once()
    assert "Cotton Mix/Girls/Red/Hoodies/m0001_front.jpg" in drive.files and "Cotton Mix/Girls/Red/Hoodies/m0001_back.jpg" not in drive.files
    assert "Cotton Mix/Girls/Red/Co-ords/Hoodies set/m0001_back.jpg" in drive.files


def test_the_shops_bottom_photo_goes_with_every_outfit_shot_and_the_edit_runs_only_when_needed(tmp_path):
    ref = studio.bottom_ref_path(tmp_path, "Trackpants")
    ref.parent.mkdir(parents=True)
    Image.new("RGB", (60, 120), (40, 40, 40)).save(ref)
    seen, edits = [], []

    def gen(prompt, refs):
        if prompt == studio.RETOUCH_BOTTOMS:
            edits.append(refs)
        else:
            seen.append((prompt, refs))
        return fake_generate(prompt, refs)

    class Strings(Eyes):  # the agent sees a drawstring on the first front only
        def __call__(self, message, image):
            if message.count("\n") == 1 and "drawstring" in message:
                self.asked.append(message)
                n = sum(1 for m in self.asked if m.count("\n") == 1)
                return {"ok": n > 1, "problems": [] if n > 1 else ["drawstring at the waist"]}
            return super().__call__(message, image)
    b = studio.Batch(spec(models=2, colour_pairs=[["Red", "Red"]], retouch_bottoms=True, provider="gemini"), out=tmp_path,
                     generate=gen, workers=1, inspector=Strings())
    b.run()
    assert seen and all(str(ref) == r[-1] for _, r in seen)
    assert all(f"Image {len(r)} is a product photo of the exact track pants" in p for p, r in seen)
    assert len(edits) == 1 and b.counts["retouch_skipped"] == 1  # two fronts: one edited, one clean


def test_a_tucked_top_is_looked_for_in_a_waist_close_up(tmp_path):
    shown = []

    class WaistEyes(Eyes):  # the whole photo looks fine; only the close-up shows the waistband over the hem
        def __call__(self, message, image):
            shown.append((message, Image.open(__import__("io").BytesIO(image)).size))
            if "close-up of a child's waist" in message:
                return {"ok": False, "problems": ["the waistband sits over the top's hem"]}
            return {"ok": True, "problems": []}
    calls = []

    def gen(prompt, refs):
        calls.append(prompt)
        return fake_generate(prompt, refs)
    b = studio.Batch(spec(poses=["front_main", "back"]), out=tmp_path, generate=gen, workers=1, inspector=WaistEyes(), retries=2)
    counts = b.run()
    assert counts["base_rejected"] == 2 and counts["errors"] >= 1  # the front is rejected each time it is made
    waist = [size for m, size in shown if "close-up of a child's waist" in m]
    full = [size for m, size in shown if "close-up of a child's waist" not in m]
    assert waist and waist[0][0] < full[0][0]  # the close-up is a crop, not the whole photo


def test_the_kids_bottom_photo_is_not_used_for_adults(tmp_path):
    ref = studio.bottom_ref_path(tmp_path, "Trackpants")
    ref.parent.mkdir(parents=True)
    Image.new("RGB", (60, 120), (40, 40, 40)).save(ref)
    b = studio.Batch(spec(category="Men", top="HalfSleeveTee", colour_pairs=[["Red", "Black"]]), out=tmp_path)
    assert b.bottom_ref() is None
    assert studio.Batch(spec(), out=tmp_path).bottom_ref() == ref


def test_a_lavender_top_over_lilac_based_shorts_is_not_counted_as_lilac_left():
    import numpy as np
    from PIL import Image

    from jagimage.recolor import quality_check

    h, w = 120, 80
    before = np.full((h, w, 3), 200, np.uint8)
    before[60:, :] = (196, 163, 238)  # the shorts, shot in lilac
    top = np.zeros((h, w), np.float32)
    top[:70] = 1
    bottom = np.zeros((h, w), np.float32)
    bottom[60:] = 1  # the top's hem lies over the waistband: rows 60-70 are in both
    after = before.copy()
    after[60:] = (21, 21, 21)

    def run(top_hex, hem):
        a = after.copy()
        a[:70] = hem
        return quality_check(Image.fromarray(before), Image.fromarray(a), [top, bottom], ["#8FD3A8", "#C4A3EE"],
                             [top_hex, "#151515"])
    assert run("#C7B2DE", (199, 178, 222))["base_left_px"] == 0  # lavender: its own colour, not leftover
    # a red top: lilac really left at the waistband still counts
    a = after.copy()
    a[:60] = (207, 11, 21)
    a[60:70] = (196, 163, 238)
    res = quality_check(Image.fromarray(before), Image.fromarray(a), [top, bottom], ["#8FD3A8", "#C4A3EE"],
                        ["#CF0B15", "#151515"])
    assert res["base_left_px"] > 0


def test_the_vision_check_does_not_call_lilac_a_fault_on_a_lavender_or_purple_garment():
    from jagimage.qcagent import photo_points

    assert "lilac" not in photo_points("Lavender", "Black", True, False, ["#C7B2DE", "#151515"])[0]
    assert "lilac" not in photo_points("Purple", "Black", True, False, ["#7F35B2", "#151515"])[0]
    red = photo_points("Red", "Black", True, False, ["#CF0B15", "#151515"])[0]
    assert "lilac" in red and "mint green" in red
    assert "mint green" not in photo_points("Green", "Black", True, False, ["#2E7D32", "#151515"])[0]


def test_the_held_stage_makes_only_held_photos_again(tmp_path):
    b = studio.Batch(spec(qc_agent=False), out=tmp_path, generate=fake_generate, workers=1)
    b.run()
    qc_file = next(b.root.glob("m0001/Red__Red/qc.json"))
    rows = json.loads(qc_file.read_text())
    rows["back"]["check"].update(passed=False, held=True, problems=["base colour left"])
    qc_file.write_text(json.dumps(rows))
    again = studio.Batch(spec(qc_agent=False), out=tmp_path, generate=fake_generate, workers=1, redo="held")
    front, back = (again.out_path(1, "Red", "Red", p) for p in ("front_main", "back"))
    assert again._done(front, "front_main") and not again._done(back, "back")
    counts = again.run(stage="recolor")
    assert counts["recoloured"] == 1 and counts["ai_done"] == 0
    assert not json.loads(qc_file.read_text())["back"]["check"].get("held")


def test_local_fix_paints_a_thin_strip_of_base_colour_along_a_seam():
    import numpy as np
    from PIL import Image

    from jagimage.qcagent import local_fix
    from jagimage.recolor import hex_to_lab, lab_to_rgb

    h, w = 60, 60
    lav = lab_to_rgb(hex_to_lab("#C7B2DE")[None, None, :]).reshape(3)
    before = np.full((h, w, 3), 220, np.uint8)
    after = before.copy()
    after[10:50, 15:45] = lav
    after[12:48, 29:31] = (143, 211, 168)  # a 2 px mint line down a side seam
    m = np.zeros((h, w), np.float32)
    m[10:50, 15:45] = 1
    out, _ = local_fix(Image.fromarray(before), Image.fromarray(after), [m], ["#C7B2DE"], lambda ms: Image.fromarray(after))
    o = np.asarray(out).astype(int)
    assert abs(o[30, 29] - lav.astype(int)).max() < 25  # the seam line took the lavender


def test_base_qc_checks_the_cut_and_that_nothing_else_is_in_the_photo():
    # the AI made half sleeves for a full-sleeve batch (copying a reference), and pasted the shorts' product photo
    full = " ".join(qcagent.base_points(spec(top="FullSleeveTee", category="Kids_Boys"), "front_main"))
    assert "reach the wrists" in full and "FAIL for short or half sleeves" in full
    assert "product photo beside" in full
    sl = " ".join(qcagent.base_points(spec(top="SleevelessTee", category="Women"), "back"))
    assert "FAIL for any sleeves" in sl
    for top in studio.TOPS:
        assert top in qcagent.STYLE_POINTS
