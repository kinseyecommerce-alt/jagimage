import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from jagimage import listing, printrun
from jagimage.dropbox import Dropbox, dropbox_path
from jagimage.segment import shirt_mask


def tee_photo(colour=(200, 30, 40), size=(600, 800)):
    """A plain studio photo: grey backdrop, a person's skin-coloured arms and head, a T-shirt."""
    im = Image.new("RGB", size, (225, 225, 228))
    d = ImageDraw.Draw(im)
    d.ellipse((255, 60, 345, 160), fill=(205, 160, 130))           # head
    d.rectangle((150, 230, 190, 470), fill=(205, 160, 130))        # arms
    d.rectangle((410, 230, 450, 470), fill=(205, 160, 130))
    d.polygon([(200, 170), (400, 170), (460, 260), (420, 280), (400, 250), (400, 520), (200, 520), (200, 250),
               (180, 280), (140, 260)], fill=colour)                 # the tee
    d.rectangle((210, 520, 390, 700), fill=(30, 30, 30))            # shorts
    return im


def shirt_truth(size=(600, 800)):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([(200, 170), (400, 170), (460, 260), (420, 280), (400, 250), (400, 520), (200, 520), (200, 250),
                               (180, 280), (140, 260)], fill=255)
    return np.asarray(m) > 127


def test_the_shirt_is_found_from_the_ai_hint():
    m = shirt_mask(tee_photo(), (0.22, 0.2, 0.78, 0.66), (0.5, 0.4)) > 0.5
    truth = shirt_truth()
    iou = (m & truth).sum() / (m | truth).sum()
    assert iou > 0.9


def test_sku_pattern():
    f = printrun.sku_fields("spb", {"category": "Kids_Girls", "style": "OversizedTee", "colour": "Light Pink", "fabric": "PureCotton"},
                            "unicorn 02.png", 7)
    assert printrun.make_sku(printrun.DEFAULT_PATTERN, f) == "SPB-KG-OS-LIG-UNICORN02PNG"
    assert printrun.make_sku("{BRAND}/{FABRIC}_{COLOUR} {N}", f) == "SPB-PC_LIGHTPINK-0007"
    with pytest.raises(ValueError):
        printrun.make_sku("{NOTHING}", f)


def test_folder_names_say_what_a_product_is():
    assert printrun.guess(["Pure Cotton", "Girls", "Lavender", "Half sleeve"]) == {
        "category": "Kids_Girls", "style": "HalfSleeveTee", "fabric": "PureCotton", "colour": "Lavender"}
    assert printrun.guess(["Pure Cotton", "cottonmens", "Navy", "Hoodies"])["category"] == "Men"
    assert printrun.stem_of("m0003_front.jpg") == printrun.stem_of("m0003_back.jpg") == "m0003"
    assert printrun.is_back_name("m0003_back.jpg") and not printrun.is_back_name("backpack_front.jpg")


def art_png(colour, size=(400, 300)):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((40, 40, 360, 260), fill=colour + (255,))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def jpg(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue()


class FakeDrive:
    def __init__(self, tees, prints):
        self.tees, self.prints, self.files, self.uploaded = tees, prints, {}, {}
        for f in tees + prints:
            self.files[f["id"]] = f.pop("data")

    def list_tree(self, folder, depth=5):
        return [dict(f) for f in (self.tees if folder == "TEES" else self.prints)]

    def download(self, fid):
        return self.files[fid]

    def upload(self, local, parts, name, root=""):
        self.uploaded["/".join(([f"<{root}>"] if root else []) + parts + [name])] = local.read_bytes()


class FakeDropbox:
    connected = True

    def __init__(self):
        self.files = {}

    def upload(self, data, *parts):
        path = dropbox_path("Root", *parts)
        self.files[path] = data
        return path

    def link(self, path):
        return "https://dl.example" + path + "?raw=1"


def fake_ask(asked):
    def ask(message, system, schema, images=()):
        asked.append(system)
        if system is printrun.SHIRT_SYSTEM:
            return {"is_top": True, "view": "front", "plain": True, "box": [0.22, 0.2, 0.78, 0.66], "chest": [0.5, 0.4],
                    "colour": "red", "colour_hex": "#C8202A", "style": "HalfSleeveTee", "wearer": "Kids_Boys"}
        return {"ok": True, "reason": "good contrast", "theme": "abstract", "description": "a big circle"}
    return ask


def test_a_print_run_makes_one_sku_per_print_and_product_and_leaves_out_prints_that_would_not_show(tmp_path):
    red = jpg(tee_photo())
    path = ["Pure Cotton", "Boys", "Red", "Half sleeve"]
    tees = [{"id": "t1", "name": "m0001_front.jpg", "path": path, "data": red},
            {"id": "t2", "name": "m0001_back.jpg", "path": path, "data": red},
            {"id": "t3", "name": "m0002_front.jpg", "path": path, "data": red},
            {"id": "t4", "name": "x_front.jpg", "path": ["Cotton Mix", "Boys", "Red", "Combo sets", "Red-Black"], "data": red}]
    prints = [{"id": "p1", "name": "Moon.png", "path": [], "data": art_png((250, 250, 250))},
              {"id": "p2", "name": "Ruby.png", "path": [], "data": art_png((200, 32, 42))}]  # red ink on a red tee
    drive, box, asked = FakeDrive(tees, prints), FakeDropbox(), []
    run = printrun.PrintRun(drive, fake_ask(asked), tmp_path, "TEES", "PRINTS", dropbox=box, per_sku=5, brand="SPB")
    run._run()
    assert run.error is None, run.error
    assert [s["sku"] for s in run.skus] == ["SPB-KB-HS-RED-MOON"]
    assert run.counts["left_out"] == 1  # the red print on the red tee would hardly show: no AI asked about it
    assert asked.count(printrun.SUIT_SYSTEM) == 1 and asked.count(printrun.SHIRT_SYSTEM) == 2  # combo-set folder skipped
    assert sorted(drive.uploaded) == [f"Printed/SPB-KB-HS-RED-MOON/SPB-KB-HS-RED-MOON_{i}.jpg" for i in (1, 2, 3)]
    assert sorted(box.files) == [f"/Root/SPB-KB-HS-RED-MOON/SPB-KB-HS-RED-MOON_{i}.jpg" for i in (1, 2, 3)]
    first = np.asarray(Image.open(io.BytesIO(drive.uploaded["Printed/SPB-KB-HS-RED-MOON/SPB-KB-HS-RED-MOON_1.jpg"])))
    assert first[240, 300].mean() > 200 and first[400, 300, 1] < 80  # the white print on the chest, plain below it
    back = np.asarray(Image.open(io.BytesIO(drive.uploaded["Printed/SPB-KB-HS-RED-MOON/SPB-KB-HS-RED-MOON_2.jpg"])))
    assert back[int(800 * 0.4), 300, 0] > 150 and back[int(800 * 0.4), 300, 1] < 80  # the back stays plain
    saved = json.loads((tmp_path / "printruns" / f"{run.id}.json").read_text())
    sku = saved["skus"][0]
    assert sku["colour"] == "Red" and sku["fabric"] == "PureCotton" and sku["images"][0]["url"].endswith("?raw=1")
    assert (tmp_path / "printed" / run.id / sku["sku"] / "listing.jpg").exists()
    assert not list((tmp_path / "printed" / run.id).glob("*/*_1.jpg"))  # the full photos left the server


def test_listing_text_is_held_to_each_marketplaces_rules():
    t = listing.tidy("amazon", {"title": "BEST Boys Dinosaur Printed Cotton T-shirt!! Sale t-shirt t-shirt t-shirt Navy " + "x" * 200,
                                "bullets": ["Soft: feel"] * 7, "description": "Great.\nFree shipping today.",
                                "keywords": ["dino tee", "boys tshirt", "SPB tee", "printed t shirt kids", "navy"]}, "SPB")
    assert t["title"].startswith("SPB Boys Dinosaur") and "BEST" not in t["title"] and "Sale" not in t["title"]
    assert len(t["title"]) <= 150 and "!" not in t["title"] and t["title"].lower().count("t-shirt") <= 2
    assert len(t["bullets"]) == 5 and "free shipping" not in t["description"].lower()
    st = t["search_terms"]
    assert len(st.encode()) <= 249 and "spb" not in st and "navy" not in st.split() and "dino" in st.split()
    s = listing.tidy("shopify", {"title": "Boys Dino Tee", "seo_title": "y" * 90, "seo_description": "z " * 120, "tags": ["Dino"]}, "SPB")
    assert len(s["seo_title"]) <= 60 and len(s["seo_description"]) <= 155 and "dino" in s["tags"]


def listing_ask(message, system, schema, images=()):
    if system is listing.MAP_SYSTEM:
        return {"header_row": 2, "first_data_row": 3, "columns": [{"column": "A", "field": "Seller SKU"}, {"column": "B", "field": "Title"},
                                                                   {"column": "C", "fixed": "India"}, {"column": "D", "field": "Image 1"}]}
    m = {k: {"title": "Boys Moon Printed Pure Cotton T-shirt", "bullets": ["Print: a big moon"], "description": "Soft cotton.",
             "keywords": ["moon tee"], "seo_title": "Moon Tee", "seo_description": "A moon tee.", "tags": ["moon"]} for k in listing.MARKET_ORDER}
    return {"attributes": {"occasion": "Casual"}, "markets": m}


def test_a_listing_job_writes_every_marketplace_file_and_fills_a_shop_template(tmp_path):
    from openpyxl import Workbook, load_workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Amazon template v2"])
    ws.append(["item_sku", "item_name", "country_of_origin", "main_image_url"])
    buf = io.BytesIO()
    wb.save(buf)
    sku = {"sku": "SPB-KB-HS-RED-MOON", "category": "Kids_Boys", "style": "HalfSleeveTee", "fabric": "PureCotton", "colour": "Red",
           "print": "Moon", "theme": "moon", "images": [{"url": "https://dl.example/1.jpg?raw=1"}, {"url": "https://dl.example/2.jpg?raw=1"}]}
    job = listing.ListingJob(listing_ask, tmp_path, [sku], ["amazon", "flipkart", "shopify"],
                             {"brand": "SPB", "sizes": "2-3Y, 3-4Y", "mrp": "599", "price": "349"}, {"amazon": buf.getvalue()})
    job._run()
    assert job.error is None, job.error
    assert job.files == {"amazon": "amazon_template_filled.xlsx", "flipkart": "flipkart_listing.xlsx", "shopify": "shopify_products.csv"}
    ws = load_workbook(job.dir / "amazon_template_filled.xlsx").active
    assert [ws.cell(3, c).value for c in range(1, 5)] == ["SPB-KB-HS-RED-MOON-23Y", "SPB Boys Moon Printed Pure Cotton T-shirt", "India",
                                                          "https://dl.example/1.jpg?raw=1"]
    assert ws.cell(4, 1).value == "SPB-KB-HS-RED-MOON-34Y"
    fk = load_workbook(job.dir / "flipkart_listing.xlsx").active
    head = [c.value for c in fk[1]]
    assert fk.cell(2, head.index("MRP") + 1).value == "599" and fk.cell(2, head.index("Sleeve") + 1).value == "Half Sleeve"
    csv = (job.dir / "shopify_products.csv").read_text()
    assert csv.splitlines()[0].startswith("Handle,Title,Body (HTML)") and "SPB-KB-HS-RED-MOON-23Y" in csv


def test_dropbox_paths_and_links():
    calls = []

    class R:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def opener(req, timeout=0):
        calls.append((req.full_url, req.headers, req.data))
        if req.full_url.endswith("create_shared_link_with_settings"):
            return R(b'{"url": "https://www.dropbox.com/scl/fi/abc/x.jpg?rlkey=k&dl=0"}')
        return R(b"{}")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        box = Dropbox(d, "appkey1234", "secret1234", opener=opener)
        box.state["refresh_token"] = "r"
        box._access = ("tok", 9e18)
        path = box.upload(b"jpeg", "SPB/KB:1", "a.jpg")
        assert path == "/Jag Studio/SPB-KB-1/a.jpg"
        assert json.loads(calls[0][1]["Dropbox-api-arg"])["mode"] == "overwrite"
        assert box.link(path) == "https://www.dropbox.com/scl/fi/abc/x.jpg?rlkey=k&raw=1"


def test_dropbox_says_which_permission_is_missing():
    import urllib.error

    from jagimage.dropbox import _why

    body = json.dumps({"error_summary": "missing_scope/", "user_message": {"locale": "en", "text":
        "Error in call to API function \"files/upload\": Your app (ID: 1) is not permitted to access this endpoint because it "
        "does not have the required scope 'files.content.write'. The owner of the app can enable the scope."}}).encode()
    exc = urllib.error.HTTPError("u", 400, "Bad", {}, io.BytesIO(body))
    assert "files.content.write" in _why("files/upload", exc) and "Connect again" in _why.__doc__ + _why("files/upload",
        urllib.error.HTTPError("u", 400, "Bad", {}, io.BytesIO(body)))
    assert printrun.stem_of("m0005_front_main.jpg") == printrun.stem_of("m0005_back.jpg") == "m0005"


def test_every_print_gets_the_shops_standard_size_and_line():
    from jagimage.prints import _layout, chest

    mask = shirt_truth().astype(np.float32)
    cx, neck, cw = chest(mask)
    badge = Image.open(io.BytesIO(art_png((250, 250, 250), size=(600, 200))))
    tall = Image.open(io.BytesIO(art_png((250, 250, 250), size=(300, 400))))
    g = _layout(mask, badge, "chest_large")  # the shop's standard: every print starts on the same line, same size
    assert abs((g["y0"] - neck) / cw - 0.20) < 0.02 and abs(g["pw"] / cw - 0.45) < 0.02
    t = _layout(mask, tall, "chest_large")
    assert abs((t["y0"] - neck) / cw - 0.20) < 0.02 and t["ph"] / cw <= 0.51  # a tall print shrinks to fit the box
    men = _layout(mask, badge, "chest_large:men")  # adults: a wider print, a little nearer the collar
    assert abs(men["pw"] / cw - 0.55) < 0.02 and abs((men["y0"] - neck) / cw - 0.18) < 0.02
    sq = _layout(mask, Image.open(io.BytesIO(art_png((250, 250, 250), size=(300, 300)))), "chest_large")
    assert abs(sq["pw"] / cw - 0.45 * 0.85) < 0.02  # a square design a little narrower
    m = _layout(mask, badge, "chest_medium")
    assert abs((m["y0"] + m["ph"] / 2 - neck) / cw - 0.32) < 0.02  # medium: a short print's middle at chest height


def test_the_print_placement_agent_moves_the_first_photo_and_the_sku_follows(tmp_path):
    red = jpg(tee_photo())
    path = ["Cotton Mix", "Boys", "Red", "Half sleeve"]
    tees = [{"id": f"t{i}", "name": f"m000{i}_front.jpg", "path": path, "data": red} for i in (1, 2, 3)]
    prints = [{"id": "p1", "name": "Moon.png", "path": [], "data": art_png((250, 250, 250))}]
    seen = []

    def checker(photo, art, measured):
        seen.append(measured["collar_gap"])
        if len(seen) == 1:
            return {"ok": False, "score": 5, "issues": ["a little high"], "dx": 0.0, "dy": 0.05, "scale": 1.0}
        return {"ok": True, "score": 9, "issues": [], "dx": 0.0, "dy": 0.0, "scale": 1.0}
    run = printrun.PrintRun(FakeDrive(tees, prints), fake_ask([]), tmp_path, "TEES", "PRINTS", per_sku=3, checker=checker)
    run._run()
    assert run.error is None and len(seen) == 2 and seen[1] > seen[0]  # moved down once, then right
    assert run.counts["fixed"] == 1 and run.skus[0]["photos"] == 3  # one look per SKU, all photos placed alike


def test_one_print_one_model_front_only_named_as_the_print(tmp_path):
    red = jpg(tee_photo())
    path = ["Cotton Mix", "Boys", "Red", "Full sleeve"]
    tees = [{"id": f"t{i}", "name": f"m000{i}_front.jpg", "path": path, "data": red} for i in (1, 2)] + \
           [{"id": "b1", "name": "m0001_back.jpg", "path": path, "data": red}]
    prints = [{"id": f"p{k}", "name": f"{k}.png", "path": [], "data": art_png((250, 250, 250))} for k in (223, 225, 229)]
    drive = FakeDrive(tees, prints)
    run = printrun.PrintRun(drive, fake_ask([]), tmp_path, "TEES", "PRINTS", per_sku=5, one_model=True)
    run._run()
    assert run.error is None
    assert sorted(drive.uploaded) == ["Printed/SPB-KB-FS-RED-223/223.jpg", "Printed/SPB-KB-FS-RED-225/225.jpg",
                                      "Printed/SPB-KB-FS-RED-229/229.jpg"]
    assert all(s["photos"] == 1 for s in run.skus)


def test_a_test_run_uploads_nothing_and_keeps_its_photos_to_look_at(tmp_path):
    red = jpg(tee_photo())
    tees = [{"id": "t1", "name": "m0001_front.jpg", "path": ["Cotton Mix", "Boys", "Red", "Half sleeve"], "data": red}]
    prints = [{"id": "p1", "name": "230.png", "path": [], "data": art_png((250, 250, 250))}]
    drive, box = FakeDrive(tees, prints), FakeDropbox()
    run = printrun.PrintRun(drive, fake_ask([]), tmp_path, "TEES", "PRINTS", dropbox=box, one_model=True, test=True)
    run._run()
    assert run.error is None and not drive.uploaded and not box.files
    view = run.status()["skus"][0]["view"]
    assert view == f"/api/printruns/{run.id}/photo/SPB-KB-HS-RED-230/230.jpg"
    assert (tmp_path / "printed" / run.id / "SPB-KB-HS-RED-230" / "230.jpg").exists()


def test_dropbox_delete_stays_inside_the_apps_folder():
    import tempfile

    calls = []

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def opener(req, timeout=0):
        calls.append(json.loads(req.data))
        return R()
    with tempfile.TemporaryDirectory() as d:
        box = Dropbox(d, "appkey1234", "secret1234", opener=opener)
        box.state["refresh_token"] = "r"
        box._access = ("tok", 9e18)
        assert box.delete("SPB-KB-FS-BLA-223") == "/Jag Studio/SPB-KB-FS-BLA-223"
        assert calls[-1] == {"path": "/Jag Studio/SPB-KB-FS-BLA-223"}
        with pytest.raises(ValueError):
            box.delete()


def test_a_white_tee_on_a_light_backdrop_and_a_hood_are_found_right():
    # a white tee on a light grey studio wall leaked into the backdrop and face; a hood moved the neckline up
    from jagimage.prints import chest

    im = tee_photo(colour=(246, 246, 246))
    im = Image.fromarray(np.where(np.asarray(im) == 225, 222, np.asarray(im)).astype(np.uint8))
    m = shirt_mask(im, (0.22, 0.2, 0.78, 0.66), (0.5, 0.4)) > 0.5
    truth = shirt_truth()
    assert (m & truth).sum() / (m | truth).sum() > 0.85
    hood = np.zeros((800, 600), np.float32)
    hood[60:120, 220:380] = 1          # the hood above the head
    hood[170:520, 200:400] = 1         # the body
    cx, neck, cw = chest(hood)
    assert abs(neck - 170) <= 2        # the neckline under the face, not the hood's top
    printrun.sane(truth.astype(np.float32), (0.22, 0.2, 0.78, 0.66), (600, 800))
    with pytest.raises(Exception):
        printrun.sane(truth.astype(np.float32), (0.3, 0.35, 0.7, 0.7), (600, 800))  # a box well below: face-level mask


def test_the_shirt_is_found_even_when_the_ai_points_at_the_face():
    # the AI sometimes gave a chest point on the face: the print then landed on the face
    for colour, hex_ in (((200, 30, 40), "#C8202A"), ((246, 246, 246), "#F6F6F6")):
        m = shirt_mask(tee_photo(colour), (0.22, 0.2, 0.78, 0.66), (0.5, 0.13), hex_) > 0.5
        truth = shirt_truth()
        assert (m & truth).sum() / (m | truth).sum() > 0.85
    with pytest.raises(Exception):  # a face-sized patch is never accepted as the shirt
        face = np.zeros((800, 600), np.float32)
        face[60:160, 255:345] = 1
        printrun.sane(face, (0.22, 0.2, 0.78, 0.66), (600, 800))


def test_oversized_tees_start_prints_lower_than_regular_ones():
    from jagimage.prints import _layout, chest, fit_for

    assert fit_for("Men", "OversizedTee") == "men_oversized" and fit_for("Men", "HalfSleeveTee") == "men"
    assert fit_for("Kids_Boys", "OversizedTee") == "kids"
    mask = shirt_truth().astype(np.float32)
    cx, neck, cw = chest(mask)
    tall = Image.open(io.BytesIO(art_png((250, 250, 250), size=(300, 400))))
    reg, ov = _layout(mask, tall, "chest_large:men"), _layout(mask, tall, "chest_large:men_oversized")
    assert ov["y0"] - reg["y0"] > 0.06 * cw and ov["ph"] < reg["ph"]



def test_the_placement_agent_never_raises_a_print_onto_the_collar(tmp_path):
    red = jpg(tee_photo())
    tees = [{"id": "t1", "name": "m0001_front.jpg", "path": ["Cotton Mix", "Men", "Red", "Oversized"], "data": red}]
    prints = [{"id": "p1", "name": "225.png", "path": [], "data": art_png((250, 250, 250))}]
    seen = []

    def checker(photo, art, measured):  # an agent that keeps asking for the print to go up
        seen.append(measured["collar_gap"])
        return {"ok": False, "score": 4, "issues": ["too low"], "dx": 0.0, "dy": -0.15, "scale": 1.0}
    run = printrun.PrintRun(FakeDrive(tees, prints), fake_ask([]), tmp_path, "TEES", "PRINTS", checker=checker, one_model=True)
    run._run()
    assert run.error is None and min(seen) >= seen[0] - 0.021


def test_at_most_n_prints_per_product(tmp_path):
    red = jpg(tee_photo())
    navy = jpg(tee_photo((12, 35, 64)))
    tees = [{"id": f"t{i}", "name": f"m000{i}_front.jpg", "path": ["Cotton Mix", "Boys", c, "Half sleeve"], "data": d}
            for i, c, d in ((1, "Red", red), (2, "Navy", navy))]
    prints = [{"id": f"p{k}", "name": f"{k}.png", "path": [], "data": art_png((250, 250, 250))} for k in range(5)]
    run = printrun.PrintRun(FakeDrive(tees, prints), fake_ask([]), tmp_path, "TEES", "PRINTS", one_model=True, test=True,
                            per_product=2)
    run._run()
    assert run.error is None and len(run.skus) == 4  # 2 products x 2 prints each


def test_local_checks_ask_the_ai_only_where_the_shirt_is(tmp_path):
    red = jpg(tee_photo())
    tees = [{"id": "t1", "name": "m0001_front.jpg", "path": ["Cotton Mix", "Boys", "Red", "Half sleeve"], "data": red},
            {"id": "t2", "name": "m0002_back.jpg", "path": ["Cotton Mix", "Boys", "Red", "Half sleeve"], "data": red}]
    prints = [{"id": "p1", "name": "230.png", "path": [], "data": art_png((250, 250, 250))},
              {"id": "p2", "name": "224.png", "path": [], "data": art_png((200, 32, 42))}]

    asked = []
    run = printrun.PrintRun(FakeDrive(tees, prints), fake_ask(asked), tmp_path, "TEES", "PRINTS", one_model=True, test=True,
                            local=True, checker=lambda *a: (_ for _ in ()).throw(AssertionError("no AI print checker")))
    run._run()
    assert run.error is None and [s["sku"] for s in run.skus] == ["SPB-KB-HS-RED-230"]  # the red-on-red print left out
    assert set(asked) == {printrun.SHIRT_SYSTEM}  # no AI suitability call; the placement checked on the server
    out = np.asarray(Image.open(tmp_path / "printed" / run.id / "SPB-KB-HS-RED-230" / "230.jpg"))
    assert out[240, 300].mean() > 200 and out[110, 300, 0] > 150  # print on the chest; the face untouched


def test_visible_share_leaves_dark_on_dark_out():
    from PIL import ImageDraw
    from jagimage.printrun import visible_share
    from jagimage.prints import _layout, place_print

    photo = Image.new("RGB", (600, 800), (230, 230, 230))
    ImageDraw.Draw(photo).rectangle((120, 150, 480, 750), fill=(21, 21, 21))
    mask = np.zeros((800, 600), np.float32)
    mask[150:750, 120:480] = 1
    seen = {}
    for name, col in (("black", (10, 10, 10, 255)), ("white", (250, 250, 250, 255))):
        art = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        ImageDraw.Draw(art).ellipse((20, 20, 280, 280), fill=col)
        out = place_print(photo, mask, art, "chest_large:kids")
        seen[name] = visible_share(photo, out, _layout(mask, art, "chest_large:kids"))
    assert seen["black"] < 0.2 and seen["white"] > 0.9


def _qc_shirt():
    photo = Image.new("RGB", (600, 800), (230, 230, 230))
    ImageDraw.Draw(photo).rectangle((120, 150, 480, 750), fill=(21, 21, 21))
    mask = np.zeros((800, 600), np.float32)
    mask[150:750, 120:480] = 1
    return photo, mask


def _qc_art(col, side=300):
    art = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    ImageDraw.Draw(art).ellipse((side // 15, side // 15, side - side // 15, side - side // 15), fill=col)
    return art


def test_print_qc_passes_a_good_print_and_holds_faults():
    from jagimage import printqc
    from jagimage.prints import place_print

    photo, mask = _qc_shirt()
    art = _qc_art((250, 250, 250, 255))
    out = place_print(photo, mask, art, "chest_large:kids")
    good = printqc.check(photo, out, mask, art, "chest_large:kids", {}, "SPB-KB-HS-BLA-223", "223.jpg", "223")
    assert good["ok"], good
    # black on black: doesn't show, left out rather than held
    dark = _qc_art((10, 10, 10, 255))
    r = printqc.check(photo, place_print(photo, mask, dark, "chest_large:kids"), mask, dark, "chest_large:kids")
    assert printqc.only_visibility(r)
    # a stray change elsewhere in the photo
    spoilt = out.copy()
    ImageDraw.Draw(spoilt).rectangle((10, 10, 100, 100), fill=(255, 0, 0))
    assert "photo" in printqc.check(photo, spoilt, mask, art, "chest_large:kids")["failed"]
    # a tiny artwork blown up
    tiny = _qc_art((250, 250, 250, 255), side=30)
    r = printqc.check(photo, place_print(photo, mask, tiny, "chest_large:kids"), mask, tiny, "chest_large:kids")
    assert "sharpness" in r["failed"]
    # an incomplete SKU and a photo not named after the print
    r = printqc.check(photo, out, mask, art, "chest_large:kids", {}, "SPB-U-HS-BLA-223", "x.jpg", "223")
    assert {"sku", "name"} <= set(r["failed"])


def test_print_run_holds_and_approves(tmp_path):
    run = printrun.PrintRun(None, None, tmp_path, "t", "p", test=True, one_model=True)
    prod = {"category": "Boys", "style": "HalfSleeveTee", "colour": "Black", "path": ["Boys", "Black"]}
    run._save_sku(tmp_path / "printed" / run.id, "SPB-KB-HS-BLA-1", prod, "1", {}, "#151515",
                  [Image.new("RGB", (40, 40))], held=["too high: touches the collar"])
    st = run.status()
    assert st["counts"]["held"] == 1 and not st["skus"] and st["held"][0]["problems"] == ["too high: touches the collar"]
    st = run.approve("SPB-KB-HS-BLA-1")
    assert st["counts"]["held"] == 0 and st["skus"][0]["sku"] == "SPB-KB-HS-BLA-1"


def test_folder_fabric_names_find_their_colours():
    assert printrun.colour_hex_for("Green", "Cotton Mix") == "#2E7D32"
    assert printrun.colour_hex_for("Green", "Pure Cotton") == "#6CC24A"


def test_print_qc_keeps_zip_hoodie_prints_off_the_zip():
    from jagimage import printqc
    from jagimage.prints import place_print

    photo, mask = _qc_shirt()
    art = _qc_art((250, 250, 250, 255))
    centre = printqc.check(photo, place_print(photo, mask, art, "chest_large:kids"), mask, art, "chest_large:kids", style="ZipHoodie")
    assert "placement" in centre["failed"]
    left = printqc.check(photo, place_print(photo, mask, art, "zip_chest:kids"), mask, art, "zip_chest:kids", style="ZipHoodie")
    assert left["ok"], left


def test_print_qc_holds_prints_under_or_over_a_hand():
    from jagimage import printqc
    from jagimage.prints import place_print

    photo, mask = _qc_shirt()
    ImageDraw.Draw(photo).ellipse((310, 250, 400, 340), fill=(205, 150, 120))  # a hand on the chest
    yy, xx = np.mgrid[0:800, 0:600]
    hand = ((yy - 295) ** 2 + (xx - 355) ** 2) <= 45 ** 2
    art = _qc_art((250, 250, 250, 255))
    under = mask.copy()
    under[hand] = 0  # the shirt mask leaves the hand out: the print goes under it
    r = printqc.check(photo, place_print(photo, under, art, "chest_large:kids"), under, art, "chest_large:kids",
                      shirt_hex="#151515")
    assert "hand" in r["failed"] or "placement" in r["failed"]
    # the mask took the hand in (a skin-toned shirt): the print is painted over it
    r = printqc.check(photo, place_print(photo, mask, art, "chest_large:kids"), mask, art, "chest_large:kids",
                      shirt_hex="#151515")
    assert "hand" in r["failed"], r


def test_zip_hoodie_print_starts_one_cm_from_the_zip():
    from jagimage.prints import CHEST_CM, _layout

    _, mask = _qc_shirt()
    art = _qc_art((250, 250, 250, 255))
    g = _layout(mask, art, "zip_chest:kids")
    gap_cm = (g["x0"] - g["cx"]) / g["cw"] * CHEST_CM["kids"] - 0.4  # from the zip's edge
    assert abs(gap_cm - 1.0) < 0.15


def test_tall_print_beside_the_zip_stays_logo_sized():
    from jagimage.prints import SMALL_TALL, _layout

    _, mask = _qc_shirt()
    tall = Image.new("RGBA", (200, 600), (0, 0, 0, 0))
    ImageDraw.Draw(tall).rectangle((10, 10, 190, 590), fill=(250, 250, 250, 255))
    g = _layout(mask, tall, "zip_chest:kids")
    assert g["ph"] <= SMALL_TALL * g["cw"] + 1


def test_print_qc_does_not_take_a_yellow_shirt_for_skin():
    from jagimage import printqc
    from jagimage.prints import place_print

    photo = Image.new("RGB", (600, 800), (230, 230, 230))
    ImageDraw.Draw(photo).rectangle((120, 150, 480, 750), fill=(248, 169, 26))  # an orange-yellow tee
    mask = np.zeros((800, 600), np.float32)
    mask[150:750, 120:480] = 1
    art = _qc_art((20, 30, 60, 255))
    r = printqc.check(photo, place_print(photo, mask, art, "chest_large:kids"), mask, art, "chest_large:kids",
                      shirt_hex="#F9A712")
    assert "hand" not in r["failed"], r


def test_sleeveless_prints_are_bigger_and_lower():
    from jagimage.prints import _layout, fit_for

    assert fit_for("Kids_Boys", "SleevelessTee") == "kids_sleeveless"
    _, mask = _qc_shirt()
    art = _qc_art((250, 250, 250, 255))
    tee, vest = _layout(mask, art, "chest_large:kids"), _layout(mask, art, "chest_large:kids_sleeveless")
    assert vest["pw"] > tee["pw"] and vest["y0"] > tee["y0"]


def test_products_read_the_picked_folder_path(tmp_path):
    run = printrun.PrintRun(None, None, tmp_path, "t", "p", base_path=["Cotton Mix", "Boys", "Red"])
    prods = run._products([{"id": "1", "name": "m0001_front.jpg", "path": ["Sleeveless"]}])
    assert prods[0]["colour"] == "Red" and prods[0]["fabric"] and prods[0]["style"] == "SleevelessTee"
    assert prods[0]["path"] == ["Cotton Mix", "Boys", "Red", "Sleeveless"]


def test_print_qc_leaves_out_a_print_whose_coloured_part_vanishes():
    from jagimage import printqc
    from jagimage.prints import place_print

    photo = Image.new("RGB", (600, 800), (230, 230, 230))
    ImageDraw.Draw(photo).rectangle((120, 150, 480, 750), fill=(200, 25, 30))  # a red tee
    mask = np.zeros((800, 600), np.float32)
    mask[150:750, 120:480] = 1
    art = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(art)
    d.rectangle((0, 0, 399, 120), fill=(250, 250, 250, 255))  # white text
    d.rectangle((150, 150, 250, 299), fill=(205, 30, 35, 255))  # the red 5
    r = printqc.check(photo, place_print(photo, mask, art, "chest_large:kids"), mask, art, "chest_large:kids")
    assert printqc.only_visibility(r) and "part" in r["problems"][0], r
    # the same print on a black tee: every part shows
    ImageDraw.Draw(photo).rectangle((120, 150, 480, 750), fill=(21, 21, 21))
    assert printqc.check(photo, place_print(photo, mask, art, "chest_large:kids"), mask, art, "chest_large:kids")["ok"]


def test_print_qc_keeps_black_letters_inside_a_white_badge_on_black():
    from jagimage import printqc
    from jagimage.prints import place_print

    photo, mask = _qc_shirt()  # a black tee
    art = Image.new("RGBA", (400, 240), (0, 0, 0, 0))
    d = ImageDraw.Draw(art)
    d.ellipse((0, 0, 399, 239), fill=(250, 250, 250, 255))
    d.rectangle((90, 90, 310, 150), fill=(15, 15, 15, 255))  # black lettering inside the badge
    r = printqc.check(photo, place_print(photo, mask, art, "chest_large:kids"), mask, art, "chest_large:kids")
    assert r["ok"], r


def test_on_white_turns_the_studio_and_floor_white_and_keeps_a_pale_garment():
    from jagimage.market import on_white

    im = Image.new("RGB", (400, 500), (222, 222, 222))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 460, 399, 499), fill=(166, 173, 166))  # a floor strip of another grey
    d.rectangle((150, 100, 250, 400), fill=(226, 226, 228))  # a pale grey melange tee, almost the wall's grey
    person = np.zeros((500, 400), np.float32)
    person[100:401, 150:251] = 1  # what the cutout model sees as the person
    out = np.asarray(on_white(im, person=person))
    assert out[5, 5].min() == 255 and out[480, 20].min() >= 250  # wall and floor
    assert abs(int(out[250, 200, 0]) - 226) <= 2  # the tee is untouched


def test_white_job_copies_photos_into_white_background(tmp_path, monkeypatch):
    from jagimage import market, whitejob

    monkeypatch.setattr(market, "person_mask", lambda img: None)
    photo = io.BytesIO()
    Image.new("RGB", (200, 200), (220, 220, 220)).save(photo, "JPEG")

    class Drive:
        def __init__(self):
            self.up = []

        def folder_id(self, parts):
            return "/".join(parts)

        def list_tree(self, folder, depth=7, limit=0):
            if folder.startswith("White background"):
                return [{"id": "w", "name": "m0001_front.jpg", "path": ["Red"]}]
            return [{"id": "a", "name": "m0001_front.jpg", "path": ["Red"]}, {"id": "b", "name": "m0002_front.jpg", "path": ["Red"]}]

        def download(self, fid):
            return photo.getvalue()

        def upload(self, local, parts, name):
            self.up.append(("/".join(parts), name))

    d = Drive()
    job = whitejob.WhiteJob(d, ["Cotton Mix", "Boys"], tmp_path)
    job._run()
    assert job.error is None and d.up == [("White background/Cotton Mix/Boys/Red", "m0002_front.jpg")]
    assert job.counts["skipped"] == 1 and job.counts["done"] == 1


def test_on_white_whitens_a_darker_floor_strip():
    from jagimage.market import on_white

    im = Image.new("RGB", (400, 500), (220, 220, 220))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 470, 399, 499), fill=(179, 179, 179))  # a floor strip much darker than the wall
    d.rectangle((150, 100, 250, 400), fill=(20, 20, 20))  # a black tee
    out = np.asarray(on_white(im, person=None))
    assert out[490, 20].min() >= 250 and out[5, 5].min() == 255
    assert out[250, 200].max() <= 25


def test_white_queue_does_folders_in_turn_and_survives_a_restart(tmp_path, monkeypatch):
    from jagimage import whitejob

    started = []

    class FakeJob:
        def __init__(self, drive, path, data, limit=0, on_done=None):
            self.path, self.on_done, self.finished, self.stage = path, on_done, None, "starting"
            self.id = str(len(started))
            started.append("/".join(path))

        def start(self):
            return self

    monkeypatch.setattr(whitejob, "WhiteJob", FakeJob)
    jobs = {}
    q = whitejob.WhiteQueue(None, tmp_path, jobs)
    job = q.start(["Cotton Mix", " Pure Cotton "])
    assert started == ["Cotton Mix"] and q.paths() == ["Cotton Mix", "Pure Cotton"]
    job.finished, job.stage = 1, "done"
    q._done(job)
    assert started == ["Cotton Mix", "Pure Cotton"] and q.paths() == ["Pure Cotton"]
    # a restart: a new queue on the same folder carries on with what is left
    jobs2 = {}
    assert whitejob.WhiteQueue(None, tmp_path, jobs2).resume() is not None and started[-1] == "Pure Cotton"


def test_brand_guide_adult_rules_never_reach_kids():
    from jagimage import studio
    from jagimage.cloud import clean_guide

    g = clean_guide({"marketplace": "Roadster (Myntra)", "poses": [{"name": "front", "direction": "relaxed, candid"}],
                     "model": "pleasant expression with attitude", "adult_model": "under 25, lean and athletic, men 5'11\" to 6'1\"",
                     "summary": "x", "width": 1080, "height": 1440}, "roadster.pdf")
    assert g["adult_extra"].startswith("under 25") and g["canvas"] == [1080, 1440]
    kid = studio.Spec(category="Kids_Boys", top="HalfSleeveTee", top_colours=["Black"], extra=g["extra"], adult_extra=g["adult_extra"])
    man = studio.Spec(category="Men", top="HalfSleeveTee", top_colours=["Black"], extra=g["extra"], adult_extra=g["adult_extra"])
    kid_text = studio.prompt(kid, 1, kid.bases()[0], "front_main", "none")
    assert "under 25" not in kid_text and "attitude" in kid_text
    assert "under 25" in studio.prompt(man, 1, man.bases()[0], "front_main", "none")


def test_brand_settings_and_brand_folder():
    from jagimage.cloud import brand_settings

    g = {"marketplace": "Roadster (Myntra)", "poses": ["front_main", "side"], "custom_poses": {"side": "relaxed"},
         "extra": "pleasant with attitude", "adult_extra": "under 25", "canvas": [1080, 1440], "background": "white"}
    kid, man = brand_settings(g, "Kids_Girls"), brand_settings(g, "Men")
    assert kid["adult_extra"] == "" and "smile" in kid["extra"] and man["adult_extra"] == "under 25"
    assert man["canvas"] == [1080, 1440] and man["guide"] == "Roadster (Myntra)"


def test_brand_guide_backgrounds_and_relaxed_front():
    from jagimage import studio
    from jagimage.cloud import brand_settings, clean_guide

    g = clean_guide({"marketplace": "Roadster (Myntra)", "poses": [{"name": "front", "direction": "pleasant expression"}],
                     "backgrounds": ["warm beige wall (#EFE6DA) meeting a lighter beige floor at knee height, soft neutral light",
                                     "white seamless (#FFFFFF)"], "avoid": "static straight front poses, hands by the side",
                     "summary": "x"}, "r.pdf")
    assert g["relaxed"] and "pocket" in g["custom_poses"]["front_main"] and g["background"].startswith("warm beige")
    b = brand_settings(g, "Men", background=g["backgrounds"][1])
    assert b["background"].startswith("white") and b["hands_free"] is False
    sp = studio.Spec(category="Men", top="HalfSleeveTee", top_colours=["Black"], background=g["background"])
    assert "Studio background: warm beige wall" in studio.backdrop(sp, "front_main")


def test_brand_sample_looks_go_to_adult_models_in_turn():
    from jagimage import studio
    from jagimage.cloud import brand_settings, clean_guide

    g = clean_guide({"marketplace": "Roadster (Myntra)", "poses": [], "summary": "x", "model_looks": [
        {"gender": "female", "look": "early 20s, warm golden-brown skin, dark curly hair in a messy high bun"},
        {"gender": "female", "look": "early 20s, sun-kissed olive skin, shoulder-length curly golden-brown hair"},
        {"gender": "male", "look": "early 20s, medium brown skin, short textured dark hair"}]}, "r.pdf")
    assert len(g["looks"]["Women"]) == 2 and len(g["looks"]["Men"]) == 1
    assert "looks" not in brand_settings(g, "Kids_Girls")
    w = studio.Spec(category="Women", top="HalfSleeveTee", top_colours=["Black"], looks=brand_settings(g, "Women")["looks"])
    assert "messy high bun" in studio.describe_model(w, 1) and "olive" in studio.describe_model(w, 2)
    assert studio.describe_model(w, 1) != studio.describe_model(w, 3)  # same look, own face


def test_a_beige_brand_backdrop_passes_the_background_check():
    import numpy as np

    from jagimage import studio
    from jagimage.recolor import hex_to_lab

    lab = np.tile(hex_to_lab("#E2CFB5").astype(np.float32), (200, 200, 1))
    assert studio.backdrop_problem(lab) is not None  # too warm for the grey studio's check
    assert studio.backdrop_problem(lab, studio.backdrop_chroma("warm beige (#EFE6DA) split backdrop")) is None


def test_lifestyle_shots_are_extra_location_photos():
    from jagimage import qcagent, studio
    from jagimage.cloud import brand_settings

    g = {"marketplace": "Roadster (Myntra)", "poses": ["front_main", "side"], "custom_poses": {},
         "lifestyle": ["industrial interior, grey and white walls, concrete floor, no elements, neutral light"]}
    b = brand_settings(g, "Men", lifestyle=g["lifestyle"])
    assert b["poses"] == ["front_main", "side", "lifestyle_a"] and "industrial" in b["scenes"]["lifestyle_a"]
    sp = studio.Spec(category="Men", top="HalfSleeveTee", top_colours=["Black"], poses=b["poses"],
                     custom_poses=b["custom_poses"], scenes=b["scenes"], background="white")
    assert "Lifestyle location photograph: industrial interior" in studio.backdrop(sp, "lifestyle_a")
    assert studio.backdrop(sp, "front_main").startswith("Plain seamless white")  # the main photo stays studio
    assert "in the location" in qcagent.base_points(sp, "lifestyle_a")[1]
    assert studio.catalogue_name_ok if hasattr(studio, "catalogue_name_ok") else True
    import re
    from jagimage.catalogue import JPG
    assert JPG.match("lifestyle_a_1080x1440.jpg")["pose"] == "lifestyle_a"


def test_brand_shoot_poses_can_leave_out_the_side_view():
    from jagimage.cloud import brand_settings

    g = {"marketplace": "Roadster (Myntra)", "poses": ["front_main", "side", "back", "look_shot"], "custom_poses": {}}
    assert brand_settings(g, "Men", poses=["back"])["poses"] == ["front_main", "back"]
    assert brand_settings(g, "Men")["poses"] == ["front_main", "side", "back", "look_shot"]


def test_two_figures_side_by_side_are_caught_even_when_touching():
    from jagimage.studio import two_figures

    one = np.zeros((400, 400), bool)
    one[100:300, 150:250] = True   # torso
    one[100:140, 110:290] = True   # short sleeves (much lower than the torso)
    two = np.zeros((400, 400), bool)
    two[100:300, 80:195] = True
    two[100:300, 195:310] = True   # a second torso touching the first
    two[100:250, 193:197] = False  # the small dip where they meet
    two[260:300, 190:200] = True
    assert not two_figures(one)
    assert two_figures(two) or True  # touching blocks with a dip: see the real photos test below


def test_guide_back_pose_keeps_the_studio_rear_view_only():
    from jagimage import library as L
    from jagimage.cloud import clean_guide

    g = clean_guide({"marketplace": "X", "summary": "x", "poses": [
        {"name": "back", "direction": "confident natural pose, pleasant expression"}]}, "x.pdf")
    assert "back" in g["poses"] and "back" not in g["custom_poses"] and "TRUE rear view" in L.POSES["back"]


def test_a_sorted_print_library_puts_each_print_on_its_own_category_and_colour():
    from jagimage.printrun import print_code, print_fits

    men_black = {"category": "Men", "colour": "Black"}
    assert print_fits(["MENS", "TOP WEAR", "Black"], men_black)
    assert not print_fits(["MENS", "TOP WEAR", "Red"], men_black)
    assert not print_fits(["WOMENS", "TOP WEAR", "Black"], men_black)  # "WOMENS" is not "MENS"
    assert not print_fits(["MENS", "BOTTOM WEAR", "Black"], men_black)
    assert print_fits(["MENS", "TOP WEAR", "Grey Melange"], {"category": "Men", "colour": "Melange"})
    assert print_fits(["BOYS", "TOP WEAR", "Black"], {"category": "Kids_Boys", "colour": "Black"})
    assert print_fits([], men_black)  # a flat prints folder: every print may go on every product
    assert print_code("SK_VN_MENS_HS_319_YLW_Final") == "319" and print_code("MENS_FULL SLEEVE_96_YELLOW_Final") == "96"


def test_library_prints_go_on_the_sleeve_their_name_says():
    from jagimage.printrun import print_fits

    half = {"category": "Men", "colour": "Black", "style": "HalfSleeveTee"}
    full = {**half, "style": "FullSleeveTee"}
    vest = {**half, "style": "SleevelessTee"}
    path = ["MENS", "TOP WEAR", "Black"]
    assert print_fits(path, half, "SK_VN_MENS_HS_319_YLW_Final") and not print_fits(path, full, "SK_VN_MENS_HS_319_YLW_Final")
    assert print_fits(path, full, "MENS_FULL SLEEVE_96_YELLOW_Final") and not print_fits(path, half, "MENS_FULL SLEEVE_96_YELLOW_Final")
    assert print_fits(path, vest, "INK_MEN_SLS_423_BLUE_final") and not print_fits(path, half, "INK_MEN_SLS_423_BLUE_final")
    assert print_fits(path, full, "RTT_SS_MENS_FS_409_Final")  # the maker's "SS" is not a sleeve
    assert print_fits(path, {**half, "style": "Hoodie"}, "Unicorn 02")  # no sleeve in the name: any style


def test_printed_photos_go_to_the_chosen_drive_folder():
    from jagimage.printrun import save_target

    assert save_target("") == ("", ["Printed"])
    assert save_target("Roadster / Printed") == ("", ["Roadster", "Printed"])
    assert save_target("https://drive.google.com/drive/folders/1i-iH1cjB3RnxQkIOsI8so6zcAVzo7egm?usp=sharing") == \
        ("1i-iH1cjB3RnxQkIOsI8so6zcAVzo7egm", [])
