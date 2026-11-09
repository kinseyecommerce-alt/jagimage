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

    def upload(self, local, parts, name):
        self.uploaded["/".join(parts + [name])] = local.read_bytes()


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
