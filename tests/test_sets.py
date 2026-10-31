import io
from pathlib import Path

import pytest

from jagimage import library as L
from jagimage import sheets, studio
from test_studio import fake_generate

WOMEN = {"Black": "#000000", "Blue": "#006CD1", "Maroon": "#3D0202", "White": "#FFFFFF", "Red": "#CF1D1D",
         "Yellow": "#FFAB19", "Light Pink": "#F6ADD2", "Dark Pink": "#E43F8C"}
MEN = {"Black": "#000000", "Blue": "#006CD1", "Green": "#132E13", "Maroon": "#3D0202", "Melange": "#DADDDA",
       "White": "#FFFFFF", "Red": "#CF1D1D", "Yellow": "#FFAB19"}


@pytest.fixture
def shop_colours():
    before = {f: {c: dict(v) for c, v in cats.items()} for f, cats in L.CATEGORY_COLOURS.items()}
    L.set_colours("CottonMix", WOMEN, ["Women", "Kids_Girls"])
    L.set_colours("CottonMix", MEN, ["Men", "Kids_Boys"])
    yield
    for f, cats in before.items():
        for c, cols in cats.items():
            L.set_colours(f, cols, [c])


def sheet(rows_by_sheet) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in rows_by_sheet.items():
        ws = wb.create_sheet(name)
        ws.append(["S.NO", "CATEGORY", "STYLE", "TOP COLOR", "BOTTOM COLOR", "MODEL IMAGES SET REQUIRED QTY"])
        for i, r in enumerate(rows, 1):
            ws.append([i, *r])
        ws.append([None, None, None, None, "TOTAL", sum(r[-1] for r in rows)])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_a_sets_sheet_becomes_coord_and_combo_batches(shop_colours):
    data = sheet({
        "SETS": [("BOYS", "SET-HOOD-TRACK PANT", "BLACK", "GREY", 60), ("BOYS", "SET-HOOD-TRACK PANT", "GREY", "BLACK", 45),
                 ("BOYS", "SET-HOOD-TRACK PANT", "WHITE", "BLACK", 60), ("BOYS", "SET-HALF SLEEVE-SHORTS", "R.BLUE", "GREY", 30)],
        "COORDS": [("GIRLS", "COORD-FULL SLEEVE-PANT", "D.PINK", "D.PINK", 60), ("GIRLS", "COORD-FULL SLEEVE-PANT", "L.PINK", "L.PINK", 45),
                   ("GIRLS", "COORD-HALF SLEEVE-SHORTS", "WHITE", "WHITE", 40), ("GIRLS", "COORD-HOOD-TRACK PANT", "M.GREEN", "M.GREEN", 10)],
    })
    bodies, problems = sheets.read_plan(data, "CottonMix")
    by = {(b["category"], b["top"], b["set_type"]): b for b in bodies}
    hood = by[("Kids_Boys", "Hoodie", "combo")]
    assert hood["bottom"] == "Trackpants" and hood["models"] == 60 and hood["shot"] == "outfit"
    assert hood["colour_pairs"] == [["Black", "Melange", 60], ["Melange", "Black", 45], ["White", "Black", 60]]
    assert by[("Kids_Boys", "HalfSleeveTee", "combo")]["colour_pairs"] == [["Blue", "Melange", 30]]
    full = by[("Kids_Girls", "FullSleeveTee", "coord")]
    assert full["colour_pairs"] == [["Dark Pink", "Dark Pink", 60], ["Light Pink", "Light Pink", 45]] and full["models"] == 60
    # a white coord gets a black bottom (bottoms aren't made in white)
    assert by[("Kids_Girls", "HalfSleeveTee", "coord")]["colour_pairs"] == [["White", "Black", 40]]
    assert any("COORDS row 4" in p and "Black bottom" in p for p in problems)
    # Green isn't one of the girls' Cotton Mix colours
    assert any("COORDS row 5" in p and "M.GREEN" in p for p in problems)
    assert ("Kids_Girls", "Hoodie", "coord") not in by
    for b in bodies:  # every batch is a valid Studio batch
        sp = studio.Spec(**{**b, "provider": "openai"})
        assert sp.name.endswith("_Coord" if b["set_type"] == "coord" else "_Combo")


def test_colour_pairs_go_only_on_their_models(tmp_path, shop_colours):
    sp = studio.Spec(category="Kids_Boys", top="Hoodie", bottom="Trackpants", fabric="CottonMix", models=3, set_type="combo",
                     colour_pairs=[["Black", "Melange", 3], ["Melange", "Black", 1]], poses=["front_main"])
    assert sp.pairs(1) == [("Black", "Melange"), ("Melange", "Black")] and sp.pairs(3) == [("Black", "Melange")]
    assert sp.bases(1) == [("dark", "light"), ("light", "dark")] and sp.bases(2) == [("dark", "light")]
    p = sp.plan()
    assert p["ai_images"] == 4 and p["recoloured_images"] == 4   # model 1: two shoots; models 2-3: one
    with pytest.raises(ValueError, match="aren't made in white"):
        studio.Spec(category="Kids_Boys", top="Hoodie", bottom="Trackpants", fabric="CottonMix", colour_pairs=[["Black", "White"]])
    with pytest.raises(ValueError, match="between 1 and 2"):
        studio.Spec(category="Kids_Boys", top="Hoodie", bottom="Trackpants", fabric="CottonMix", models=2, colour_pairs=[["Black", "Melange", 5]])
    counts = studio.Batch(sp, out=tmp_path, generate=fake_generate, workers=1).run()
    assert counts["ai_done"] == 4 and counts["recoloured"] == 4 and counts["errors"] == 0
    made = sorted(str(f.relative_to(tmp_path / sp.name)) for f in (tmp_path / sp.name).glob("m*/*/front_main_*.jpg"))
    assert made == ["m0001/Black__Melange/front_main_1500x1500.jpg", "m0001/Melange__Black/front_main_1500x1500.jpg",
                    "m0002/Black__Melange/front_main_1500x1500.jpg", "m0003/Black__Melange/front_main_1500x1500.jpg"]


def test_the_shops_real_plan_reads_whole(shop_colours):
    plan = Path("/root/.claude/uploads/073f8049-dff8-5d97-9ec3-1cbe28eadb51/23aaac49-KKI_sets_plan.xlsx")
    if not plan.exists():
        pytest.skip("the shop's sheet isn't on this machine")
    bodies, problems = sheets.read_plan(plan.read_bytes(), "CottonMix")
    sets = sum(n for b in bodies for _, _, n in b["colour_pairs"])
    assert len(bodies) == 15 and sets == 2250 + 3905
    assert all("Black bottom" in p for p in problems) and len(problems) == 3  # the three white coords


def test_sets_plan_upload_prices_and_runs_on_the_website(tmp_path, shop_colours, monkeypatch):
    import base64
    import json
    import threading
    import time

    from jagimage.cloud import Agent, Cloud, Drive
    from jagimage.server import serve
    from test_cloud import PASSWORD, call

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cloud = Cloud(tmp_path / "data", lambda: str(tmp_path / "out"), password=PASSWORD, drive=Drive("", "", tmp_path / "data"),
                  agent=Agent(client=None, model="gemini-test"))
    app, httpd = serve(port=0, out=str(tmp_path / "out"), engine=lambda spec: fake_generate, cloud=cloud)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cookie = call(base + "/login", {"password": PASSWORD})[1]["Set-Cookie"].split(";")[0]
    data = sheet({"SETS": [("BOYS", "SET-HOOD-TRACK PANT", "BLACK", "GREY", 2), ("BOYS", "SET-HOOD-TRACK PANT", "RED", "BLACK", 1)],
                  "COORDS": [("GIRLS", "COORD-HALF SLEEVE-SHORTS", "WHITE", "WHITE", 1)]})
    try:
        code, _, body = call(base + "/api/sets", {"filename": "plan.xlsx", "data": base64.b64encode(data).decode(),
                                                  "fabric": "CottonMix", "provider": "openai", "quality": "medium"}, cookie=cookie)
        plan = json.loads(body)
        assert code == 200, plan
        assert [(b["category"], b["set_type"], b["pairs"]) for b in plan["batches"]] == [
            ("Kids_Boys", "combo", ["Black/Melange ×2", "Red/Black ×1"]), ("Kids_Girls", "coord", ["White/Black ×1"])]
        assert plan["total"]["photos"] == (2 + 1 + 1) * 2 and plan["total"]["cost_inr"] > 0 and len(plan["problems"]) == 1
        assert call(base + "/api/sets", {"data": base64.b64encode(b"not excel").decode(), "provider": "openai"}, cookie=cookie)[0] == 400
        assert call(base + "/api/agent/start", {"plan_id": plan["plan_id"]}, cookie=cookie)[0] == 200
        for _ in range(900):
            runs = json.loads(call(base + "/api/agent/runs", cookie=cookie)[2])["runs"]
            if not runs[0]["running"]:
                break
            time.sleep(0.1)
        assert runs[0]["done"] == 2 and not runs[0]["error"], runs
        out = tmp_path / "out"
        assert len(list(out.glob("Kids_Boys_Hoodie_Trackpants_CottonMix_Combo/m*/*/front_main_*.jpg"))) == 3
        assert list(out.glob("Kids_Girls_HalfSleeveTee_Shorts_CottonMix_Coord/m*/White__Black/back_*.jpg"))
    finally:
        httpd.shutdown()
