import json
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from jagimage import library as L
from jagimage.cli import main
from jagimage.providers import GeminiProvider
from jagimage.recolor import recolor_image


def fake_photo(garment_hex, size=(300, 400)):
    """Grey studio, skin face, black hair, a shaded garment in garment_hex, white shoes."""
    img = Image.new("RGB", size, "#EDEDED")
    d = ImageDraw.Draw(img)
    d.ellipse((115, 30, 185, 110), fill="#C99B80")           # face
    d.ellipse((112, 22, 188, 70), fill="#111111")             # hair
    d.polygon([(90, 120), (210, 120), (240, 200), (215, 210), (205, 330), (95, 330), (85, 210), (60, 200)], fill=garment_hex)
    arr = np.asarray(img).astype(np.float32)
    shade = np.linspace(0.8, 1.1, size[0])[None, :, None]     # side light and knit noise
    rng = np.random.default_rng(0)
    g = (np.abs(arr - np.array(Image.new("RGB", (1, 1), garment_hex))[0, 0]).sum(-1) < 1)[..., None]
    arr = np.where(g, np.clip(arr * shade + rng.normal(0, 3, arr.shape), 0, 255), arr)
    out = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    ImageDraw.Draw(out).rectangle((100, 360, 200, 380), fill="#FAFAFA")  # shoes
    return out


def test_recolor_hits_hex_and_leaves_hair_and_skin():
    base = fake_photo("#1E4636")
    out, de, mask = recolor_image(base, "#1E4636", "#CD001A")
    assert de < 3
    o, b = np.asarray(out.convert("RGB")), np.asarray(base)
    assert (o[40, 150] == b[40, 150]).all() and (o[90, 150] == b[90, 150]).all()  # hair, face untouched
    assert mask[200, 150] > 0.95 and mask[10, 10] < 0.05


def test_pure_black_keeps_its_knit():
    # #000000 can't show folds or knit, so it renders as photographed black and keeps the texture
    out, de, mask = recolor_image(fake_photo("#1E4636"), "#1E4636", "#000000")
    o = np.asarray(out.convert("L")).astype(float)[mask > 0.95]
    assert de < 3 and 15 < o.mean() < 40 and o.std() > 2


def test_white_edge_has_no_tint_and_follows_the_photo():
    from scipy import ndimage
    from jagimage.recolor import rgb_to_lab
    out, _, mask = recolor_image(fake_photo("#8FD3A8"), "#8FD3A8", "#FFFFFF")
    lab = rgb_to_lab(np.asarray(out.convert("RGB")))
    m = mask > 0.5
    edge = ndimage.binary_dilation(m, iterations=3) & ~ndimage.binary_erosion(m, iterations=3)
    assert np.hypot(lab[..., 1], lab[..., 2])[edge].max() < 0.03  # no pink or mint outline
    # the photo's own anti-aliased edge is kept: a soft step, not a hard on/off staircase
    red, _, _ = recolor_image(fake_photo("#1E4636"), "#1E4636", "#CD001A")
    a = rgb_to_lab(np.asarray(red.convert("RGB")))[250, :150, 1]
    assert ((a > 0.01) & (a < 0.15)).any() and a.max() > 0.18


def test_backdrop_between_legs_is_not_recoloured():
    img = fake_photo("#8FD3A8")
    d = ImageDraw.Draw(img)
    d.polygon([(150, 250), (147, 330), (153, 330)], fill="#EDEDED")  # thin backdrop gap between the legs
    d.line([(95, 330), (205, 330)], fill="#8FD3A8", width=3)           # legs touch below it: an enclosed hole
    out, _, mask = recolor_image(img, "#8FD3A8", "#FFE01B")
    assert mask[300, 150] < 0.5 and mask[270, 150] < 0.5
    assert (np.asarray(out.convert("RGB"))[300, 150] == np.asarray(img)[300, 150]).all()


def test_bright_backdrop_gap_between_arm_and_body_is_not_recoloured():
    """Studio backdrops are brighter mid-frame than at the photo's edges: a thin gap between a sleeve
    and the body can be lighter than the edge grey by more than a fixed tolerance. It stays backdrop."""
    img = fake_photo("#8FD3A8")
    d = ImageDraw.Draw(img)
    d.rectangle((209, 150, 211, 205), fill="#FFFFFF")  # 3 px bright gap inside the sleeve line
    out, _, mask = recolor_image(img, "#8FD3A8", "#E31B23")
    assert mask[180, 210] < 0.5
    assert np.asarray(out.convert("RGB"))[180, 210].min() > 240  # still white backdrop, not red


def test_backdrop_slot_between_garment_and_hand_gets_no_paint():
    """Where garment, backdrop and a hand meet, the nearest surround of a grey rim pixel is the hand:
    unmixing against it called the grey half garment and painted a red block beside the hand."""
    img = fake_photo("#8FD3A8")
    d = ImageDraw.Draw(img)
    d.rectangle((206, 290, 207, 328), fill="#EDEDED")   # 2 px backdrop slot along the garment's side
    d.rectangle((208, 290, 236, 328), fill="#6B4A3A")   # the hand just beyond it
    out, _, mask = recolor_image(img, "#8FD3A8", "#E31B23")
    px = np.asarray(out.convert("RGB")).astype(int)[300:320, 206:208]
    assert (np.abs(px - np.asarray(img).astype(int)[300:320, 206:208]).max() < 12)  # still backdrop grey


def test_hem_shadow_between_top_and_bottom_is_painted():
    """Pixel values from a real Flash photo: mint shirt, a dark grey-lilac shadow line under the hem
    (in the top's mask), lilac trousers. Recoloured top then bottom, the shadow line must not keep
    its base colour as a dotted line along the hem."""
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[:19], rgb[19], rgb[20:] = (167, 210, 193), (109, 103, 131), (182, 148, 199)
    top = np.zeros((40, 60), np.float32)
    top[:20] = 1
    bottom = 1 - top
    img = Image.fromarray(rgb)
    out, _, _ = recolor_image(img, "#8FD3A8", "#E31B23", mask=top, others=bottom)
    out, _, _ = recolor_image(out, "#C4A3EE", "#E31B23", mask=bottom, others=top)
    r, g, b = np.asarray(out.convert("RGB")).astype(int)[19, 10:50].T
    assert (r - g > 60).all() and (r - b > 60).all()


def test_pale_tinted_halo_and_shoes_beside_a_deep_base_stay_out():
    """Gemini Flash leaves a faint purple tint in the bright backdrop along the trousers and in the
    shoe shadows: same hue, pale. It must not join the mask (it showed as a ragged dark fringe)."""
    img = fake_photo("#3B1F66")
    d = ImageDraw.Draw(img)
    d.rectangle((206, 150, 214, 330), fill="#E4DCEC")   # pale lilac halo touching the garment's side
    d.rectangle((100, 331, 200, 362), fill="#D9CFE3")   # tinted shoe shadow under the hem, joined to the shoes
    out, _, mask = recolor_image(img, "#3B1F66", "#000000")
    assert mask[250, 210] < 0.5 and mask[345, 150] < 0.5
    assert mask[250, 150] > 0.95  # the garment itself is still all in


def test_tinted_or_coloured_backdrop_is_never_the_garment():
    """A studio with a faint mint cast (or a maroon studio, for a lilac base's hue range) used to be
    swallowed by the garment mask, so the whole background was recoloured."""
    from jagimage.recolor import garment_mask, rgb_to_lab
    for studio, base in [("#D4E6DA", "#8FD3A8"), ("#E6DCEC", "#C4A3EE"), ("#6E1520", "#C4A3EE")]:
        img = fake_photo(base)
        arr = np.asarray(img).copy()
        bg = (np.abs(arr.astype(int) - 0xED).sum(-1) < 6)
        arr[bg] = np.array(Image.new("RGB", (1, 1), studio))[0, 0]
        m, _ = garment_mask(rgb_to_lab(arr), base)
        assert m[10, 10] < 0.5 and m[200, 20] < 0.5 and m[390, 280] < 0.5, studio  # backdrop stays out
        assert m[250, 150] > 0.95, studio                                           # the garment is still in


def test_sleeve_hems_of_a_real_flash_photo_recolour_without_a_torn_fringe():
    """A real Gemini/OpenAI base (tests/data): the AI draws a thin, uneven brown shadow on the arm under
    each sleeve hem. Recoloured black, its darkest pixels used to be painted one by one, so the hem
    looked torn. Nothing outside the shirt may be painted dark, and the hem line stays even."""
    from pathlib import Path
    from scipy import ndimage
    from jagimage.recolor import garment_mask, rgb_to_lab
    img = Image.open(Path(__file__).parent / "data" / "men_flash_base.jpg").convert("RGB")
    before = rgb_to_lab(np.asarray(img))
    bottom, _ = garment_mask(before, "#C4A3EE")
    out, _, mask = recolor_image(img, "#8FD3A8", "#000000", others=bottom)
    after = rgb_to_lab(np.asarray(out.convert("RGB")))
    for y0, y1, x0, x1 in [(440, 540, 290, 400), (430, 530, 660, 770)]:  # left and right sleeve hem
        a, b, g = after[y0:y1, x0:x1, 0], before[y0:y1, x0:x1, 0], mask[y0:y1, x0:x1] > 0.5
        near = ndimage.binary_dilation(g, iterations=4) & ~g
        assert not (near & (a < b - 0.15)).any()  # no skin painted dark
        dark = a < 0.3
        hem = np.array([np.nonzero(dark[:, x])[0].max() for x in range(dark.shape[1]) if dark[:, x].any()])
        assert (np.abs(np.diff(hem)) > 1).sum() <= 8  # an even hem line, not a ragged one


def test_white_keeps_knit_and_stitches():
    from jagimage.recolor import rgb_to_lab
    img = fake_photo("#8FD3A8")
    ImageDraw.Draw(img).line([(100, 300), (200, 300)], fill="#6FA888", width=1)  # a stitch line
    out, de, mask = recolor_image(img, "#8FD3A8", "#FFFFFF")
    L = rgb_to_lab(np.asarray(out.convert("RGB")))[..., 0]
    assert de < 3 and L[mask > 0.95].std() > 0.005      # texture, not a flat white shape
    assert L[300, 150] < L[290, 150] - 0.02              # the stitch still shows


def test_crushed_black_creases_dont_become_blotches():
    from jagimage.recolor import rgb_to_lab
    img = fake_photo("#1A1A1A")
    ImageDraw.Draw(img).rectangle((140, 220, 160, 260), fill="#000000")  # a crease the camera crushed to 0,0,0
    out, _, mask = recolor_image(img, "#1A1A1A", "#FFE01B")
    L = rgb_to_lab(np.asarray(out.convert("RGB")))[..., 0]
    assert abs(L[240, 150] - L[240, 125]) < 0.08  # a soft crease, not a dark patch with hard edges


def test_thin_crease_joins_the_garment_but_hair_does_not():
    from jagimage.recolor import garment_mask, rgb_to_lab
    img = fake_photo("#8FD3A8")
    d = ImageDraw.Draw(img)
    d.line([(95, 331), (205, 331)], fill="#20302A", width=2)        # dark hem crease under the garment
    d.rectangle((100, 112, 200, 119), fill="#151515")                # a big dark area touching the top (hair)
    m, _ = garment_mask(rgb_to_lab(np.asarray(img)), "#8FD3A8")
    assert m[331, 150] > 0.5 and m[114, 150] < 0.5


def test_gemini_provider_sends_refs_then_prompt_and_returns_png(tmp_path):
    ref = tmp_path / "face.png"
    fake_photo("#8FD3A8").save(ref)
    calls = []

    def generate_content(model, contents, config):
        calls.append((model, contents, config))
        import io
        buf = io.BytesIO()
        Image.new("RGB", (30, 40), "red").save(buf, format="JPEG")
        part = SimpleNamespace(inline_data=SimpleNamespace(data=buf.getvalue()))
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    png = GeminiProvider(client=client, image_size="4K").generate("make it", [str(ref)])
    assert png[:4] == b"\x89PNG"
    model, contents, config = calls[0]
    assert model == "gemini-3.1-flash-image" and contents[-1] == "make it" and len(contents) == 2
    assert config.image_config.image_size == "4K" and config.image_config.aspect_ratio == "3:4"


class BaseShooter:
    """Fake image engine: draws the garment in whatever hex the prompt asks for."""
    def __init__(self):
        self.prompts = []

    def _call(self, prompt, refs):
        self.prompts.append(prompt)
        hex_ = re.search(r"Exact garment colour .*? \((#[0-9A-F]{6})\)", prompt).group(1)
        import io
        buf = io.BytesIO()
        fake_photo(hex_).save(buf, format="PNG")
        return buf.getvalue()


def test_two_base_pipeline(tmp_path):
    out = str(tmp_path)
    runner = L.LibraryRunner(BaseShooter(), L.load_models(None), out=out)
    style = {"category": "Kids_Boys", "product": "HalfSleeveTee", "fabric": "PureCotton"}
    poses = ["front_main", "back"]
    for which in ("light", "dark"):
        list(runner.run(L.base_item(style, which), poses))
    assert any("plain Mint regular-fit" in p for p in runner.agent.prompts)
    results = {}
    for it in L.items(["Kids_Boys"], ["HalfSleeveTee"], ["PureCotton"]):
        rows = list(L.recolor_item(it, poses, out=out))
        assert [r[2] for r in rows] == ["made", "made"]
        results[it["colour"]] = max(r[3] for r in rows)
    # near-white keeps its folds as shadow, so its average sits a little under #FFFFFF
    assert len(results) == 9 and max(v for k, v in results.items() if k != "White") < 3, results
    assert results["White"] < 4.5
    qc = json.loads((tmp_path / "Kids_Boys/HalfSleeveTee/PureCotton/Yellow/qc.json").read_text())
    assert qc["front_main"]["from"].startswith("light base") and qc["front_main"]["on_shade"]
    navy = json.loads((tmp_path / "Kids_Boys/HalfSleeveTee/PureCotton/Navy/qc.json").read_text())
    assert navy["back"]["from"].startswith("dark base")
    # rerun skips; a hand-made mask is reused
    assert {r[2] for r in L.recolor_item(it, poses, out=out)} == {"skipped"}
    assert (tmp_path / "Kids_Boys/HalfSleeveTee/PureCotton/_base_light/front_main_mask.png").exists()


def test_cli_bases_dry_run(capsys):
    assert main(["library", "--bases", "--category", "Men", "--product", "Hoodie", "--fabric", "CottonMix",
                 "--poses", "front_main", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 styles x 2 base colours x 1 poses = 2 AI images; 17 colours" in out
    assert "plain mint green regular-fit pullover hoodie" in out and "Exact garment colour bottle green (#1E4636)" in out
    assert "Men/Hoodie/CottonMix/Beige <- light base" in out and "Men/Hoodie/CottonMix/Maroon <- dark base" in out


def test_dark_base_copies_child_from_light_front(tmp_path):
    out = str(tmp_path)
    shooter = BaseShooter()
    calls = []
    orig = shooter._call
    shooter._call = lambda prompt, refs: calls.append((prompt, list(refs))) or orig(prompt, refs)
    runner = L.LibraryRunner(shooter, L.load_models(None), out=out)
    style = {"category": "Kids_Boys", "product": "Shorts", "fabric": "CottonMix"}
    assert {s for _, _, s in runner.run(L.base_item(style, "dark"), ["front_main"])} == {"waiting for the light base front"}
    list(runner.run(L.base_item(style, "light"), ["front_main"]))
    list(runner.run(L.base_item(style, "dark"), ["front_main", "back"]))
    light_front = str(tmp_path / "Kids_Boys/Shorts/CottonMix/_base_light/front_main.png")
    dark_front = str(tmp_path / "Kids_Boys/Shorts/CottonMix/_base_dark/front_main.png")
    assert calls[0][1] == []                                    # light front: new child
    assert calls[1][1] == [light_front]                         # dark front: same child as light
    assert calls[1][0].startswith("Use the reference image ONLY for identity")
    assert calls[2][1] == [light_front, dark_front]             # dark poses: face + approved dark front
