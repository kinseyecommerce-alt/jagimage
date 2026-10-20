import json
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from jagimage import library as L
from jagimage.cli import main
from jagimage.providers import GeminiProvider
from jagimage.recolor import hex_to_lab, recolor_image


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
    assert model == "gemini-3-pro-image" and contents[-1] == "make it" and len(contents) == 2
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
    assert len(results) == 9 and max(results.values()) < 3, results
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
