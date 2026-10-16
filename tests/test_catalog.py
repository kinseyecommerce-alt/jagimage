import base64
from types import SimpleNamespace

from jagimage import catalog
from jagimage.cli import main

PNG = base64.b64encode(b"png").decode()


class FakeImages:
    def __init__(self):
        self.calls = []

    def generate(self, **p):
        self.calls.append(("generate", p))
        return SimpleNamespace(data=[SimpleNamespace(b64_json=PNG)])

    def edit(self, image, **p):
        self.calls.append(("edit", dict(p, image=[f.name for f in image])))
        return SimpleNamespace(data=[SimpleNamespace(b64_json=PNG)])


KIDS = catalog.load_kids("examples/kids.json")


def test_sku_counts():
    assert len(list(catalog.skus(KIDS))) == 3 * 7 + 2 * 9
    assert len(list(catalog.skus(KIDS, gender="Girls", style="FullSleeveTee"))) == 9


def test_joggers_swap_only_for_full_sleeve():
    full = next(catalog.skus(KIDS, style="FullSleeveTee", colour="Navy"))
    half = next(catalog.skus(KIDS, style="HalfSleeveTee", colour="Navy"))
    assert "jogger pockets" in catalog.pose_prompt(full, "05-handpocket")
    assert "shorts" not in catalog.pose_prompt(full, "05-handpocket")
    assert "shorts pockets" in catalog.pose_prompt(half, "05-handpocket")
    assert "#0C2340" in catalog.front_prompt(half)


def test_face_ref_lead_and_poses_use_front(tmp_path):
    face = tmp_path / "face.jpg"
    face.write_bytes(b"x")
    s = next(catalog.skus(KIDS, gender="Boys", style="HalfSleeveTee", colour="Red"))
    s["face_ref"] = str(face)
    assert catalog.front_prompt(s).startswith("Use the reference portrait ONLY")

    client = SimpleNamespace(images=FakeImages())
    agent = catalog.CatalogAgent(client=client, output_dir=tmp_path / "out")
    results = list(agent.run_sku(s))
    assert [r[0] for r in results] == ["01-front", *catalog.POSES]
    kind, p = client.images.calls[0]
    assert kind == "edit" and p["image"] == [str(face)] and p["model"] == "gpt-image-2.5-flare"
    front = str(tmp_path / "out/Boys/HalfSleeveTee/Red/01-front.png")
    assert client.images.calls[1][1]["image"] == [front, str(face)]
    assert "input_fidelity" not in p


def test_resume_skips_existing(tmp_path):
    s = next(catalog.skus(KIDS, colour="White"))
    client = SimpleNamespace(images=FakeImages())
    agent = catalog.CatalogAgent(client=client, output_dir=tmp_path)
    list(agent.run_sku(s))
    assert client.images.calls[0][0] == "generate"  # no face ref -> text-to-image front
    again = list(agent.run_sku(s))
    assert {r[2] for r in again} == {"skipped"} and len(client.images.calls) == 5


def test_cli_dry_run(capsys):
    assert main(["catalog", "--kids", "examples/kids.json", "--gender", "Boys", "--colour", "Navy", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "3 SKUs x 5 shots = up to 15 images" in out and "Boys/SleevelessTee/Navy" in out


def test_cli_rejects_unknown_pose():
    assert main(["catalog", "--kids", "examples/kids.json", "--poses", "99-jump", "--dry-run"]) == 2


def test_fabric_text_unchanged_by_refactor():
    s = next(catalog.skus(KIDS, colour="Green"))
    assert ("Clearly visible fine single-jersey knit texture, matte dry cotton, no sheen, soft natural folds, ribbed crew collar, "
            "crisp double-needle stitching visible on hems, solid uniform dye, clean fabric with no lint or specks. Paired with") in catalog.front_prompt(s)
    assert "swatch" not in catalog.front_prompt(s)


def test_swatch_lookup_prefers_style_specific(tmp_path):
    (tmp_path / "Navy.jpg").write_bytes(b"x")
    (tmp_path / "FullSleeveTee-Navy.png").write_bytes(b"x")
    found = {s["style"]: s["swatch"] for s in catalog.skus(KIDS, gender="Boys", colour="Navy", swatches=str(tmp_path))}
    assert found["FullSleeveTee"].endswith("FullSleeveTee-Navy.png")
    assert found["HalfSleeveTee"].endswith("Navy.jpg")
    assert next(catalog.skus(KIDS, colour="Red", swatches=str(tmp_path)))["swatch"] is None


def test_texture_shot_and_swatch_refs(tmp_path):
    (tmp_path / "Red.jpg").write_bytes(b"x")
    s = next(catalog.skus(KIDS, gender="Boys", style="HalfSleeveTee", colour="Red", swatches=str(tmp_path)))
    client = SimpleNamespace(images=FakeImages())
    agent = catalog.CatalogAgent(client=client, output_dir=tmp_path / "out")
    names = [r[0] for r in agent.run_sku(s, poses=["04-back"], texture=True)]
    assert names == ["01-front", "04-back", "06-fabric"]
    swatch = str((tmp_path / "Red.jpg").resolve())
    front = str(tmp_path / "out/Boys/HalfSleeveTee/Red/01-front.png")
    assert [c[1]["image"] for c in client.images.calls] == [[swatch], [front, swatch], [front, swatch]]
    assert client.images.calls[0][1]["prompt"].startswith("Use the fabric swatch photo ONLY")
    assert "Extreme macro close-up" in client.images.calls[2][1]["prompt"]
