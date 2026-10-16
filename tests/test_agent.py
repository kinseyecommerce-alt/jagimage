import base64
from types import SimpleNamespace

import pytest

from jagimage import Brand, ImageAgent
from jagimage.cli import main

PNG = base64.b64encode(b"\x89PNG fake").decode()


class FakeImages:
    def __init__(self):
        self.calls = []

    def _respond(self, kind, **params):
        self.calls.append((kind, params))
        return SimpleNamespace(data=[SimpleNamespace(b64_json=PNG) for _ in range(params.get("n", 1))])

    def generate(self, **params):
        return self._respond("generate", **params)

    def edit(self, **params):
        return self._respond("edit", **params)


@pytest.fixture
def client():
    return SimpleNamespace(images=FakeImages())


def test_generate_uses_flare_and_preset(client, tmp_path):
    agent = ImageAgent(client=client, output_dir=tmp_path, brand=Brand(style="minimal", avoid=["text"]))
    paths = agent.create("steel water bottle", shot="social", n=2)
    kind, params = client.images.calls[0]
    assert kind == "generate"
    assert params["model"] == "gpt-image-2.5-flare"
    assert params["size"] == "1024x1536"
    assert "steel water bottle" in params["prompt"] and "Avoid: text." in params["prompt"]
    assert [p.name for p in paths] == ["social-1.png", "social-2.png"]
    assert paths[0].read_bytes() == b"\x89PNG fake"


def test_photo_uses_edit_and_never_overwrites(client, tmp_path):
    photo = tmp_path / "bag.jpg"
    photo.write_bytes(b"img")
    agent = ImageAgent(client=client, output_dir=tmp_path / "out")
    first = agent.create("leather bag", shot="lifestyle", photo=str(photo))
    second = agent.create("leather bag", shot="lifestyle", photo=str(photo))
    kind, params = client.images.calls[0]
    assert kind == "edit" and "reference image" in params["prompt"]
    assert first[0] != second[0]


def test_transparent_forces_png(client, tmp_path):
    agent = ImageAgent(client=client, output_dir=tmp_path, output_format="jpeg")
    path = agent.create("mug", shot="cutout")[0]
    assert client.images.calls[0][1]["background"] == "transparent"
    assert path.suffix == ".png"


def test_unknown_shot(client, tmp_path):
    with pytest.raises(ValueError):
        ImageAgent(client=client, output_dir=tmp_path).create("mug", shot="nope")


def test_batch_continues_after_error(client, tmp_path):
    csv_file = tmp_path / "p.csv"
    csv_file.write_text("name,description,shots,photo\nMug,white,\"hero,bogus\",\nLamp,,,\n")
    agent = ImageAgent(client=client, output_dir=tmp_path / "out")
    results = list(agent.run_batch(str(csv_file)))
    assert [(n, s) for n, s, _ in results] == [("Mug", "hero"), ("Mug", "bogus"), ("Lamp", "hero")]
    assert isinstance(results[1][2], ValueError)


def test_cli_lists_shots(capsys):
    assert main(["shots"]) == 0
    assert "hero" in capsys.readouterr().out
