import base64
import io
import json

import pytest
from PIL import Image

from jagimage import pricing, studio
from jagimage.providers import MetaProvider, make_provider


def png(colour="#1E4636"):
    buf = io.BytesIO()
    Image.new("RGB", (20, 30), colour).save(buf, "PNG")
    return buf.getvalue()


class Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def test_meta_sends_reference_photos_as_json_with_search_off(tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(png())
    sent = []

    def opener(req, timeout):
        sent.append(req)
        return Reply(json.dumps({"data": [{"b64_json": base64.b64encode(png("#8FD3A8")).decode()}]}).encode())

    out = MetaProvider(api_key="k", opener=opener).generate("a model in a t-shirt", [str(ref)])
    assert Image.open(io.BytesIO(out)).size == (20, 30)
    req = sent[0]
    body = json.loads(req.data)
    assert req.full_url == "https://api.meta.ai/v1/images/edits" and req.get_header("Authorization") == "Bearer k"
    assert body["model"] == "muse-image-1.0" and body["output_format"] == "png" and body["size"] == "1024x1536"
    assert body["images"][0]["image_url"].startswith("data:image/png;base64,")
    assert not any(body["tool_enablement"].values())
    assert "quality" not in body and "input_fidelity" not in body  # Meta refuses unknown fields

    MetaProvider(api_key="k", opener=opener).generate("text only", [])
    assert sent[1].full_url.endswith("/images/generations") and "images" not in json.loads(sent[1].data)


def test_meta_needs_a_key_and_is_priced_flat(monkeypatch):
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Meta Model API key"):
        make_provider("meta")
    spec = studio.Spec(category="Men", top="HalfSleeveTee", bottom="Trackpants", top_colours=["Navy"], models=10,
                       poses=["front_main", "back"], provider="meta")
    e = pricing.estimate(spec)
    assert e["per_image_usd"] == 0.01 and e["model"] == "muse-image-1.0"
