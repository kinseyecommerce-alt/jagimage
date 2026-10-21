import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from jagimage import pricing, studio
from jagimage.providers import RunPodProvider, make_provider
from test_studio import fake_generate

TOKEN = "pilot-token"


@pytest.fixture
def pod():
    """A stand-in for pilot/runpod/qwen_server.py: same API, fake_generate instead of the GPU."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self._json({"error": "wrong token"}, 401)
            self._json({"ok": True, "model": "Qwen/Qwen-Image-Edit-2511", "gpu": "fake H100", "offload": False})

        def do_POST(self):
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self._json({"error": "wrong token"}, 401)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            png = fake_generate(body["prompt"], [None] * len(body["images"]))
            self._json({"png": base64.b64encode(png).decode(), "seconds": 30.0})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{srv.server_port}", seen=seen)
    srv.shutdown()


def test_runpod_provider_sends_prompt_refs_and_token(pod, tmp_path):
    ref = tmp_path / "front.png"
    ref.write_bytes(fake_generate("top mint (#8FD3A8), bottom lilac (#C4A3EE)", []))
    p = RunPodProvider(url=pod.url + "/", token=TOKEN)
    png = p.generate("top mint (#8FD3A8), bottom lilac (#C4A3EE)", [str(ref)])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and p.seconds == [30.0]
    sent = pod.seen[0]
    assert base64.b64decode(sent["images"][0]) == ref.read_bytes()
    assert (sent["width"], sent["height"], sent["steps"]) == (1104, 1472, 40)


def test_runpod_needs_address_and_token(monkeypatch):
    monkeypatch.delenv("JAGIMAGE_RUNPOD_URL", raising=False)
    monkeypatch.delenv("JAGIMAGE_RUNPOD_TOKEN", raising=False)
    with pytest.raises(ValueError, match="RunPod"):
        make_provider("runpod")
    with pytest.raises(ValueError, match="provider"):
        studio.Spec(top_colours=["Red"], provider="midjourney")
    with pytest.raises(ValueError, match="Batch mode"):
        studio.Spec(top_colours=["Red"], provider="runpod", batch=True)


def test_batch_on_runpod_makes_bases_and_recolours(pod, tmp_path):
    spec = studio.Spec(top_colours=["Navy", "White"], models=2, poses=["front_main", "back"], provider="runpod")
    p = make_provider("runpod", url=pod.url, api_key=TOKEN)
    b = studio.Batch(spec, out=tmp_path, generate=p.generate, workers=2)
    b.run()
    assert b.counts["errors"] == 0 and b.counts["ai_done"] == spec.plan()["ai_images"]
    assert b.counts["recoloured"] == spec.plan()["recoloured_images"]
    firsts = [s for s in pod.seen if not s["images"]]
    assert len(firsts) == 2  # only each model's first front is made without a reference photo


def test_wrong_token_or_stopped_pod_stops_the_batch(pod, tmp_path):
    spec = studio.Spec(top_colours=["Red"], models=3, provider="runpod")
    for url, token, says in ((pod.url, "wrong", "refused the token"), ("http://127.0.0.1:9", TOKEN, "isn't answering")):
        b = studio.Batch(spec, out=tmp_path / says, generate=RunPodProvider(url=url, token=token).generate, workers=1)
        b.run()
        assert says in b.fatal and b.counts["ai_done"] == 0


def test_runpod_cost_is_gpu_hours():
    c = pricing.estimate(studio.Spec(top_colours=["Red"], models=1000, poses=["front_main", "back"], provider="runpod"))
    each = pricing.RUNPOD_USD_PER_HOUR * pricing.RUNPOD_SECONDS / 3600
    assert c["per_image_usd"] == round(each, 4) and c["gpu_seconds_per_image"] == pricing.RUNPOD_SECONDS
    assert c["model"] == "qwen-image-edit-2511"


def test_server_saves_the_pod_address_and_token(monkeypatch):
    from jagimage import server

    monkeypatch.delenv("JAGIMAGE_RUNPOD_URL", raising=False)
    monkeypatch.delenv("JAGIMAGE_RUNPOD_TOKEN", raising=False)
    assert server.ready()["runpod"] is False
    monkeypatch.setenv("JAGIMAGE_RUNPOD_TOKEN", TOKEN)
    assert server.ready()["runpod"] is False  # token without an address isn't enough
    monkeypatch.setenv("JAGIMAGE_RUNPOD_URL", "https://abc-8000.proxy.runpod.net")
    assert server.ready()["runpod"] is True and server.options()["runpod_url"].endswith("runpod.net")


def test_pilot_compare_reports_cost_and_pass_rate(pod, tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pilot" / "runpod"))
    import compare

    monkeypatch.setenv("JAGIMAGE_RUNPOD_URL", pod.url)
    monkeypatch.setenv("JAGIMAGE_RUNPOD_TOKEN", TOKEN)
    # Navy and Red both recolour from the dark base: 2 models x 2 poses = 4 AI images
    r = compare.main(["--category", "Men", "--models", "2", "--colours", "Navy,Red", "--out", str(tmp_path)])
    assert r["bases_accepted"] == 4 and r["pass_rate"] == 1.0 and r["gpu_seconds_per_image"] == 30.0
    assert r["pod"]["gpu"] == "fake H100" and Path(r["sheet"]).exists()
    assert json.loads((tmp_path / "report.json").read_text())["recoloured"] == r["recoloured"]
