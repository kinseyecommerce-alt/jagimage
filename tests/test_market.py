import io
import json
import threading
import time
import urllib.error
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from jagimage import market, studio
from jagimage.server import serve
from test_studio import _get, _post, fake_generate


def studio_photo():
    """A grey studio photo with a dark figure and a contact shadow under it, 2:3 like an AI photo."""
    img = Image.new("RGB", (400, 600), "#E6E6E6")
    d = ImageDraw.Draw(img)
    d.ellipse((150, 560, 250, 590), fill="#C9C9C9")   # contact shadow on the floor
    d.rectangle((160, 120, 240, 570), fill="#20304A")   # the model
    d.ellipse((175, 50, 225, 120), fill="#C99B80")
    return img


def test_every_market_gets_its_size_and_white_where_asked():
    for key, (_, (w, h), white) in market.MARKETS.items():
        out = market.for_market(studio_photo(), key)
        a = np.asarray(out).astype(int)
        assert out.size == (w, h), key
        corner = a[5, 5]
        if white:
            assert (corner == 255).all(), key                       # pure white main image
            y = round(585 * h / 600) if w / 400 > h / 600 else h // 2 + round((585 - 300) * min(w / 400, h / 600))
            assert a[:, w // 2][max(0, y - 30):y + 1].min() < 250, key  # the floor shadow is kept
        else:
            assert abs(corner.mean() - 0xE6) < 6, key                # studio grey kept
        assert a[h // 2, w // 2].mean() < 90                          # the model is still there


def test_market_download_for_a_batch(tmp_path):
    app, httpd = serve(port=0, out=str(tmp_path), engine=lambda spec: fake_generate)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        code, job = _post(base + "/api/jobs", {"category": "Men", "top": "HalfSleeveTee", "bottom": "Trackpants",
                                              "top_colours": ["Navy", "White"], "models": 1, "keep_png": False})
        for _ in range(600):
            st = json.loads(_get(f"{base}/api/jobs/{job['id']}")[1])
            if not st["running"]:
                break
            time.sleep(0.1)
        opts = json.loads(_get(base + "/api/options")[1])
        assert {m["key"] for m in opts["markets"]} == set(market.MARKETS)
        code, data = _get(f"{base}/api/batches/{st['name']}/market?to=myntra")
        assert code == 200
        z = zipfile.ZipFile(io.BytesIO(data))
        names = z.namelist()
        assert len(names) == 2 and all(n.startswith(f"{st['name']}_Myntra/") for n in names)
        assert Image.open(z.open(names[0])).size == (1500, 2000)
        cached = sorted((tmp_path / st["name"] / "_market" / "myntra").glob("*.jpg"))
        assert len(cached) == 2
        mtime = cached[0].stat().st_mtime
        _get(f"{base}/api/batches/{st['name']}/market?to=myntra")
        assert cached[0].stat().st_mtime == mtime                       # made once, then reused
        try:
            _get(f"{base}/api/batches/{st['name']}/market?to=ebay")
            assert False, "an unknown marketplace must be refused"
        except urllib.error.HTTPError as e:
            assert e.code == 400 and b"Pick a marketplace" in e.read()
        assert len(studio.Batch(studio.Spec(**json.loads((tmp_path / st["name"] / "spec.json").read_text())), out=tmp_path).results()) == 2
    finally:
        httpd.shutdown()
