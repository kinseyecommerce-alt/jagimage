"""RunPod pilot: make a few models' base photos on your pod (Qwen-Image-Edit-2511) through the same
Studio flow as Gemini, recolour them, and report what it costs and how often the AI got it right.

    export JAGIMAGE_RUNPOD_URL=https://<pod-id>-8000.proxy.runpod.net
    export JAGIMAGE_RUNPOD_TOKEN=<the token the pod was started with>
    python pilot/runpod/compare.py --category Men --models 10 --gpu-usd-per-hour 2.69 \
        --gemini output/studio/test_Men

Writes <out>/report.json and <out>/sheet.jpg: one row per model (front and back base, three recolours),
with the Gemini photos of the same category in the last row for comparison. Uses its own output
folder, so the pilot doesn't use up model numbers in your real catalogue.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jagimage import pricing, studio  # noqa: E402
from jagimage.providers import RunPodProvider  # noqa: E402

THUMB = 300


def health(p: RunPodProvider) -> dict:
    from urllib.request import Request, urlopen

    req = Request(p.url + "/health", headers={"Authorization": f"Bearer {p.token}"})
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def thumb(path: Path):
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((THUMB, THUMB * 4 // 3))
    tile = Image.new("RGB", (THUMB, THUMB * 4 // 3), "white")
    tile.paste(im, ((THUMB - im.width) // 2, (tile.height - im.height) // 2))
    return tile


def row_images(model_dir: Path, colours):
    base = sorted(model_dir.glob("_base/*/front_main.png"))[:1] + sorted(model_dir.glob("_base/*/back.png"))[:1]
    looks = [model_dir / studio.pair_dir(c, b) / "front_main.png" for c, b in colours]
    return [p for p in base + looks if p.exists()]


def sheet(rows, out: Path):
    from PIL import Image, ImageDraw

    rows = [(name, imgs) for name, imgs in rows if imgs]
    if not rows:
        return None
    cols = max(len(i) for _, i in rows)
    h = THUMB * 4 // 3
    canvas = Image.new("RGB", (120 + cols * THUMB, len(rows) * h), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (name, imgs) in enumerate(rows):
        draw.text((8, r * h + h // 2), name, fill="black")
        for c, p in enumerate(imgs):
            canvas.paste(thumb(p), (120 + c * THUMB, r * h))
    canvas.save(out, quality=88)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="Men", choices=list(studio.L.CATEGORIES))
    ap.add_argument("--top", default="HalfSleeveTee", choices=studio.TOPS)
    ap.add_argument("--bottom", default="Shorts", choices=studio.BOTTOMS)
    ap.add_argument("--models", type=int, default=10)
    ap.add_argument("--colours", default="Navy,White,Red", help="top colours to recolour to (comma separated)")
    ap.add_argument("--poses", default="front_main,back")
    ap.add_argument("--out", default="output/pilot_runpod")
    ap.add_argument("--gemini", help="an existing Gemini batch folder of the same category, for the sheet")
    ap.add_argument("--gpu-usd-per-hour", type=float, default=pricing.RUNPOD_USD_PER_HOUR,
                    help="what your pod costs (RunPod console > Pods); H100 SXM community is $2.69")
    ap.add_argument("--steps", type=int, default=None, help="Qwen steps (default 40; fewer is faster, rougher)")
    a = ap.parse_args(argv)

    provider = RunPodProvider(steps=a.steps)
    info = health(provider)
    print(f"Pod OK: {info.get('gpu')} running {info.get('model')}{' (CPU offload)' if info.get('offload') else ''}")

    colours = [c.strip() for c in a.colours.split(",") if c.strip()]
    spec = studio.Spec(category=a.category, top=a.top, bottom=a.bottom, top_colours=colours, pairing="match",
                       models=a.models, first_model=studio.ModelLedger(a.out).next_free(a.category),
                       poses=[p.strip() for p in a.poses.split(",")], provider="runpod",
                       name=f"pilot_{a.category}_{a.top}")
    calls = {"n": 0}

    def generate(prompt, refs):
        calls["n"] += 1
        return provider.generate(prompt, refs)

    t0 = time.time()

    def on_event(e):
        if e["kind"] in ("ai_done", "errors"):
            print(f"  {e['kind']}: model {e.get('model')} {Path(e.get('path', '')).name} {e.get('error', '')}", flush=True)

    # One GPU makes one image at a time, and the pod's proxy address drops requests that wait over ~100 s.
    batch = studio.Batch(spec, out=a.out, generate=generate, workers=1, on_event=on_event)
    batch.run()
    wall = time.time() - t0

    made = batch.counts["ai_done"]
    gpu = provider.seconds
    per_image_gpu = sum(gpu) / len(gpu) if gpu else 0
    # The pod bills for every second it runs, including rejected images and idle gaps: cost from wall time.
    usd = a.gpu_usd_per_hour * wall / 3600
    report = {
        "pod": info, "category": a.category, "models": a.models, "poses": spec.poses,
        "ai_calls": calls["n"], "bases_accepted": made, "errors": batch.counts["errors"],
        "pass_rate": round(made / calls["n"], 3) if calls["n"] else None,
        "gpu_seconds_per_image": round(per_image_gpu, 1), "wall_minutes": round(wall / 60, 1),
        "pod_cost_usd": round(usd, 2), "pod_cost_inr": round(usd * pricing.USD_INR),
        "per_accepted_base_inr": round(usd * pricing.USD_INR / made, 2) if made else None,
        "gemini_batch_per_base_inr": round(pricing.per_image("gemini", "2K") * pricing.BATCH_DISCOUNT * pricing.USD_INR, 2),
        "recoloured": batch.counts["recoloured"], "fatal": batch.fatal,
    }
    rows = [(f"model {d.name[1:].lstrip('0')}", row_images(d, spec.pairs()[:3]))
            for d in sorted(batch.root.glob("m[0-9][0-9][0-9][0-9]"))]
    if a.gemini:
        g = sorted(Path(a.gemini).glob("m[0-9][0-9][0-9][0-9]"))
        if g:
            rows.append(("Gemini", row_images(g[0], spec.pairs()[:3])))
    report["sheet"] = str(sheet(rows, Path(a.out) / "sheet.jpg"))
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nNow check sheet.jpg by eye: same face front and back? real-looking fabric and hands? Then "
          "STOP THE POD in the RunPod console (it bills by the second while running).")
    return report


if __name__ == "__main__":
    main()
