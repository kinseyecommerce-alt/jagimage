# RunPod pilot: base photos from an open model on your own GPU

The goal is to find out whether **Qwen-Image-Edit-2511** (open model, Apache 2.0 licence, free for commercial
use) on a rented GPU makes base photos good enough to replace Gemini. If it does, a base photo costs
about ₹2–4 instead of ₹6.6 (Gemini batch) or ₹13 (Gemini standard).

Budget for the pilot is about **$10–20 (₹1,000–2,000)**: 1–3 hours of an H100 plus the model download.

## 1. RunPod account

1. Sign up at runpod.io and add credit (Billing → **$20** is plenty).
2. **Pods → Deploy**:
   - GPU: **H100 SXM 80 GB** (Community Cloud, about $2.69/h). An **A100 80 GB** is cheaper and slower.
     The model needs about 60 GB of GPU memory. A 48 GB card (L40S, A6000) works with `QWEN_OFFLOAD=1`, but it is several times slower.
   - Template: **RunPod PyTorch** (2.4 or newer).
   - Container disk: 30 GB. Volume disk: **80 GB** (holds the ~58 GB model so a restart doesn't download it again).
   - **Expose HTTP ports: `8000`** (keep 8888 for Jupyter).
   - Environment variables:
     - `JAGIMAGE_RUNPOD_TOKEN` = a long random password you make up (for example 32 letters and numbers).
     - `HF_HOME` = `/workspace/hf`
3. Deploy and wait until the pod shows *Running*.

## 2. Start the server on the pod

Open the pod's **Jupyter Lab** (Connect → port 8888) and upload `qwen_server.py` to `/workspace`. Then, in a Jupyter terminal:

```bash
pip install -U "git+https://github.com/huggingface/diffusers" transformers accelerate
cd /workspace && python qwen_server.py
```

The first start downloads the model (about 58 GB, 5–15 minutes), then prints `Serving on 0.0.0.0:8000`.
Leave that terminal running.

The pod's address is **`https://<pod-id>-8000.proxy.runpod.net`** (Connect → HTTP 8000). Check it from your computer:

```bash
curl -H "Authorization: Bearer <your token>" https://<pod-id>-8000.proxy.runpod.net/health
```

## 3. Run the pilot (on your computer)

```bash
export JAGIMAGE_RUNPOD_URL=https://<pod-id>-8000.proxy.runpod.net
export JAGIMAGE_RUNPOD_TOKEN=<your token>
python pilot/runpod/compare.py --category Men --models 10 --gpu-usd-per-hour 2.69 --gemini output/studio/test_Men
```

This makes 10 new models (front and back), checks every base photo's colours exactly as for Gemini,
recolours Navy, White and Red, and writes:

- `output/pilot_runpod/report.json`: seconds per image, how often the AI got the base colours right (`pass_rate`),
  what the pod cost, and **₹ per accepted base photo** next to Gemini batch.
- `output/pilot_runpod/sheet.jpg`: one row per model, with Gemini in the last row. Look at it closely:
  - Is it the same face front and back?
  - Does it look like a real person (not a mannequin)?
  - Are the hands natural and not touching the clothes?
  - Is the fabric real-looking?

You can also use the pod from the Studio app: **Engine → RunPod (own GPU)**, paste the address and token,
**Save key**, and keep **parallel requests at 1**. The pod's web address gives up on a request after about
100 seconds, so queued requests fail.

## 4. Stop paying

**Stop the pod** in the RunPod console as soon as you're done: it bills by the second while running.
The 80 GB volume costs about $5.60 a month while stopped. **Terminate** the pod if you won't use it again.

## Reading the result

| `pass_rate` and sheet | What it means |
|---|---|
| ≥ 0.8, faces consistent, looks real | Open model works: move base photos to RunPod; cost is GPU hours |
| 0.5–0.8 | Usable but needs prompt tuning for Qwen; retries raise the real cost |
| < 0.5 or faces drift | Stay on Gemini (batch mode); recolouring is already free |

Then set `JAGIMAGE_RUNPOD_SECONDS` to the measured `gpu_seconds_per_image`, so the Studio app's cost
estimate uses the real figure.
