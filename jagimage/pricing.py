"""Cost estimates for a Studio batch: only the AI base photos cost money; recolouring runs on your
computer for free. List prices checked on PRICES_CHECKED from ai.google.dev/gemini-api/docs/pricing
and developers.openai.com/api/docs/pricing; check them again before a big order."""

import os
from typing import Optional

PRICES_CHECKED = "2026-10-21"

# USD per generated image at list price (standard, not Batch API).
OUTPUT = {
    ("gemini", "gemini-3-pro-image"): {"1K": 0.134, "2K": 0.134, "4K": 0.24},
    ("gemini", "gemini-3.1-flash-image"): {"1K": 0.067, "2K": 0.101, "4K": 0.151},
    # OpenAI's published per-image figures for 1024x1536 (GPT Image 2; gpt-image-2.5-flare has the same token rates)
    ("openai", "*"): {"low": 0.005, "medium": 0.041, "high": 0.165},
}
# Estimated input per image: the ~900-token prompt plus 0-2 reference photos (identity / approved front).
INPUT = {"gemini-3-pro-image": 0.003, "gemini-3.1-flash-image": 0.001, "openai": 0.02}
RETRY_ALLOWANCE = 0.10  # images the app re-asks for when the AI gets the base colours wrong, network errors
BATCH_DISCOUNT = 0.5    # batch mode: half the standard price (Google and OpenAI list it this way)
# RunPod: you rent the GPU by the hour, so an image costs (price per hour) x (seconds per image).
# Defaults: H100 SXM Community Cloud list price and a guess at Qwen-Image-Edit at 40 steps; the pilot
# measures the real seconds (pilot/runpod/compare.py), then set JAGIMAGE_RUNPOD_SECONDS to that.
RUNPOD_USD_PER_HOUR = float(os.environ.get("JAGIMAGE_RUNPOD_USD_PER_HOUR", "2.69"))
RUNPOD_SECONDS = float(os.environ.get("JAGIMAGE_RUNPOD_SECONDS", "45"))
USD_INR = float(os.environ.get("JAGIMAGE_USD_INR", "96.5"))
# USD per 1M tokens at standard price (batch is half): input text/image, image output, other output.
TOKEN_PRICES = {"gemini-3.1-flash-image": (0.50, 60.0, 3.0), "gemini-3-pro-image": (2.0, 120.0, 12.0)}
TOKENS_PER_IMAGE = {"gemini-3.1-flash-image": 1680, "gemini-3-pro-image": 1120}  # 2K


def gemini_model(choice: Optional[str] = None) -> str:
    """Model id for a Studio choice (flash/pro), else the command-line default."""
    from .providers import GEMINI_DEFAULT_MODEL, GEMINI_MODELS

    if choice:
        return GEMINI_MODELS[choice]
    return os.environ.get("JAGIMAGE_GEMINI_MODEL", GEMINI_DEFAULT_MODEL).replace("-preview", "")


def per_image(provider: str, setting: str, model: Optional[str] = None) -> Optional[float]:
    """All-in USD estimate for one AI image (output + input), or None when no price is known
    (e.g. OpenAI xhigh/max, which OpenAI doesn't list per image)."""
    if provider == "runpod":
        return RUNPOD_USD_PER_HOUR * RUNPOD_SECONDS / 3600
    if provider == "gemini":
        model = model or gemini_model()
        out = OUTPUT.get(("gemini", model), {}).get(setting)
        return None if out is None else out + INPUT.get(model, 0.003)
    out = OUTPUT[("openai", "*")].get(setting)
    return None if out is None else out + INPUT["openai"]


def estimate(spec) -> dict:
    """Money for a batch: AI images x price, with an allowance for retries."""
    plan = spec.plan()
    setting = {"gemini": spec.image_size, "openai": spec.quality}.get(spec.provider, "gpu")
    model = gemini_model(getattr(spec, "gemini_model", None)) if spec.provider == "gemini" else None
    each = per_image(spec.provider, setting, model)
    if each is not None and getattr(spec, "batch", False):
        each *= BATCH_DISCOUNT  # Gemini batch mode: output and input both half price
    if each is None:
        return {"per_image_usd": None, "note": f"No list price for {spec.provider} {setting}; pick another setting to estimate."}
    base = plan["ai_images"] * each
    total = base * (1 + RETRY_ALLOWANCE)
    photos = max(1, plan["recoloured_images"])
    return {"per_image_usd": round(each, 4), "ai_cost_usd": round(base, 2), "with_retries_usd": round(total, 2),
            "per_photo_usd": round(total / photos, 4), "usd_inr": USD_INR, "with_retries_inr": round(total * USD_INR),
            "per_photo_inr": round(total / photos * USD_INR, 2), "retry_allowance": RETRY_ALLOWANCE,
            "prices_checked": PRICES_CHECKED, "batch": bool(getattr(spec, "batch", False)), "model": {"gemini": model, "runpod": "qwen-image-edit-2511"}.get(spec.provider, "openai"),
            **({"gpu_usd_per_hour": RUNPOD_USD_PER_HOUR, "gpu_seconds_per_image": RUNPOD_SECONDS} if spec.provider == "runpod" else {})}


def sum_estimates(rows: list, photos: int) -> dict:
    """One estimate for several batches (e.g. the same models in several top styles)."""
    if len(rows) == 1 or any(r.get("per_image_usd") is None for r in rows):
        return rows[0]
    out = dict(rows[0])
    for k in ("ai_cost_usd", "with_retries_usd"):
        out[k] = round(sum(r[k] for r in rows), 2)
    out["with_retries_inr"] = round(out["with_retries_usd"] * out["usd_inr"])
    out["per_photo_usd"] = round(out["with_retries_usd"] / max(1, photos), 4)
    out["per_photo_inr"] = round(out["with_retries_usd"] * out["usd_inr"] / max(1, photos), 2)
    return out


def actual_cost(usage_file, photos: int = 0) -> Optional[dict]:
    """The real AI cost of a batch, from Google's billed token counts (saved per request in
    _usage.jsonl), rather than an estimate. Includes photos the app rejected and asked for again,
    and extra images the AI sometimes returns in one response (billed, though only one is used)."""
    import json
    from pathlib import Path

    p = Path(usage_file)
    if not p.exists():
        return None
    usd = extra = 0.0
    n = unknown = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rates = TOKEN_PRICES.get(r.get("model"))
        if not rates:
            unknown += 1
            continue
        half = BATCH_DISCOUNT if r.get("batch") else 1.0
        per_image = TOKENS_PER_IMAGE[r["model"]]
        usd += half * (r["prompt"] * rates[0] + r["image_out"] * rates[1] + r["other_out"] * rates[2]) / 1e6
        extra += half * max(0, r["image_out"] - per_image) * rates[1] / 1e6
        n += 1
    if not n:
        return None
    out = {"requests": n, "usd": round(usd, 2), "inr": round(usd * USD_INR), "per_request_inr": round(usd * USD_INR / n, 2),
           "extra_images_inr": round(extra * USD_INR), "usd_inr": USD_INR, "unpriced_requests": unknown}
    if photos:
        out["per_photo_inr"] = round(usd * USD_INR / photos, 2)
    return out
