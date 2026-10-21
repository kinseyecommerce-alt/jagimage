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
USD_INR = float(os.environ.get("JAGIMAGE_USD_INR", "96.5"))


def gemini_model() -> str:
    from .providers import GEMINI_DEFAULT_MODEL

    return os.environ.get("JAGIMAGE_GEMINI_MODEL", GEMINI_DEFAULT_MODEL).replace("-preview", "")


def per_image(provider: str, setting: str, model: Optional[str] = None) -> Optional[float]:
    """All-in USD estimate for one AI image (output + input), or None when no price is known
    (e.g. OpenAI xhigh/max, which OpenAI doesn't list per image)."""
    if provider == "gemini":
        model = model or gemini_model()
        out = OUTPUT.get(("gemini", model), {}).get(setting)
        return None if out is None else out + INPUT.get(model, 0.003)
    out = OUTPUT[("openai", "*")].get(setting)
    return None if out is None else out + INPUT["openai"]


def estimate(spec) -> dict:
    """Money for a batch: AI images x price, with an allowance for retries."""
    plan = spec.plan()
    setting = spec.image_size if spec.provider == "gemini" else spec.quality
    each = per_image(spec.provider, setting)
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
            "prices_checked": PRICES_CHECKED, "batch": bool(getattr(spec, "batch", False)), "model": gemini_model() if spec.provider == "gemini" else "openai"}


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
