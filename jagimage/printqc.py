"""Print QC: the last check on every printed photo before it is saved, uploaded or listed. It runs on the
server from the photo itself (no AI cost), and holds a photo with the reasons in plain words:

- the print shows on the shirt's colour (not black line art on a black tee)
- the print shop's placement rules: below the collar, above the hem, not too wide or too small, centred,
  not hidden by an arm or hand
- the print is on the shirt, not on a face, the neck or an arm
- nothing else in the photo changed (only the print's own box)
- the artwork is sharp enough for the size it is printed at (not a small file blown up)
- the SKU and file name are complete (every part of the SKU filled, the photo named as the print)"""

import re
from typing import List, Optional

import numpy as np
from PIL import Image

from .prints import PrintError, _layout, measure, rule_check
from .recolor import rgb_to_lab

MIN_WIDTH = 0.22     # a print narrower than this share of the chest looks lost on the shirt
MAX_UPSCALE = 3.0    # artwork shown this many times its own size looks soft or blocky
OUTSIDE_DE = 0.04    # OKLab: a pixel outside the print's box this far from the plain photo has changed
OUTSIDE_MAX = 0.002  # at most this share of the photo outside the print may change
VISIBILITY = "shows"  # the name of the visibility check (a print that doesn't show is left out, not held)


def check(before: Image.Image, after: Image.Image, mask: np.ndarray, art: Image.Image, place: str,
          adj: Optional[dict] = None, sku: str = "", file_name: str = "", print_name: str = "") -> dict:
    """QC of one printed photo: {"ok", "problems": [plain sentences], "failed": [check names], "scores": {...}}."""
    from .printrun import SKIN_MAX, VISIBLE_MIN, on_skin, visible_share

    adj = adj or {}
    problems, failed, scores = [], [], {}

    def fail(name: str, why: str):
        failed.append(name)
        problems.append(why)

    try:
        g = _layout(mask, art, place, **adj)
        info = measure(mask, art, place, **adj)
    except PrintError as exc:
        return {"ok": False, "problems": [str(exc)], "failed": ["fits"], "scores": {}}
    scores.update(info)
    seen = visible_share(before, after, g)
    scores["shows"] = round(seen, 3)
    if seen < VISIBLE_MIN:
        fail(VISIBILITY, f"only {round(seen * 100)}% of the print shows on this shirt colour")
    issues, _ = rule_check(info, place)
    for issue in issues:
        fail("placement", issue)
    if info["width"] < MIN_WIDTH:
        fail("placement", "too small for the chest")
    skin = on_skin(before, g, mask)
    scores["on_skin"] = round(skin, 3)
    if skin > SKIN_MAX:
        fail("on_skin", f"{round(skin * 100)}% of the print lies on skin (face, neck or arm)")
    # nothing but the print's own box may change
    a = rgb_to_lab(np.asarray(before.convert("RGB").reduce(4)))
    b = rgb_to_lab(np.asarray(after.convert("RGB").reduce(4)))
    if a.shape != b.shape:
        fail("photo", "the printed photo is not the same size as the plain one")
    else:
        changed = np.linalg.norm(b - a, axis=-1) > OUTSIDE_DE
        y0, x0 = max(0, g["y0"] // 4 - 2), max(0, g["x0"] // 4 - 2)
        changed[y0:(g["y0"] + g["ph"]) // 4 + 3, x0:(g["x0"] + g["pw"]) // 4 + 3] = False
        scores["changed_outside"] = round(float(changed.mean()), 4)
        if changed.mean() > OUTSIDE_MAX:
            fail("photo", "the photo changed outside the print")
    up = g["pw"] / max(1, g["art"].width)
    scores["upscale"] = round(up, 2)
    if up > MAX_UPSCALE:
        fail("sharpness", f"the artwork is low-resolution ({g['art'].width}px wide, shown {up:.1f}× bigger): "
                          "it will look blurry; use a bigger file")
    if sku and (not re.fullmatch(r"[A-Za-z0-9_]+(-[A-Za-z0-9_]+)+", sku) or sku.endswith("-U") or "-U-" in sku):
        fail("sku", f"the SKU {sku} is incomplete (a part couldn't be filled in)")
    if file_name and print_name and not file_name.lower().startswith(str(print_name).lower()[:20]):
        fail("name", f"the photo {file_name} isn't named after the print {print_name}")
    return {"ok": not failed, "problems": problems, "failed": failed, "scores": scores}


def only_visibility(result: dict) -> bool:
    """A print that only fails on not showing is left out of that colour (a pairing, not a fault to review)."""
    return bool(result["failed"]) and set(result["failed"]) == {VISIBILITY}


def summary(results: List[dict]) -> dict:
    """How many passed and what failed most, for the tab."""
    out = {"checked": len(results), "passed": sum(1 for r in results if r["ok"]), "reasons": {}}
    for r in results:
        for name in set(r["failed"]):
            out["reasons"][name] = out["reasons"].get(name, 0) + 1
    return out


def summary_rows(rows: List[dict]) -> dict:
    """The whole run's print QC: SKUs checked, passed, held, and the reasons counted."""
    out = {"skus": len(rows), "passed": sum(1 for r in rows if r["passed"] == r["checked"]), "reasons": {}}
    out["held"] = out["skus"] - out["passed"]
    for r in rows:
        for k, v in r["reasons"].items():
            out["reasons"][k] = out["reasons"].get(k, 0) + v
    return out
