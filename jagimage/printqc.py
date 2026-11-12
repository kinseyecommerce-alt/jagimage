"""Print QC: the last check on every printed photo before it is saved, uploaded or listed. It runs on the
server from the photo itself (no AI cost), and holds a photo with the reasons in plain words:

- the print shows on the shirt's colour (not black line art on a black tee)
- the print shop's placement rules (a zip hoodie's print never across the zip): below the collar, above the hem, not too wide or too small, centred,
  not hidden by an arm or hand
- the print is on the shirt: not on a face, the neck or an arm, not under a hand, and never painted over a hand
- nothing else in the photo changed (only the print's own box)
- the artwork is sharp enough for the size it is printed at (not a small file blown up)
- the SKU and file name are complete (every part of the SKU filled, the photo named as the print)"""

import re
from typing import List, Optional

import numpy as np
from PIL import Image

from .prints import MAX_HIDDEN as MAX_HIDDEN_RULE, PrintError, _layout, measure, rule_check
from .recolor import rgb_to_lab

MIN_WIDTH = 0.22     # a print narrower than this share of the chest looks lost on the shirt
MAX_UPSCALE = 3.0    # artwork shown this many times its own size looks soft or blocky
OUTSIDE_DE = 0.04    # OKLab: a pixel outside the print's box this far from the plain photo has changed
OUTSIDE_MAX = 0.002  # at most this share of the photo outside the print may change
HAND_MAX = 0.02      # at most this share of a print may go under a hand or arm (the photo shows the shirt, not the hand)
INK_ON_SKIN_MAX = 0.01  # at most this share of the print's ink may be painted over skin (a hand the shirt mask took in)
SKIN_FROM_SHIRT = 0.08  # OKLab: skin this far from the shirt's colour is told apart from a pink or beige shirt
VISIBILITY = "shows"  # the name of the visibility check (a print that doesn't show is left out, not held)


def check(before: Image.Image, after: Image.Image, mask: np.ndarray, art: Image.Image, place: str,
          adj: Optional[dict] = None, sku: str = "", file_name: str = "", print_name: str = "", style: str = "",
          shirt_hex: str = "") -> dict:
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
    if style == "ZipHoodie" and not str(place).startswith(("left_chest", "zip_chest")):
        fail("placement", "a zip hoodie opens down the middle: the print would run across the zip")
    issues, _ = rule_check(info, place)
    for issue in issues:
        fail("placement", issue)
    if info["width"] < MIN_WIDTH and not str(place).startswith(("left_chest", "zip_chest")):  # a left-chest logo is small on purpose
        fail("placement", "too small for the chest")
    if MAX_HIDDEN_RULE >= info["hidden"] > HAND_MAX:
        fail("hand", f"{round(info['hidden'] * 100)}% of the print goes under a hand or arm")
    # ink painted over a hand: skin-coloured pixels (not the shirt's own colour) that the print changed
    x0, y0, x1, y1 = g["x0"], g["y0"], g["x0"] + g["pw"], g["y0"] + g["ph"]
    a0 = rgb_to_lab(np.asarray(before.convert("RGB").crop((x0, y0, x1, y1))))
    b0 = rgb_to_lab(np.asarray(after.convert("RGB").crop((x0, y0, x1, y1))))
    C = np.hypot(a0[..., 1], a0[..., 2])
    skinish = (a0[..., 1] > 0.015) & (a0[..., 2] > 0.0) & (C > 0.035)
    if shirt_hex:
        from .recolor import hex_to_lab

        try:
            dt = a0 - hex_to_lab(shirt_hex)
            dt[..., 0] *= 0.35  # a fold's shadow on a pink shirt is darker, not skin
            skinish &= np.linalg.norm(dt, axis=-1) > SKIN_FROM_SHIRT
        except (ValueError, TypeError):
            pass
    ink = np.asarray(g["art"].resize((g["pw"], g["ph"])).getchannel("A"))[:a0.shape[0], :a0.shape[1]] > 128
    painted = ink & skinish & (np.linalg.norm(b0 - a0, axis=-1) > OUTSIDE_DE)
    share = float(painted.sum()) / max(1, int(ink.sum()))
    scores["ink_on_skin"] = round(share, 3)
    if share > INK_ON_SKIN_MAX:
        fail("hand", f"{round(share * 100)}% of the print is painted over skin (a hand or arm)")
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
