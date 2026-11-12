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
SKIN_FROM_SHIRT = 0.08
SKIN_C_MAX = 0.13       # OKLab chroma: skin is never this saturated (a yellow or orange shirt is)  # OKLab: skin this far from the shirt's colour is told apart from a pink or beige shirt
PART_SHARE = 0.06    # a colour making up at least this share of a print's ink is a part of it (the red 5 in GYM 5)
PART_SEEN_MIN = 0.35  # ...and at least this share of that part must show, or the print is left out of that colour
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
    else:
        lost = lost_part(before, after, g)
        if lost:
            fail(VISIBILITY, f"{lost} of the print hardly shows on this shirt colour")
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
    hue = np.degrees(np.arctan2(a0[..., 2], a0[..., 1]))
    # skin: a warm, muted colour (a yellow or orange shirt is far more saturated, a red one redder)
    skinish = (a0[..., 1] > 0.015) & (C > 0.035) & (C < SKIN_C_MAX) & (hue > 20) & (hue < 80)
    # ...and not the shirt's own colour, as it is in this photo (folds and shade included)
    small = before.convert("RGB").reduce(4)
    on = np.asarray(Image.fromarray(((np.asarray(mask) > 0.5) * 255).astype(np.uint8)).resize(small.size)) > 127
    ref = [np.median(rgb_to_lab(np.asarray(small))[on], axis=0)] if on.any() else []
    if shirt_hex:
        from .recolor import hex_to_lab

        try:
            ref.append(hex_to_lab(shirt_hex))
        except (ValueError, TypeError):
            pass
    for r in ref:
        dt = a0 - r
        dt[..., 0] *= 0.35  # a fold's shadow is darker, not skin
        skinish &= np.linalg.norm(dt, axis=-1) > SKIN_FROM_SHIRT
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


def _colour_word(rgb) -> str:
    from .printrun import nearest_colour

    hex_ = "#%02X%02X%02X" % tuple(int(v) for v in rgb)
    return (nearest_colour(hex_) or "a coloured").lower() + " part"


def lost_part(before: Image.Image, after: Image.Image, g: dict) -> str:
    """A coloured part of the print (one of its main ink colours) that hardly shows on this shirt, though the rest
    does: the red 5 of a white-and-red print on a red tee. Its name ("the red part"), or "" when every part shows."""
    from scipy import ndimage

    from .printrun import VISIBLE_DE

    pw, ph = g["pw"], g["ph"]
    pad = max(3, int(0.03 * max(pw, ph)))  # a margin of bare shirt around the print
    x0, y0 = max(0, g["x0"] - pad), max(0, g["y0"] - pad)
    x1, y1 = min(after.width, g["x0"] + pw + pad), min(after.height, g["y0"] + ph + pad)
    b = rgb_to_lab(np.asarray(after.convert("RGB").crop((x0, y0, x1, y1))))
    art = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    art.paste(g["art"].resize((pw, ph)), (g["x0"] - x0, g["y0"] - y0))
    ink = np.asarray(art.getchannel("A")) > 128
    if ink.sum() < 50:
        return ""
    # the artwork's main colours (a few at most), each ink pixel given to one
    pal = art.convert("RGB").quantize(colors=6, method=Image.Quantize.MEDIANCUT)
    idx = np.asarray(pal)
    colours = np.asarray(pal.getpalette()[:18], dtype=np.uint8).reshape(-1, 3)
    art_lab = rgb_to_lab(np.asarray(art.convert("RGB")))
    for k in range(len(colours)):
        part = ink & (idx == k)
        if part.sum() < PART_SHARE * ink.sum():
            continue
        core = ndimage.binary_erosion(part, iterations=2)
        if core.sum() < 0.4 * part.sum():  # the soft edge around the artwork's shapes, not a part of its own
            continue
        # a part shows when it stands out from what is around it once printed (other inks or the bare shirt):
        # black letters inside a white badge show on a black tee, a red 5 on the bare red shirt doesn't
        own = rgb_to_lab(colours[k][None, None, :])[0, 0]
        same = ink & (np.linalg.norm(art_lab - own, axis=-1) < 0.1)  # the same ink (its soft edge too)
        ring = ndimage.binary_dilation(part, iterations=pad) & ~same
        if not ring.any():
            continue
        around = np.median(b[ring], axis=0)
        if (np.linalg.norm(b[core] - around, axis=-1) > VISIBLE_DE).mean() < PART_SEEN_MIN:
            return _colour_word(colours[k])
    return ""
