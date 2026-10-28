"""Puts a print (the shop's artwork, a PNG) on the chest of a finished T-shirt photo, exactly as drawn:
no AI, so every photo carries the same design. The artwork takes on the shirt's light and shadow, bends a
little with its folds and shows the knit through the ink, and anything in front of the shirt (an arm, a
hand, hair) stays in front of the print."""

import math
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

# where the print goes: width as a share of the chest, top below the neckline (share of the chest width),
# and the centre's offset from the chest's middle (share of the chest width; + is the viewer's right)
PLACES: Dict[str, Tuple[float, float, float]] = {
    "chest_large": (0.62, 0.12, 0.0),
    "chest_medium": (0.42, 0.14, 0.0),
    "left_chest": (0.17, 0.16, 0.22),  # the wearer's left chest = the viewer's right in a front photo
}
PLACE_NAMES = {"chest_large": "Centre chest, large", "chest_medium": "Centre chest, medium", "left_chest": "Left chest, small logo"}
MAX_TALL = 0.8     # a tall print stops at this share of the chest width in height, so it stays above the hem
INK = 0.97         # screen-print ink hides almost all of the fabric colour under it
KNIT = 0.6         # how much of the fabric's knit texture shows through the ink
FOLD_SHIFT = 2.5   # px: how far the artwork bends with the folds at most


class PrintError(ValueError):
    pass


MAX_PIXELS = 36_000_000  # a print or guide picture larger than this (e.g. 6000 x 6000) is refused before it is decoded
ART_SIDE = 3000          # bigger artwork is scaled down to this first: a chest print never needs more


def open_image(src) -> Image.Image:
    """An uploaded or downloaded picture, refused before decoding when it is huge (a small file can
    hold a 13000 x 13000 picture that takes many GB of memory to decode)."""
    import io

    try:
        im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    except Exception as exc:
        raise PrintError(f"that picture can't be read ({type(exc).__name__})") from exc
    if im.width * im.height > MAX_PIXELS:
        raise PrintError(f"that picture is too large ({im.width} x {im.height}); keep it under {MAX_PIXELS // 1_000_000} megapixels")
    try:
        im.load()
    except Exception as exc:
        raise PrintError(f"that picture can't be read ({type(exc).__name__})") from exc
    return im


def artwork(img: Image.Image) -> Image.Image:
    """The print as RGBA. A PNG with transparency is used as it is. A design on a plain white (or black, or
    any single-colour) background has that background colour taken out, as a screen printer would: it
    means "no ink". Soft edges keep their shape and lose the background's tint."""
    img = img.convert("RGBA")
    if max(img.size) > ART_SIDE:
        img.thumbnail((ART_SIDE, ART_SIDE), Image.LANCZOS)
    a = np.asarray(img).astype(np.float32) / 255
    if (a[..., 3] < 0.98).mean() > 0.01:
        return img
    rgb = a[..., :3]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    if np.percentile(np.abs(border - bg).max(axis=1), 90) > 0.08:
        return img  # no plain background (a photo-style print): printed as a whole rectangle
    d = np.abs(rgb - bg).max(axis=-1)
    alpha = np.clip((d - 0.05) / 0.10, 0, 1)
    # the edge pixels mixed with the background get their own colour back
    safe = np.maximum(alpha, 1e-3)[..., None]
    colour = np.clip((rgb - bg * (1 - alpha[..., None])) / safe, 0, 1)
    out = np.concatenate([colour, alpha[..., None]], -1)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


def chest(mask: np.ndarray):
    """The chest of the shirt in a front photo: (centre x, neckline y, chest width) in pixels."""
    m = mask > 0.5
    ys, xs = np.nonzero(m)
    if len(ys) < 1000:
        raise PrintError("no shirt found in this photo")
    top, bottom = ys.min(), ys.max()
    h = bottom - top
    cx = int(np.median(xs))
    widths, centres = [], []
    for y in range(int(top + 0.35 * h), int(top + 0.65 * h), max(1, h // 60)):
        row = m[y]
        if not row[cx]:
            continue
        left = cx - np.argmin(row[cx::-1]) + 1 if not row[:cx + 1].all() else 0
        right = cx + np.argmin(row[cx:]) - 1 if not row[cx:].all() else len(row) - 1
        widths.append(right - left)
        centres.append((left + right) / 2)
    if not widths:
        raise PrintError("the shirt's chest isn't clear in this photo")
    # the narrowest rows are the torso alone (sleeves or arms touching it make some rows wider)
    w = float(np.percentile(widths, 25))
    cx = float(np.median([c for c, wd in zip(centres, widths) if wd <= w * 1.15]))
    band = m[:, max(0, int(cx) - 3):int(cx) + 4].any(axis=1)
    neck = int(np.argmax(band))
    return cx, neck, w


LIMITS = {"dx": (-0.3, 0.3), "dy": (-0.15, 0.4), "scale": (0.5, 1.6)}  # how far a placement may be adjusted


def adjust(dx=0.0, dy=0.0, scale=1.0) -> dict:
    """A placement adjustment, kept within LIMITS: dx, dy move the print by a share of the chest width
    (+ is right and down in the photo); scale makes it bigger or smaller."""
    def clamp(v, k):
        try:
            v = float(v if v is not None else (1 if k == "scale" else 0))
        except (TypeError, ValueError):
            v = 1.0 if k == "scale" else 0.0
        if not math.isfinite(v):  # NaN would slip through min/max
            v = 1.0 if k == "scale" else 0.0
        return float(min(max(v, LIMITS[k][0]), LIMITS[k][1]))
    return {"dx": round(clamp(dx, "dx"), 3), "dy": round(clamp(dy, "dy"), 3), "scale": round(clamp(scale, "scale"), 3)}


def _layout(mask: np.ndarray, art: Image.Image, place: str, dx=0.0, dy=0.0, scale=1.0) -> dict:
    """Where the print goes on this shirt: the chest (cx, neck, cw) and the print's box (x0, y0, pw, ph)."""
    if place not in PLACES:
        raise PrintError(f"place must be one of {', '.join(PLACES)}")
    share, below, offset = PLACES[place]
    a_ = adjust(dx, dy, scale)
    share, below, offset = share * a_["scale"], below + a_["dy"], offset + a_["dx"]
    H, W = mask.shape[:2]
    cx, neck, cw = chest(mask)
    art = artwork(art)
    bbox = art.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not bbox:
        raise PrintError("the print is empty (fully transparent)")
    art = art.crop(bbox)  # the design itself, without the empty canvas around it
    pw = share * cw
    ph = pw * art.height / art.width
    if ph > MAX_TALL * a_["scale"] * cw:
        ph = MAX_TALL * a_["scale"] * cw
        pw = ph * art.width / art.height
    pw, ph = max(2, int(round(pw))), max(2, int(round(ph)))
    x0 = int(round(cx + offset * cw - pw / 2))
    y0 = int(round(neck + below * cw))
    if x0 < 0 or y0 < 0 or x0 + pw > W or y0 + ph > H:
        raise PrintError("the print doesn't fit on this photo")
    return {"cx": cx, "neck": neck, "cw": cw, "art": art, "x0": x0, "y0": y0, "pw": pw, "ph": ph}


# the firm rules every chest print is held to (shares of the chest width); the AI checker judges the rest
MIN_COLLAR_GAP = 0.07   # the print starts at least this far below the top of the neckband (it never touches the collar)
MIN_HEM_GAP = 0.12      # ... and ends at least this far above the hem
MAX_WIDTH = 0.78        # ... and is at most this wide
MAX_OFF_CENTRE = 0.05   # a centred print may sit at most this far off the chest's middle
MAX_HIDDEN = 0.08       # at most this share of the print may be hidden by an arm or hand


def measure(mask: np.ndarray, art: Image.Image, place: str = "chest_large", dx=0.0, dy=0.0, scale=1.0) -> dict:
    """The print's placement in numbers, as shares of the chest width: gap below the collar, gap above the
    hem, width, how far off-centre, and the share of it hidden by an arm or hand."""
    g = _layout(mask, art, place, dx, dy, scale)
    m = mask > 0.5
    col = m[:, max(0, int(g["cx"]) - 3):int(g["cx"]) + 4].any(axis=1)
    hem = len(col) - 1 - int(np.argmax(col[::-1]))
    alpha = np.asarray(g["art"].resize((g["pw"], g["ph"])).getchannel("A")) > 128
    under = m[g["y0"]:g["y0"] + g["ph"], g["x0"]:g["x0"] + g["pw"]]
    cw = g["cw"]
    return {"collar_gap": round((g["y0"] - g["neck"]) / cw, 3), "hem_gap": round((hem - g["y0"] - g["ph"]) / cw, 3),
            "width": round(g["pw"] / cw, 3), "off_centre": round((g["x0"] + g["pw"] / 2 - g["cx"]) / cw - PLACES[place][2], 3),
            "hidden": round(float((alpha & ~under).sum()) / max(1, int(alpha.sum())), 3)}


def rule_check(info: dict, place: str = "chest_large"):
    """The firm rules on a measured placement: (issues, fix) where fix is a move/resize like adjust()."""
    issues, fix = [], {"dx": 0.0, "dy": 0.0, "scale": 1.0}
    if info["collar_gap"] < MIN_COLLAR_GAP:
        issues.append("too high: touches the collar")
        fix["dy"] = round(PLACES[place][1] - info["collar_gap"], 3)
    elif info["hem_gap"] < MIN_HEM_GAP:
        issues.append("too low: runs towards the hem")
        fix["dy"] = round(info["hem_gap"] - MIN_HEM_GAP - 0.02, 3)
    if info["width"] > MAX_WIDTH:
        issues.append("too big for the chest")
        fix["scale"] = round(PLACES[place][0] / info["width"], 3)
    if abs(info["off_centre"]) > MAX_OFF_CENTRE:
        issues.append("off-centre")
        fix["dx"] = round(-info["off_centre"], 3)
    if info["hidden"] > MAX_HIDDEN:
        issues.append(f"partly hidden by an arm or hand ({round(info['hidden'] * 100)}%)")
    return issues, fix


def place_print(photo: Image.Image, mask: np.ndarray, art: Image.Image, place: str = "chest_large",
                dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> Image.Image:
    """The photo with the artwork printed on the shirt's chest. mask: the shirt (0..1, same size).
    dx, dy, scale: an adjustment of the place (see adjust())."""
    g = _layout(mask, art, place, dx, dy, scale)
    cx, neck, cw, art, x0, y0, pw, ph = (g[k] for k in ("cx", "neck", "cw", "art", "x0", "y0", "pw", "ph"))
    rgb = np.asarray(photo.convert("RGB")).astype(np.float32) / 255
    H, W = rgb.shape[:2]
    a = np.asarray(art.resize((pw, ph), Image.LANCZOS)).astype(np.float32) / 255
    # the region of the photo the print covers, with a margin for the fold shift
    pad = int(FOLD_SHIFT) + 2
    ya, yb, xa, xb = max(0, y0 - pad), min(H, y0 + ph + pad), max(0, x0 - pad), min(W, x0 + pw + pad)
    region = rgb[ya:yb, xa:xb]
    shirt = np.clip(mask[ya:yb, xa:xb].astype(np.float32), 0, 1)
    L = region @ np.array([0.299, 0.587, 0.114], np.float32)
    # folds and light: the fabric's large-scale shading, relative to its typical brightness here
    big = max(2.0, cw / 70)
    wsum = ndimage.gaussian_filter(shirt, big) + 1e-4
    shade = ndimage.gaussian_filter(L * shirt, big) / wsum
    ref = float(np.median(shade[shirt > 0.5])) if (shirt > 0.5).any() else float(shade.mean())
    gain = np.clip(shade / max(ref, 1e-3), 0.55, 1.3)
    knit = (L - ndimage.gaussian_filter(L, 1.2)) * KNIT
    # bend the artwork a little along the folds (it follows the slope of the shading)
    gy, gx = np.gradient(ndimage.gaussian_filter(shade, big * 1.5))
    scale = FOLD_SHIFT / max(1e-4, float(np.percentile(np.hypot(gx, gy), 99)))
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    sy, sx = yy - y0 + gy * scale, xx - x0 + gx * scale
    warped = np.stack([ndimage.map_coordinates(a[..., c], [sy, sx], order=1, mode="constant", cval=0) for c in range(4)], -1)
    alpha = warped[..., 3] * INK * shirt  # an arm or hand in front of the shirt hides the print there
    ink = np.clip(warped[..., :3] * gain[..., None] + knit[..., None], 0, 1)
    ink = ndimage.gaussian_filter(ink, (0.5, 0.5, 0))  # printed ink is as soft as the photo, not razor sharp
    out = rgb.copy()
    out[ya:yb, xa:xb] = region * (1 - alpha[..., None]) + ink * alpha[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
