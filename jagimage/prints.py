"""Puts a print (the shop's artwork, a PNG) on the chest of a finished T-shirt photo, exactly as drawn:
no AI, so every photo carries the same design. The artwork takes on the shirt's light and shadow, bends a
little with its folds and shows the knit through the ink, and anything in front of the shirt (an arm, a
hand, hair) stays in front of the print."""

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


def artwork(img: Image.Image) -> Image.Image:
    """The print as RGBA. A PNG with transparency is used as it is. A design on a plain white (or black, or
    any single-colour) background has that background colour taken out, as a screen printer would: it
    means "no ink". Soft edges keep their shape and lose the background's tint."""
    img = img.convert("RGBA")
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


def place_print(photo: Image.Image, mask: np.ndarray, art: Image.Image, place: str = "chest_large") -> Image.Image:
    """The photo with the artwork printed on the shirt's chest. mask: the shirt (0..1, same size)."""
    if place not in PLACES:
        raise PrintError(f"place must be one of {', '.join(PLACES)}")
    share, below, offset = PLACES[place]
    rgb = np.asarray(photo.convert("RGB")).astype(np.float32) / 255
    H, W = rgb.shape[:2]
    cx, neck, cw = chest(mask)
    art = artwork(art)
    bbox = art.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not bbox:
        raise PrintError("the print is empty (fully transparent)")
    art = art.crop(bbox)  # the design itself, without the empty canvas around it
    pw = share * cw
    ph = pw * art.height / art.width
    if ph > MAX_TALL * cw:
        ph = MAX_TALL * cw
        pw = ph * art.width / art.height
    pw, ph = max(2, int(round(pw))), max(2, int(round(ph)))
    x0 = int(round(cx + offset * cw - pw / 2))
    y0 = int(round(neck + below * cw))
    if x0 < 0 or y0 < 0 or x0 + pw > W or y0 + ph > H:
        raise PrintError("the print doesn't fit on this photo")
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
