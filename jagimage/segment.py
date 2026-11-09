"""The shirt in any photo (a model photo, a flat lay, a hanger shot): a 0..1 mask of the garment, found
from a hint the vision AI gives (the shirt's box and a point on its chest) and the photo's own edges.

A watershed on the colour gradient grows two seeds: the shirt (a patch around the chest point, plus the
pixels in the box that share the chest's colour) and everything else (the photo's border, all outside the
box, and pixels in the box far from the shirt's colour, such as skin and hair). The shirt is where its
seed wins, cleaned to one solid piece."""

from typing import Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from .recolor import rgb_to_lab

WORK_SIDE = 900        # the mask is found at this size, then scaled up to the photo
SEED_RADIUS = 0.035    # the chest seed: a disc this share of the box width
NEAR = 0.06            # OKLab distance: a pixel this close to the chest colour seeds the shirt
FAR = 0.16             # ...and one this far from it (inside the box) seeds the background


class SegmentError(ValueError):
    pass


def _clamp_box(box: Sequence[float]):
    try:
        x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in box)
    except (TypeError, ValueError):
        raise SegmentError("no shirt box")
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        raise SegmentError("the shirt box is too small")
    return x0, y0, x1, y1


def shirt_mask(photo: Image.Image, box: Sequence[float], chest: Sequence[float]) -> np.ndarray:
    """The shirt as a float mask (photo size). box: (x0, y0, x1, y1) and chest: (x, y), shares of the photo's
    width and height, as the vision AI saw them."""
    W, H = photo.size
    s = min(1.0, WORK_SIDE / max(W, H))
    small = photo.convert("RGB").resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
    lab = rgb_to_lab(np.asarray(small))
    h, w = lab.shape[:2]
    x0, y0, x1, y1 = _clamp_box(box)
    bx0, by0, bx1, by1 = int(x0 * w), int(y0 * h), int(np.ceil(x1 * w)), int(np.ceil(y1 * h))
    try:
        cx, cy = int(min(max(float(chest[0]), 0), 1) * (w - 1)), int(min(max(float(chest[1]), 0), 1) * (h - 1))
    except (TypeError, ValueError, IndexError):
        cx, cy = (bx0 + bx1) // 2, int(by0 + 0.35 * (by1 - by0))
    if not (bx0 <= cx < bx1 and by0 <= cy < by1):
        cx, cy = (bx0 + bx1) // 2, int(by0 + 0.35 * (by1 - by0))
    r = max(2, int(SEED_RADIUS * (bx1 - bx0)))
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    colour = np.median(lab[disc], axis=0)
    diff = lab - colour
    diff[..., 0] *= 0.35  # a fold's shadow is darker, not another colour
    d = np.linalg.norm(diff, axis=-1)
    inbox = np.zeros((h, w), bool)
    inbox[by0:by1, bx0:bx1] = True
    markers = np.zeros((h, w), np.int16)
    out = ~ndimage.binary_dilation(inbox, iterations=max(2, int(0.02 * max(h, w))))
    out[[0, -1], :] = True
    out[:, [0, -1]] = True
    markers[out] = 1
    markers[inbox & (d > FAR)] = 1
    core = disc | ndimage.binary_erosion(inbox & (d < NEAR), iterations=2)
    markers[core] = 2
    # the colour gradient (lightness counts less: folds and shadows are not the shirt's edge)
    g = np.zeros((h, w), np.float32)
    for c, wt in ((0, 0.6), (1, 1.0), (2, 1.0)):
        ch = ndimage.gaussian_filter(lab[..., c], 1.0)
        g += wt * np.hypot(ndimage.sobel(ch, 0), ndimage.sobel(ch, 1))
    g = (np.clip(g / max(1e-6, float(np.percentile(g, 99))), 0, 1) * 65535).astype(np.uint16)
    lab_ws = ndimage.watershed_ift(g, markers)
    m = lab_ws == 2
    m = ndimage.binary_opening(m, iterations=1)
    labels, n = ndimage.label(m)
    if not n:
        raise SegmentError("no shirt found in this photo")
    keep = labels == labels[cy, cx] if labels[cy, cx] else labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    keep = ndimage.binary_fill_holes(keep)
    if keep.mean() < 0.01:
        raise SegmentError("the shirt found is too small")
    soft = ndimage.gaussian_filter(keep.astype(np.float32), 0.8)
    big = Image.fromarray((soft * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return np.asarray(big).astype(np.float32) / 255
