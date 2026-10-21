"""Exact-hex garment recolouring in OKLab (the same maths as the Jag Recolor web app).

Used to turn AI-generated base-colour photos into every palette colour: find the garment
by its known base colour, then shift it to the target while keeping folds and knit.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

W_L = 0.35  # weight of lightness vs colour when matching the garment (shadows stay in)
MODES = ("match", "hue", "scale")
WHITE_FROM = 0.93  # targets this light keep texture as shadow below the hex (white can't go brighter)
FOLD_CAP = 0.2  # largest lightness change a fold/shadow may make after boosting


class MaskError(RuntimeError):
    pass


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = _lin(rgb.astype(np.float32) / 255.0)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], axis=-1)
    return np.round(_gam(rgb) * 255).astype(np.uint8)


def hex_to_lab(hex_: str) -> np.ndarray:
    h = hex_.lstrip("#")
    return rgb_to_lab(np.array([[[int(h[i:i + 2], 16) for i in (0, 2, 4)]]], dtype=np.uint8))[0, 0]


def lightness(hex_: str) -> float:
    return float(hex_to_lab(hex_)[0])


def garment_mask(lab: np.ndarray, base_hex: str, tol: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Soft mask (0..1) of the garment drawn in `base_hex`, and the garment's mean OKLab colour.
    Picks pixels near the base colour, refines the centre from what was found, keeps the
    largest connected piece (so same-coloured hair or props stay out) and fills knit speckle."""
    w = np.array([W_L, 1, 1], np.float32)
    base = hex_to_lab(base_hex)
    if np.hypot(base[1], base[2]) >= 0.05:
        return _hue_mask(lab, base, base_hex)
    if tol is None:  # tighter for low-chroma bases so grey backgrounds stay out
        tol = float(np.clip(0.55 * np.hypot(base[1], base[2]) + 0.03, 0.04, 0.09))
    d = np.linalg.norm((lab - base) * w, axis=-1)
    cand = d < tol * 1.6
    if cand.sum() < 0.01 * cand.size:
        raise MaskError(f"Couldn't find the {base_hex} garment in this photo.")
    centre = np.median(lab[cand], axis=0)
    m = np.linalg.norm((lab - centre) * w, axis=-1) < tol
    lbl, n = ndimage.label(m)
    if n == 0:
        raise MaskError(f"Couldn't find the {base_hex} garment in this photo.")
    sizes = ndimage.sum(m, lbl, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= 0.5 * sizes.max()) + 1
    m = np.isin(lbl, keep)
    # grow into connected deep folds/shadows of the same fabric that the strict match missed
    loose = np.linalg.norm((lab - centre) * w, axis=-1) < tol * 2.2
    m = ndimage.binary_propagation(m, mask=m | loose)
    m = ndimage.binary_closing(m, structure=np.ones((5, 5), bool))
    # fill only small holes (knit speckle, stitching); keep real gaps like arm-to-body background
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    if hn:  # fill small holes, and larger ones that are just deep shadow of the same fabric
        idx = np.arange(1, hn + 1)
        hs = ndimage.sum(holes, hl, index=idx)
        hm = np.stack([ndimage.mean(lab[..., c], hl, index=idx) for c in range(3)], axis=1)
        same = (np.abs(hm[:, 0] - centre[0]) < 0.25) & (np.hypot(hm[:, 1] - centre[1], hm[:, 2] - centre[2]) < 0.06)
        m |= np.isin(hl, idx[(hs <= 0.002 * m.sum()) | same])
    soft = ndimage.gaussian_filter(m.astype(np.float32), 0.8)
    core = soft > 0.95
    return soft, lab[core].mean(axis=0) if core.any() else centre


def texture_gain(mean_L: float, target_L: float) -> float:
    """How much stronger the fabric's light-and-shadow must be at the new colour. Real fabric
    shows its folds and knit with more contrast in light dyes than in dark ones, so going from
    a black photo to yellow multiplies the texture (about 2.5x) and white to navy reduces it."""
    return float(np.clip(((target_L + 0.08) / (mean_L + 0.08)) ** 0.8, 0.35, 3.5))


def fabric_layers(L: np.ndarray, mask: Optional[np.ndarray]):
    """Split lightness into: folds/shading (soft, large scale), knit grain (fine, after removing
    1-pixel camera/JPEG noise) and that noise. Blurs only inside the garment so the background
    doesn't bleed in at the edges."""
    Ld = ndimage.median_filter(L, size=3)
    if mask is None:
        return Ld, Ld, L - Ld
    m = (mask > 0.5).astype(np.float32)
    num = ndimage.gaussian_filter(Ld * m, 3.0)
    den = ndimage.gaussian_filter(m, 3.0)
    Lb = np.where(den > 1e-3, num / np.maximum(den, 1e-3), Ld)
    return Lb, Ld - Lb, L - Ld


def _cleanup(m, lab, centre):
    """Keep the largest piece(s), close knit speckle, fill shadow holes, feather the edge."""
    lbl, n = ndimage.label(m)
    if n == 0:
        raise MaskError("Couldn't find the garment in this photo.")
    sizes = ndimage.sum(m, lbl, index=np.arange(1, n + 1))
    m = np.isin(lbl, np.flatnonzero(sizes >= 0.5 * sizes.max()) + 1)
    m = ndimage.binary_closing(m, structure=np.ones((5, 5), bool))
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    if hn:
        idx = np.arange(1, hn + 1)
        hs = ndimage.sum(holes, hl, index=idx)
        hm = np.stack([ndimage.mean(lab[..., c], hl, index=idx) for c in range(3)], axis=1)
        same = (np.abs(hm[:, 0] - centre[0]) < 0.25) & (np.hypot(hm[:, 1] - centre[1], hm[:, 2] - centre[2]) < 0.06)
        m |= np.isin(hl, idx[(hs <= 0.002 * m.sum()) | same])
    soft = ndimage.gaussian_filter(m.astype(np.float32), 0.8)
    core = soft > 0.95
    return soft, lab[core].mean(axis=0) if core.any() else centre


def _hue_mask(lab, base, base_hex):
    """Coloured bases (e.g. mint or bottle green): find the garment by hue, which grey studio,
    skin, black hair and white shoes never share, even when the AI made it paler or deeper."""
    C = np.hypot(lab[..., 1], lab[..., 2])
    H = np.arctan2(lab[..., 2], lab[..., 1])
    hb = np.arctan2(base[2], base[1])
    dh = np.abs(np.angle(np.exp(1j * (H - hb))))
    cand = (dh < np.radians(40)) & (C > 0.025)
    if cand.sum() < 0.01 * cand.size:
        raise MaskError(f"Couldn't find the {base_hex} garment in this photo.")
    hc = np.angle(np.mean(np.exp(1j * H[cand])))      # the hue the AI actually produced
    cc = float(np.median(C[cand]))
    dh = np.abs(np.angle(np.exp(1j * (H - hc))))
    m = (dh < np.radians(28)) & (C > max(0.02, 0.35 * cc))
    # shadows lose chroma but keep hue: grow into connected pixels with a looser chroma floor
    loose = (dh < np.radians(40)) & (C > max(0.012, 0.18 * cc))
    lbl, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lbl, index=np.arange(1, n + 1))
        m = np.isin(lbl, np.flatnonzero(sizes >= 0.5 * sizes.max()) + 1)
    m = ndimage.binary_propagation(m, mask=m | loose)
    centre = lab[m].mean(axis=0) if m.any() else base
    return _cleanup(m, lab, centre)


def recolor_lab(lab: np.ndarray, mean: np.ndarray, target_hex: str, mode: str = "match",
                mask: Optional[np.ndarray] = None, texture: float = 1.0) -> np.ndarray:
    """Recolour in OKLab. The garment's own colour is discarded; only its texture is kept.
    match: the fabric's folds, shading and knit (scaled for the new lightness) around the exact hex.
    hue:   keep the photo's lightness as it is; only hue and colour strength change.
    scale: multiply lightness by target/original (older behaviour, can look grainy)."""
    T = hex_to_lab(target_hex)
    L, ab = lab[..., 0], lab[..., 1:]
    if mode == "hue":
        L2 = L
    elif mode == "scale":
        f = T[0] / mean[0] if mean[0] > 0.02 else 1.0
        L2 = L * f
    else:
        g = texture_gain(float(mean[0]), float(T[0]))
        Lb, knit, noise = fabric_layers(L, mask)
        fold = g * (Lb - mean[0])  # cap deep creases so boosted shadows don't read as dirt
        L2 = T[0] + FOLD_CAP * np.tanh(fold / FOLD_CAP) + g * texture * knit + noise * min(1.0, 1.0 / g)
        if T[0] > WHITE_FROM:  # near-white: highlights stay on the hex, folds show as shadow only
            return np.concatenate([np.clip(T[0] + np.minimum(L2 - T[0], 0), 0, 1)[..., None],
                                   T[1:] + (ab - mean[1:]) * 0.35], axis=-1)
        # Light targets: roll highlights off below 1 instead of clipping them flat, then
        # re-centre so the garment still averages the exact target lightness.
        for _ in range(2):
            over = L2 > T[0]
            head = max(1.0 - T[0], 1e-3)
            L2 = np.where(over, T[0] + head * np.tanh((L2 - T[0]) / head), L2)
            if mask is not None and (mask > 0.95).any():
                L2 = L2 + (T[0] - L2[mask > 0.95].mean())
    ab2 = T[1:] + (ab - mean[1:]) * (0.35 if mode != "scale" else 0.0)
    return np.concatenate([np.clip(L2, 0, 1)[..., None], ab2], axis=-1)


def recolor_image(img: Image.Image, base_hex: str, target_hex: str, mode: str = "match",
                  mask: Optional[np.ndarray] = None, texture: float = 1.0) -> Tuple[Image.Image, float, np.ndarray]:
    """Returns (recoloured image, ΔE x100 of the garment vs target, soft mask used)."""
    rgba = np.asarray(img.convert("RGBA"))
    rgb = rgba[..., :3]
    lab = rgb_to_lab(rgb)
    if mask is None:
        mask, mean = garment_mask(lab, base_hex)
    else:
        core = mask > 0.95
        if not core.any():
            raise MaskError("The supplied mask is empty.")
        mean = lab[core].mean(axis=0)
    new = lab_to_rgb(recolor_lab(lab, mean, target_hex, mode, mask=mask, texture=texture)).astype(np.float32)
    out = _composite(rgb.astype(np.float32), new, mask)
    core = mask > 0.95
    de = float(np.linalg.norm(rgb_to_lab(out)[core].mean(axis=0) - hex_to_lab(target_hex)) * 100)
    return Image.fromarray(np.dstack([out, rgba[..., 3]]), "RGBA"), de, mask


def _composite(rgb, new, mask):
    """Inside the garment use the recoloured pixels. In a thin edge band, rebuild each pixel
    as alpha * (nearest recoloured garment colour) + (1 - alpha) * (nearest clean surround),
    so the old garment colour doesn't leave a dark or coloured outline."""
    m = mask > 0.5
    inner = ndimage.binary_erosion(m, iterations=1)
    outer = ndimage.binary_dilation(m, iterations=3)
    band = outer & ~inner
    a = np.clip(ndimage.gaussian_filter(m.astype(np.float32), 0.9), 0, 1)[..., None]
    out = np.where(m[..., None], new, rgb)
    if inner.any() and (~outer).any() and band.any():
        _, fi = ndimage.distance_transform_edt(~inner, return_indices=True)
        _, bi = ndimage.distance_transform_edt(ndimage.binary_dilation(outer, iterations=1), return_indices=True)
        F = new[fi[0], fi[1]]
        B = rgb[bi[0], bi[1]]
        out = np.where(band[..., None], a * F + (1 - a) * B, out)
    return np.round(np.clip(out, 0, 255)).astype(np.uint8)


def fit_canvas(img: Image.Image, width: int = 1500, height: int = 1500) -> Image.Image:
    """Marketplace canvas (e.g. 1500x1500): scale the photo to fit, then extend the studio
    background sideways (or up/down) row by row from the photo's own edge colour, so the
    seamless backdrop continues instead of showing bars."""
    img = img.convert("RGB")
    s = min(width / img.width, height / img.height)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    im = np.asarray(img.resize((w, h), Image.LANCZOS)).astype(np.float32)
    out = np.zeros((height, width, 3), np.float32)
    x0, y0 = (width - w) // 2, (height - h) // 2
    if w < width:  # pad left/right with each row's edge colour
        k = max(2, w // 60)
        left = np.median(im[:, :k], axis=1)[:, None]
        right = np.median(im[:, -k:], axis=1)[:, None]
        out[y0:y0 + h, :x0] = ndimage.gaussian_filter1d(left, 3, axis=0)
        out[y0:y0 + h, x0 + w:] = ndimage.gaussian_filter1d(right, 3, axis=0)
    if h < height:
        k = max(2, h // 60)
        out[:y0, :] = np.median(im[:k], axis=0).mean(axis=0)
        out[y0 + h:, :] = np.median(im[-k:], axis=0).mean(axis=0)
    out[y0:y0 + h, x0:x0 + w] = im
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def save_mask(path: Path, mask: np.ndarray):
    Image.fromarray(np.round(mask * 255).astype(np.uint8), "L").save(path)


def write_qc(path: Path, rows: dict):
    path.write_text(json.dumps(rows, indent=2))
