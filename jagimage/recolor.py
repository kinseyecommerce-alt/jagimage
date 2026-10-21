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
PHOTO_BLACK = 0.2  # darkest garment lightness rendered (~#1A1A1A); below it knit and folds can't show
PHOTO_WHITE = 0.955  # lightest garment average rendered (~#F2F2F2); highlights roll up towards pure white
KNIT_MIN = 0.85    # stitches and knit keep at least this much of their contrast on darker targets
CLIPPED = 0.1      # lightness below this (RGB 0-3) is camera-crushed black with no texture in it
CROP_MARGIN = 32   # px around the garment kept when recolouring (edge band + blur reach)
CREASE_DROP = 0.15  # crease pixels are at least this much darker than the garment
CREASE_MAX = 5      # and in lines thinner than this (px), unlike hair
EDGE_SHARPEN_MAX = 3.0  # most an edge blend is steepened when the colour contrast to the surround grows
KNIT_CAP = 0.08    # knit/stitch contrast limit: real knit is far below it, clipped-black specks are not
FOLD_CAP = 0.2  # largest lightness change a fold/shadow may make after boosting
GROW_LIGHTER = 0.03  # shadow growth never takes pixels lighter than the garment's brightest lit fabric by more


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
    return _feather(_fill_holes(m, lab, centre), lab, centre)


def _fill_holes(m, lab, centre):
    """Close knit speckle, fill small holes (stitching) and larger ones that are just deep shadow of
    the same fabric. Never fill backdrop showing through, e.g. between the legs or arm and body."""
    studio = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
    closed = ndimage.binary_closing(m, structure=np.ones((5, 5), bool))
    # closing bridges narrow gaps too (the thin top of the gap between the legs): undo it there
    m = closed & ~(~m & (np.linalg.norm(lab - studio, axis=-1) < 0.04))
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    if hn:
        idx = np.arange(1, hn + 1)
        hs = ndimage.sum(holes, hl, index=idx)
        hm = np.stack([ndimage.mean(lab[..., c], hl, index=idx) for c in range(3)], axis=1)
        same = (np.abs(hm[:, 0] - centre[0]) < 0.25) & (np.hypot(hm[:, 1] - centre[1], hm[:, 2] - centre[2]) < 0.06)
        backdrop = np.linalg.norm(hm - studio, axis=1) < 0.04
        m = m | np.isin(hl, idx[((hs <= 0.002 * m.sum()) | same) & ~backdrop])
    return m


def _feather(m, lab, centre):
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
    # Clipped-black pixels (deep creases the camera crushed to 0,0,0) carry no fabric detail: leave
    # them out of the shading and fill them with the surrounding shading, a little darker, so they
    # don't turn into hard-edged blotches when the colour gets lighter.
    g = mask > 0.5
    clipped = ndimage.binary_dilation(L < CLIPPED, iterations=2) & g  # with the pixels it darkens
    m = (g & ~clipped).astype(np.float32)
    num = ndimage.gaussian_filter(Ld * m, 3.0)
    den = ndimage.gaussian_filter(m, 3.0)
    Lb = np.where(den > 1e-3, num / np.maximum(den, 1e-3), Ld)
    if clipped.any() and m.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~(m > 0), return_indices=True)
        fill = np.where(clipped, Lb[iy, ix], Lb)  # carry the surrounding shading inwards...
        gm = g.astype(np.float32)                 # ...then smooth it into a soft deep crease
        fill = ndimage.gaussian_filter(fill * gm, 6.0) / np.maximum(ndimage.gaussian_filter(gm, 6.0), 1e-3)
        knit = Ld - Lb
        # borrow knit texture by mirroring the real fabric next to the crushed area across its edge
        ys, xs = np.nonzero(clipped)
        ry = np.clip(2 * iy[ys, xs] - ys, 0, L.shape[0] - 1)
        rx = np.clip(2 * ix[ys, xs] - xs, 0, L.shape[1] - 1)
        borrowed = np.where(m[ry, rx] > 0, knit[ry, rx], 0.0)
        Lb = np.where(clipped, fill, Lb)
        Ld = Ld.copy()
        Ld[ys, xs] = Lb[ys, xs] + borrowed
    return Lb, Ld - Lb, np.where(clipped, 0.0, L - Ld)


def _with_creases(m, lab, centre):
    """Add the garment's thin dark crease lines along its edge (where a sleeve hem or an armpit fold
    meets the skin). They are too dark to carry the garment's colour, so the colour mask misses them
    and they'd be blended with the skin as speckle. Only thin lines count: a large dark area beside
    the garment (hair) fails the opening test and stays out."""
    dark = lab[..., 0] < centre[0] - CREASE_DROP
    thin = dark & ~ndimage.binary_opening(dark, structure=np.ones((CREASE_MAX, CREASE_MAX), bool))
    return m | (thin & ndimage.binary_dilation(m, iterations=3))


def _cleanup(m, lab, centre):
    """Keep the largest piece(s), close knit speckle, fill shadow holes, feather the edge."""
    lbl, n = ndimage.label(m)
    if n == 0:
        raise MaskError("Couldn't find the garment in this photo.")
    sizes = ndimage.sum(m, lbl, index=np.arange(1, n + 1))
    m = np.isin(lbl, np.flatnonzero(sizes >= 0.5 * sizes.max()) + 1)
    return _feather(_fill_holes(m, lab, centre), lab, centre)


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
    if m.any():  # shadows are darker than the lit fabric: a pale tinted halo or white shoe beside a deep base is not
        loose &= lab[..., 0] < np.percentile(lab[..., 0][m], 99) + GROW_LIGHTER
    m = ndimage.binary_propagation(m, mask=m | loose)
    centre = lab[m].mean(axis=0) if m.any() else base
    return _cleanup(_with_creases(m, lab, centre), lab, centre)


def render_target(target_hex: str) -> np.ndarray:
    """The OKLab colour a garment is rendered at: the hex itself, except that very dark and very light
    targets (#000000, #FFFFFF) are kept between PHOTO_BLACK and PHOTO_WHITE, the way black and white
    fabric photograph, so the knit, rib and stitches still have room to show."""
    T = hex_to_lab(target_hex)
    return np.array([float(np.clip(T[0], PHOTO_BLACK, PHOTO_WHITE)), T[1], T[2]])


def tint_carry(mean: np.ndarray, T: np.ndarray) -> float:
    """How much of the base photo's colour variation to keep: 0.35 for a target as colourful as the
    base, less for greyer targets (none for white, grey or black), so the base's green-to-grey
    shading doesn't turn into a pink cast on neutrals."""
    cb, ct = float(np.hypot(*mean[1:])), float(np.hypot(*T[1:]))
    return 0.35 * min(1.0, ct / max(cb, 1e-3))


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
        T = render_target(target_hex)
        g = texture_gain(float(mean[0]), float(T[0]))
        Lb, knit, noise = fabric_layers(L, mask)
        fold = g * (Lb - mean[0])  # cap deep creases so boosted shadows don't read as dirt
        L2 = T[0] + FOLD_CAP * np.tanh(fold / FOLD_CAP) + KNIT_CAP * np.tanh(max(g, KNIT_MIN) * texture * knit / KNIT_CAP) + noise * min(1.0, 1.0 / g)
        # Light targets: roll highlights off below 1 instead of clipping them flat, then
        # re-centre so the garment still averages the exact target lightness.
        for _ in range(2):
            over = L2 > T[0]
            head = max(1.0 - T[0], 1e-3)
            L2 = np.where(over, T[0] + head * np.tanh((L2 - T[0]) / head), L2)
            if mask is not None and (mask > 0.95).any():
                L2 = L2 + (T[0] - L2[mask > 0.95].mean())
    ab2 = T[1:] + (ab - mean[1:]) * (tint_carry(mean, T) if mode != "scale" else 0.0)
    return np.concatenate([np.clip(L2, 0, 1)[..., None], ab2], axis=-1)


def recolor_image(img: Image.Image, base_hex: str, target_hex: str, mode: str = "match",
                  mask: Optional[np.ndarray] = None, texture: float = 1.0) -> Tuple[Image.Image, float, np.ndarray]:
    """Returns (recoloured image, ΔE x100 of the garment vs target, soft mask used)."""
    rgba = np.array(img.convert("RGBA"))
    if mask is None:
        mask, _ = garment_mask(rgb_to_lab(rgba[..., :3]), base_hex)
    if not (mask > 0.95).any():
        raise MaskError("The supplied mask is empty.")
    # Work on the garment's box only (plus room for the edge and blurs): same result, much faster.
    ys, xs = np.nonzero(mask > 0.01)
    m = CROP_MARGIN
    y0, y1 = max(0, ys.min() - m), min(mask.shape[0], ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(mask.shape[1], xs.max() + m + 1)
    rgb, cm = rgba[y0:y1, x0:x1, :3], mask[y0:y1, x0:x1]
    lab = rgb_to_lab(rgb)
    core = cm > 0.95
    mean = lab[core].mean(axis=0)
    new = lab_to_rgb(recolor_lab(lab, mean, target_hex, mode, mask=cm, texture=texture)).astype(np.float32)
    gain = texture_gain(float(mean[0]), float(render_target(target_hex)[0])) if mode == "match" else 1.0
    out = _composite(rgb.astype(np.float32), new, cm, gain)
    goal = render_target(target_hex) if mode == "match" else hex_to_lab(target_hex)
    de = float(np.linalg.norm(rgb_to_lab(out)[core].mean(axis=0) - goal) * 100)
    rgba[y0:y1, x0:x1, :3] = out
    return Image.fromarray(rgba, "RGBA"), de, mask


def _composite(rgb, new, mask, gain=1.0, inside=2, outside=5):
    """Inside the garment use the recoloured pixels. Near the edge, unmix each pixel from the photo
    itself: alpha = how far it sits from the nearby background colour towards the nearby garment
    colour. The edge then keeps the photo's own anti-aliasing and dark edge creases instead of
    following the mask's pixel staircase, and the old colour leaves no outline."""
    m = mask > 0.5
    inner = ndimage.binary_erosion(m, iterations=inside)
    outer = ndimage.binary_dilation(m, iterations=outside)
    band = outer & ~inner
    out = np.where(m[..., None], new, rgb)
    if inner.any() and (~outer).any() and band.any():
        _, fi = ndimage.distance_transform_edt(~inner, return_indices=True)
        _, bi = ndimage.distance_transform_edt(outer, return_indices=True)
        ys, xs = np.nonzero(band)
        fy, fx, by, bx = fi[0][ys, xs], fi[1][ys, xs], bi[0][ys, xs], bi[1][ys, xs]
        P, Fo, Fn, Bc = rgb[ys, xs], rgb[fy, fx], new[fy, fx], rgb[by, bx]
        w = np.array([1.0, 2.0, 2.0])  # weight colour over lightness, so shadows on the backdrop stay out
        lp, lf, lb = rgb_to_lab(P) * w, rgb_to_lab(Fo) * w, rgb_to_lab(Bc) * w
        d = lf - lb
        dd = (d * d).sum(-1)
        a = ((lp - lb) * d).sum(-1) / np.maximum(dd, 1e-6)
        soft = np.clip(ndimage.gaussian_filter(m.astype(np.float32), 0.9), 0, 1)[ys, xs]
        a = np.clip(np.where(dd > 0.0016, a, soft), 0, 1)  # too alike to unmix: use the mask
        # A soft blend between garment and surround that were close in lightness (a mint collar on skin)
        # turns into a blur once the garment is much darker or lighter (navy on skin). Steepen it by
        # how much the lightness contrast grows, so the new edge is as crisp as the AI drew it.
        grow = np.abs(rgb_to_lab(Fn)[:, 0] - lb[:, 0]) / np.maximum(np.abs(lf[:, 0] - lb[:, 0]), 0.04)
        k = np.clip(grow, 1.0, EDGE_SHARPEN_MAX)
        a = np.clip((a - 0.5) * k + 0.5, 0, 1)[:, None]
        # The pure garment colour behind this pixel, recoloured: the neighbour's new colour with
        # this pixel's own lightness difference (scaled like the texture), so edge creases survive.
        fg = np.where(a > 0.05, Bc + (P - Bc) / np.maximum(a, 0.05), Fo)
        fgl, fol, fnl = rgb_to_lab(np.clip(fg, 0, 255)), rgb_to_lab(Fo), rgb_to_lab(Fn)
        nl = np.concatenate([np.clip(fnl[:, :1] + FOLD_CAP * np.tanh(gain * (fgl[:, :1] - fol[:, :1]) / FOLD_CAP), 0, 1), fnl[:, 1:]], -1)
        res = rgb_to_lab(np.clip(P + a * (lab_to_rgb(nl).astype(np.float32) - fg), 0, 255))
        # Spill: base colour left in a rim pixel beyond what alpha explains (e.g. a light mint halo
        # the AI drew around the shirt). Take that tint out along the base colour's direction.
        lp, lf, lb = lp / w, lf / w, lb / w
        dab = lf[:, 1:] - lb[:, 1:]
        u = dab / np.maximum(np.linalg.norm(dab, axis=1, keepdims=True), 1e-6)
        spill = ((lp[:, 1:] - lb[:, 1:] - a * dab) * u).sum(-1, keepdims=True)
        res[:, 1:] -= np.clip(spill, 0, None) * u * (a <= 0.05)  # larger alphas were already unmixed
        out[ys, xs] = lab_to_rgb(res).astype(np.float32)
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
