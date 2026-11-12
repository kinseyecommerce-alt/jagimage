"""Marketplace downloads: every finished photo resized for one marketplace's listing rules, sharpened
so a buyer's zoom stays crisp, and on pure white where the marketplace wants a white main image.

Sharpening is a plain unsharp mask after a Lanczos resize. In our zoom test (2026-10) it kept the knit,
the stitching and natural skin, where an AI upscaler (Real-ESRGAN) made skin waxy and wiped out the knit.
Sizes are the highest figures in 2026 seller guides; check them against each seller portal."""

import io
import threading
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from .recolor import backdrop_mask, fit_canvas, rgb_to_lab

# key: (label, (width, height), pure white background)
MARKETS = {
    "myntra": ("Myntra", (1500, 2000), False),
    "ajio": ("Ajio", (1500, 2000), False),
    "amazon": ("Amazon", (2000, 2000), True),
    "flipkart": ("Flipkart", (1500, 1500), True),
    "meesho": ("Meesho", (1080, 1440), False),
}
WHITE_FROM = 0.92  # backdrop at least this share of its lit brightness is pure white
PERSON_IN = 0.5   # cutout: at least this sure it is the person, so never whitened
PERSON_OUT = 0.08  # ...and this unsure, so a closed-in gap of wall-grey is backdrop
GAP_DL, GAP_DC = 0.08, 0.03  # OKLab: how close a closed-in gap is to the wall's colour
WHITE_EDGE = 1.5  # px: the backdrop's edge is softened this much when it turns white, so hair stays natural


_CUTOUT = {}
CUTOUT_MODEL = "isnet-general-use"  # sharp edges on hair and hands; downloaded once (about 180 MB)
_CUTOUT_LOCK = threading.Lock()


def person_mask(img: Image.Image) -> Optional[np.ndarray]:
    """Where the model (person and clothes) is, 0..1, from a cutout model; None when it isn't installed.
    The backdrop's colour alone can't tell a pale melange or white garment from a light grey studio."""
    try:
        from rembg import new_session, remove
    except ImportError:
        return None
    with _CUTOUT_LOCK:
        if "session" not in _CUTOUT:
            try:
                _CUTOUT["session"] = new_session(CUTOUT_MODEL)
            except Exception:
                _CUTOUT["session"] = None
        session = _CUTOUT["session"]
        if session is None:
            return None
        small = img.convert("RGB")
        if max(small.size) > 1024:  # the model looks at about 1024 px anyway
            small = small.copy()
            small.thumbnail((1024, 1024))
        m = remove(small, session=session, only_mask=True, post_process_mask=False)
    m = m.convert("L").resize(img.size, Image.BILINEAR)
    return np.asarray(m, np.float32) / 255


def on_white(img: Image.Image, person: Optional[np.ndarray] = None) -> Image.Image:
    """The studio backdrop made pure white (#FFFFFF). Darker spots in it (the contact shadow) keep
    their share of darkness, so the model still stands on the floor instead of floating. Where a cutout
    model is installed, nothing it sees as the person or the clothes is touched."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    lab = rgb_to_lab(rgb.astype(np.uint8))
    if person is None:
        person = person_mask(img)
    # the studio can have two parts: the wall and a floor strip of another grey (or tint) along the bottom.
    # Each border's own colour is followed, and each part turns white against its own brightness.
    bg = np.zeros(lab.shape[:2], bool)
    shade = np.ones(lab.shape[:2], np.float32)
    done = []
    for edge in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        studio = np.median(edge, axis=0)
        if any(np.linalg.norm(studio - d) < 0.02 for d in done):  # the same wall seen from another side
            continue
        done.append(studio)
        part = backdrop_mask(lab, studio) & ~bg
        if part.mean() < 0.005:
            continue
        lit = float(np.percentile(lab[..., 0][part], 75))
        # the backdrop's own soft shading turns pure white (#FFFFFF); only a real shadow stays grey
        shade[part] = (np.clip(lab[..., 0][part] / (WHITE_FROM * max(lit, 1e-3)), 0, 1) ** 2)
        bg |= part
    if bg.mean() < 0.05:  # no backdrop found: leave the photo as it is
        return img.convert("RGB")
    if person is not None:
        bg &= person < PERSON_IN  # a pale garment the backdrop's colour ran into stays as it is
        # backdrop the border can't reach (between the legs, between an arm and the body): not the person,
        # and the wall's own grey
        wall = np.median(lab[bg], axis=0)
        lit = float(np.percentile(lab[..., 0][bg], 75))
        gap = (person < PERSON_OUT) & ~bg & (np.abs(lab[..., 0] - wall[0]) < GAP_DL) & \
              (np.hypot(lab[..., 1] - wall[1], lab[..., 2] - wall[2]) < GAP_DC)
        shade[gap] = np.clip(lab[..., 0][gap] / (WHITE_FROM * max(lit, 1e-3)), 0, 1) ** 2
        bg |= gap
    white = 255.0 * shade[..., None]
    grown = ndimage.binary_dilation(bg, iterations=1)  # takes the 1 px grey rim around hair too
    m = np.clip(ndimage.gaussian_filter(grown.astype(np.float32), WHITE_EDGE), 0, 1)[..., None]
    out = rgb * (1 - m) + white * m
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8))


def sharpen(img: Image.Image, scale: float) -> Image.Image:
    """Unsharp mask matched to how much the photo was enlarged (as in the zoom test at 2x), with halo control:
    no pixel goes lighter or darker than its 3x3 neighbourhood was, so a dark garment on the light backdrop
    gets no white outline (and skin beside it no dark one)."""
    img = img.convert("RGB")
    if scale > 1.02:
        sharp = img.filter(ImageFilter.UnsharpMask(radius=min(2.0, scale), percent=60 + int(20 * min(1.0, scale - 1)), threshold=2))
    else:
        sharp = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=40, threshold=2))  # downsized: a light touch
    lo, hi = img.filter(ImageFilter.MinFilter(3)), img.filter(ImageFilter.MaxFilter(3))
    a = np.asarray(sharp, np.int16)
    return Image.fromarray(np.clip(a, np.asarray(lo, np.int16), np.asarray(hi, np.int16)).astype(np.uint8))


def for_market(photo: Image.Image, market: str) -> Image.Image:
    """One finished photo, ready for `market`."""
    _, (w, h), white = MARKETS[market]
    photo = photo.convert("RGB")
    scale = min(w / photo.width, h / photo.height)
    out = sharpen(fit_canvas(photo, w, h), scale)
    return on_white(out) if white else out  # white after sharpening, so the white stays flat


def source_photo(batch, row: dict) -> Image.Image:
    """The full photo for a results row: the full-size PNG when it was kept, else the photo cut back out
    of the marketplace JPG (whose sides or top are only extended backdrop)."""
    if row.get("png"):
        return Image.open(batch.root / row["png"])
    jpg = Image.open(batch.root / row["jpg"]).convert("RGB")
    bases = sorted((batch.root / row["model"] / "_base").glob(f"*/{row['pose']}.png"))
    if not bases:
        return jpg
    with Image.open(bases[0]) as b:
        bw, bh = b.size
    s = min(jpg.width / bw, jpg.height / bh)
    w, h = round(bw * s), round(bh * s)
    x0, y0 = (jpg.width - w) // 2, (jpg.height - h) // 2
    return jpg.crop((x0, y0, x0 + w, y0 + h))


def label(row: dict) -> str:
    return f"{row['model']}_{row['top']}" + (f"__{row['bottom']}" if row["bottom"] else "") + f"_{row['pose']}"


def export(batch, market: str) -> list:
    """Makes (once) and lists the batch's photos for `market`, kept in <batch>/_market/<market>/."""
    if market not in MARKETS:
        raise ValueError(f"Pick a marketplace: {', '.join(MARKETS)}")
    folder = batch.root / "_market" / market
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for row in batch.results():
        if not (row.get("png") or row.get("jpg")):
            continue
        src = batch.root / (row["png"] or row["jpg"])
        dest = folder / f"{label(row)}.jpg"
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:  # recoloured again since: make it again
            tmp = dest.with_suffix(".tmp.jpg")
            for_market(source_photo(batch, row), market).save(tmp, quality=95, subsampling=0)
            tmp.replace(dest)
        files.append(dest)
    return files


def zip_bytes(batch, market: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for f in export(batch, market):
            z.write(f, f"{batch.spec.name}_{MARKETS[market][0]}/{f.name}")
    return buf.getvalue()
