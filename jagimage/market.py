"""Marketplace downloads: every finished photo resized for one marketplace's listing rules, sharpened
so a buyer's zoom stays crisp, and on pure white where the marketplace wants a white main image.

Sharpening is a plain unsharp mask after a Lanczos resize. In our zoom test (2026-10) it kept the knit,
the stitching and natural skin, where an AI upscaler (Real-ESRGAN) made skin waxy and wiped out the knit.
Sizes are the highest figures in 2026 seller guides; check them against each seller portal."""

import io
import zipfile
from pathlib import Path

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
WHITE_EDGE = 1.5  # px: the backdrop's edge is softened this much when it turns white, so hair stays natural


def on_white(img: Image.Image) -> Image.Image:
    """The studio backdrop made pure white (#FFFFFF). Darker spots in it (the contact shadow) keep
    their share of darkness, so the model still stands on the floor instead of floating."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    lab = rgb_to_lab(rgb.astype(np.uint8))
    bg = backdrop_mask(lab)
    if bg.mean() < 0.05:  # no backdrop found: leave the photo as it is
        return img.convert("RGB")
    lit = float(np.percentile(lab[..., 0][bg], 75))
    # the backdrop's own soft shading turns pure white (#FFFFFF); only a real shadow stays grey
    shade = np.clip(lab[..., 0] / (WHITE_FROM * max(lit, 1e-3)), 0, 1) ** 2
    white = 255.0 * shade[..., None]
    grown = ndimage.binary_dilation(bg, iterations=1)  # takes the 1 px grey rim around hair too
    m = np.clip(ndimage.gaussian_filter(grown.astype(np.float32), WHITE_EDGE), 0, 1)[..., None]
    out = rgb * (1 - m) + white * m
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8))


def sharpen(img: Image.Image, scale: float) -> Image.Image:
    """Unsharp mask matched to how much the photo was enlarged (as in the zoom test at 2x)."""
    if scale > 1.02:
        return img.filter(ImageFilter.UnsharpMask(radius=min(2.0, scale), percent=60 + int(20 * min(1.0, scale - 1)), threshold=2))
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=40, threshold=2))  # downsized: a light touch


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
