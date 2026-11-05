"""The shop's catalogue folders on Google Drive: every finished photo of a batch filed by fabric, colour and set type,
under <fabric>/ (Cotton Mix or Pure Cotton):
<category>/<colour>/<style>/                                   single tops (cropped just above the knee)
<category>/<colour>/Co-ords/<style> set/                       top and bottom in the same colour
<category>/<colour>/Combo sets/<top>-<bottom>/<style> set/     contrast top and bottom
each holding <model>_front.jpg, <model>_back.jpg. A colour with no co-ord (White, worn with black bottoms)
gets its single top from its first combo."""

import io
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from . import studio

FABRIC_FOLDER = {"CottonMix": "Cotton Mix", "PureCotton": "Pure Cotton"}  # each fabric has its own top folder
# a category's folder in one fabric when it differs (Pure Cotton men go to a fresh folder, beside an older "Men")
CATEGORY_FOLDER_BY_FABRIC = {("PureCotton", "Men"): "cottonmens"}
CATEGORY_FOLDER = {"Kids_Boys": "Boys", "Kids_Girls": "Girls", "Men": "Men", "Women": "Women"}
STYLE_FOLDER = {"HalfSleeveTee": "Half sleeve", "SleevelessTee": "Sleeveless", "FullSleeveTee": "Full sleeve",
                "Hoodie": "Hoodies", "Sweatshirt": "Sweatshirt", "OversizedTee": "Oversized", "ZipHoodie": "Zip hoodies"}
POSE_NAME = {"front_main": "front", "back": "back"}
JPG = re.compile(r"^(?P<pose>[a-z_]+)_(?P<w>\d+)x(?P<h>\d+)\.jpg$")
KNEE_MARGIN = 0.03  # of the photo height: a single top is cut this far below the shorts' hem


def files(batch: "studio.Batch") -> List[dict]:
    """Where each finished JPG of the batch goes: [{key, src (path), parts (folders), name, crop}], crop True
    for a single top. Keys are what the uploader records once a file is on Drive."""
    spec = batch.spec
    names = {studio.slug(c): c for c in spec.top_colours + spec.bottom_colours + [b for _, b, _ in spec.colour_pairs]}
    names.update({studio.slug(studio.WHITE_TOP_BOTTOM): studio.WHITE_TOP_BOTTOM})
    fab = FABRIC_FOLDER.get(spec.fabric, spec.fabric)
    cat = CATEGORY_FOLDER_BY_FABRIC.get((spec.fabric, spec.category)) or CATEGORY_FOLDER.get(spec.category, spec.category)
    style = STYLE_FOLDER.get(spec.top, spec.top)
    rows = []
    for jpg in sorted(batch.root.glob("m[0-9][0-9][0-9][0-9]/*/*.jpg")):
        m = JPG.match(jpg.name)
        if not m or m["pose"] not in spec.poses or jpg.parent.name.startswith("_"):
            continue  # printed photos and anything else keep the batch layout
        t, _, b = jpg.parent.name.partition("__")
        rows.append({"model": jpg.parent.parent.name, "top": names.get(t, t), "bottom": names.get(b, b) if b else None,
                     "pose": m["pose"], "src": jpg, "rel": jpg.relative_to(batch.root).as_posix()})
    coord = {r["top"] for r in rows if r["top"] == r["bottom"]}
    order = [(t, b) for t, b in spec.pairs()]
    single_from: Dict[str, Optional[str]] = {}  # the bottom a colour's single top is cut from
    for t, b in order:
        if t not in coord and t not in single_from:
            single_from[t] = b
    out = []
    for r in rows:
        t, b = r["top"], r["bottom"]
        name = f"{r['model']}_{POSE_NAME.get(r['pose'], r['pose'])}.jpg"
        if b is None:  # a top-only shot is the single top itself
            out.append({**r, "key": r["rel"] + "#set", "parts": [fab, cat, t, style], "name": name, "crop": False})
            continue
        sets = [fab, cat, t, "Co-ords", f"{style} set"] if t == b else [fab, cat, t, "Combo sets", f"{t}-{b}", f"{style} set"]
        # keys differ from the batch layout's (the bare path), so photos sent there before are filed here too
        out.append({**r, "key": r["rel"] + "#set", "parts": sets, "name": name, "crop": False})
        base_png = batch.base_path(int(r["model"][1:]), spec.base_key(t, b), r["pose"])
        mask_png = base_png.with_name(f"{r['pose']}_bottom_mask.png")
        # the cut needs the base's size and bottom mask (the mask alone will do once the base is archived to Drive)
        if (t == b or single_from.get(t) == b) and (base_png.exists() or mask_png.exists()):
            out.append({**r, "key": r["rel"] + "#single", "parts": [fab, cat, t, style], "name": name, "crop": True})
    return out


def single_top(batch: "studio.Batch", f: dict) -> bytes:
    """The single top for one finished set photo: cut just above the knee (just below the hem of shorts),
    then fitted and sharpened to the batch's canvas like the set photo."""
    from .market import sharpen
    from .recolor import fit_canvas

    spec = batch.spec
    model = int(f["model"][1:])
    base = spec.base_key(f["top"], f["bottom"])
    mask_file = batch.base_path(model, base, f["pose"]).with_name(f"{f['pose']}_bottom_mask.png")
    jpg = Image.open(f["src"]).convert("RGB")
    W, H = jpg.size
    base_png = batch.base_path(model, base, f["pose"])
    with Image.open(base_png if base_png.exists() else mask_file) as bimg:  # the mask is the base's size
        bw, bh = bimg.size
    s = min(W / bw, H / bh)
    x0, y0 = (W - round(bw * s)) // 2, (H - round(bh * s)) // 2
    photo = jpg.crop((x0, y0, x0 + round(bw * s), y0 + round(bh * s)))
    cut = photo.height
    if mask_file.exists():
        rows = np.nonzero((np.asarray(Image.open(mask_file).convert("L")) > 127).any(axis=1))[0]
        if len(rows):
            top, low = rows.min() * s, rows.max() * s
            shorts = spec.bottom == "Shorts"
            cut = int(min(photo.height, low + KNEE_MARGIN * photo.height)) if shorts \
                else int(top + 0.5 * (low - top) - 0.02 * photo.height)
    piece = photo.crop((0, 0, photo.width, max(1, cut)))
    out = sharpen(fit_canvas(piece, W, H), min(W / piece.width, H / piece.height))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=95, subsampling=0)
    return buf.getvalue()


def freed(batch_root: Path) -> set:
    """Finished JPGs taken off the server once on Drive (listed in _freed.txt)."""
    f = Path(batch_root) / "_freed.txt"
    return set(f.read_text().splitlines()) if f.exists() else set()
