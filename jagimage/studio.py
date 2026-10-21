"""Studio batches: N different models wearing a top and a bottom, in any top colours and bottom
colours, from base photos made by Gemini or OpenAI and recoloured to the exact hex.

How a batch works:
- Each model gets its own look (jagimage.faces), so 300 models are 300 different people.
- The AI shoots each model in base colours only: the top in mint (light) or bottle green (dark),
  the bottom in lilac (light) or deep purple (dark). Greens and purples are never in skin, hair,
  the grey studio or white sneakers, and they are far apart from each other, so top and bottom
  can each be found and recoloured on their own.
- A model needs at most 4 base shoots (light/dark top x light/dark bottom), and only the ones its
  colour pairs use. The first base front is made from the text; every other shoot of that model
  uses it as the face reference, so all colours show the same person.
- Every selected top/bottom colour pair is then recoloured from the nearest base.

Layout: <out>/<job>/m0001/_base/<top base>-<bottom base>/<pose>.png (+ masks),
        <out>/<job>/m0001/<Top colour>__<Bottom colour>/<pose>.png, <pose>_1500x1500.jpg, qc.json
"""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import faces
from . import library as L

TOPS = ["HalfSleeveTee", "FullSleeveTee", "SleevelessTee", "Hoodie", "Sweatshirt"]
BOTTOMS = ["Shorts", "Trackpants", "Jeans"]
FIXED_BOTTOMS = {"Jeans": "straight-leg light-wash blue denim jeans"}  # not recoloured
TOP_BASES = {"light": ("mint green", "#8FD3A8"), "dark": ("bottle green", "#1E4636")}
BOTTOM_BASES = {"light": ("lilac", "#C4A3EE"), "dark": ("deep purple", "#3B1F66")}
NOT_MANNEQUIN = "mannequin, dummy, doll, ghost mannequin, faceless figure, plastic or wax figure, CGI character"
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def colour_hex(name: str, fabric: str) -> str:
    """A palette colour name (this fabric's palette first, then the other) or a #RRGGBB hex."""
    if HEX.match(name):
        return name.upper()
    for f in [fabric] + [x for x in L.COLOURS if x != fabric]:
        if name in L.COLOURS[f]:
            return L.COLOURS[f][name]
    raise ValueError(f"Unknown colour {name!r}: use a palette name or #RRGGBB")


def base_of(hex_: str) -> str:
    from .recolor import lightness

    return "light" if lightness(hex_) >= L.LIGHT_FROM else "dark"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


@dataclass
class Spec:
    category: str = "Kids_Boys"
    top: str = "HalfSleeveTee"
    bottom: str = "Shorts"
    fabric: str = "PureCotton"
    top_colours: List[str] = field(default_factory=lambda: ["Navy"])
    bottom_colours: List[str] = field(default_factory=list)  # empty: same colour as the top
    pairing: str = "match"   # match: bottom = top colour; each: every top with every bottom
    models: int = 1
    first_model: int = 1     # numbering, so a batch can be extended (301-600) later
    poses: List[str] = field(default_factory=lambda: ["front_main"])
    provider: str = "gemini"
    quality: str = "high"    # OpenAI
    image_size: str = "2K"   # Gemini
    canvas: Optional[Tuple[int, int]] = (1500, 1500)
    name: str = ""

    def __post_init__(self):
        if self.category not in L.CATEGORIES:
            raise ValueError(f"category must be one of {', '.join(L.CATEGORIES)}")
        if self.top not in TOPS:
            raise ValueError(f"top must be one of {', '.join(TOPS)}")
        if self.bottom not in BOTTOMS:
            raise ValueError(f"bottom must be one of {', '.join(BOTTOMS)}")
        if self.fabric not in L.FABRICS:
            raise ValueError(f"fabric must be one of {', '.join(L.FABRICS)}")
        if not self.top_colours:
            raise ValueError("pick at least one top colour")
        if self.pairing not in ("match", "each"):
            raise ValueError("pairing must be match or each")
        if self.pairing == "each" and not self.bottom_colours and self.bottom not in FIXED_BOTTOMS:
            raise ValueError("pick at least one bottom colour, or pair the bottom with the top colour")
        bad = [p for p in self.poses if p not in L.POSES]
        if bad or "front_main" not in self.poses:
            raise ValueError("poses must include front_main and come from: " + ", ".join(L.POSES))
        if not 1 <= int(self.models) <= 5000:
            raise ValueError("models must be between 1 and 5000")
        for c in self.top_colours + self.bottom_colours:
            colour_hex(c, self.fabric)
        if self.canvas:
            self.canvas = tuple(int(v) for v in self.canvas)
        if not self.name:
            self.name = slug(f"{self.category}_{self.top}_{self.bottom}_{self.fabric}")

    # ---- what the batch makes
    def pairs(self) -> List[Tuple[str, Optional[str]]]:
        """(top colour, bottom colour) pairs; bottom None for a fixed bottom such as jeans."""
        if self.bottom in FIXED_BOTTOMS:
            return [(t, None) for t in self.top_colours]
        if self.pairing == "match" or not self.bottom_colours:
            return [(t, t) for t in self.top_colours]
        return [(t, b) for t in self.top_colours for b in self.bottom_colours]

    def base_key(self, top: str, bottom: Optional[str]) -> Tuple[str, Optional[str]]:
        return (base_of(colour_hex(top, self.fabric)),
                base_of(colour_hex(bottom, self.fabric)) if bottom else None)

    def bases(self) -> List[Tuple[str, Optional[str]]]:
        """The base shoots each model needs, the most used first (that one is made from the text)."""
        count: Dict[Tuple[str, Optional[str]], int] = {}
        for t, b in self.pairs():
            k = self.base_key(t, b)
            count[k] = count.get(k, 0) + 1
        return sorted(count, key=lambda k: (-count[k], str(k)))

    def model_numbers(self) -> List[int]:
        return list(range(self.first_model, self.first_model + int(self.models)))

    def plan(self) -> dict:
        n, poses = int(self.models), len(self.poses)
        return {"models": n, "bases_per_model": len(self.bases()), "colour_pairs": len(self.pairs()),
                "poses": poses, "ai_images": n * len(self.bases()) * poses,
                "recoloured_images": n * len(self.pairs()) * poses}


def base_dir(name) -> str:
    t, b = name
    return f"{t}-{b}" if b else t


def pair_dir(top: str, bottom: Optional[str]) -> str:
    return slug(top) + ("__" + slug(bottom) if bottom else "")


def prompt(spec: Spec, model_no: int, base, pose: str, reference: str) -> str:
    """reference: 'none' (first base front: new face from the text), 'face' (other base fronts:
    image 1 is the model's first front), 'front' (poses of the first base: image 1 is this base's
    front), 'face+front' (poses of other bases: image 1 face, image 2 this base's front)."""
    tname, thex = TOP_BASES[base[0]]
    kids = L.is_kids(spec.category)
    top_text = L.GARMENTS[spec.top].replace("{COLOUR}", tname)
    if spec.bottom in FIXED_BOTTOMS:
        bottom_text, bcol = FIXED_BOTTOMS[spec.bottom], None
    else:
        bname, bhex = BOTTOM_BASES[base[1]]
        bottom_text, bcol = L.GARMENTS[spec.bottom].replace("{COLOUR}", bname), (bname, bhex)
    shoes = "small white kids sneakers" if kids else "clean white low-top sneakers"
    lines = {"none": "", "face": L.REFERENCE_LINE, "front": L.FRONT_ONLY_LINE,
             "face+front": L.REFERENCE_LINE + L.MATCH_FRONT_LINE}
    colours = f"top {tname} ({thex})" + (f", bottom {bcol[0]} ({bcol[1]})" if bcol else "")
    rules = L.RULES.replace("Exact garment colour {COLOUR} ({HEX}),", f"Exact garment colours: {colours}, each")
    text = (f"{lines[reference]}Real photograph of a real, living human model (not a mannequin, dummy, doll or 3D figure): "
            f"{faces.describe(spec.category, model_no)}, {L.POSES[pose]}. "
            f"Wearing {top_text}, fabric: {L.FABRIC_TEXT[spec.fabric][L.FABRIC_KIND[spec.top]]}; "
            f"with {bottom_text}" + (f", fabric: {L.FABRIC_TEXT[spec.fabric]['bottom']}" if bcol else "") +
            f"; and {shoes}. The top and the bottom are two clearly different colours, exactly as stated. "
            f"{L.BACKGROUNDS.get(pose, L.GREY)} {rules}")
    if kids:
        text += L.KIDS_RULE
    product = "Trackpants" if spec.bottom == "Trackpants" else spec.top
    negative = L.build_negative(spec.category, product, pose)
    if spec.bottom == "Trackpants":
        text += L.JOGGER_RULE
    return text + " Avoid: " + negative + ", " + NOT_MANNEQUIN + ", top and bottom in the same colour."


class Batch:
    """Runs a Spec: AI base shoots (Gemini/OpenAI) then exact-hex recolours. Safe to re-run: files
    that exist are skipped, so an interrupted batch continues where it stopped."""

    def __init__(self, spec: Spec, out="output/studio", generate: Optional[Callable[[str, List[str]], bytes]] = None,
                 workers=4, retries=3, on_event: Optional[Callable[[dict], None]] = None):
        self.spec = spec
        self.root = Path(out) / spec.name
        self.generate = generate          # (prompt, reference paths) -> image bytes
        self.workers = workers
        self.retries = retries
        self.on_event = on_event or (lambda e: None)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.counts = {"ai_done": 0, "ai_skipped": 0, "recoloured": 0, "recolour_skipped": 0, "errors": 0}

    # ---- paths
    def model_dir(self, m: int) -> Path:
        return self.root / f"m{m:04d}"

    def base_path(self, m: int, base, pose: str) -> Path:
        return self.model_dir(m) / "_base" / base_dir(base) / f"{pose}.png"

    def out_path(self, m: int, top: str, bottom: Optional[str], pose: str) -> Path:
        return self.model_dir(m) / pair_dir(top, bottom) / f"{pose}.png"

    def _event(self, kind: str, **kw):
        with self.lock:
            if kind in self.counts:
                self.counts[kind] += 1
        self.on_event({"kind": kind, **kw})

    # ---- run
    def run(self, stage="all"):
        """stage: all | bases (AI only) | recolor (no AI calls)."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "spec.json").write_text(json.dumps(asdict(self.spec), indent=2))
        if stage != "recolor" and self.generate is None:
            raise RuntimeError("No image engine: pass generate=, or use stage='recolor'")
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            list(pool.map(lambda m: self.run_model(m, stage), self.spec.model_numbers()))
        return self.counts

    def run_model(self, m: int, stage="all"):
        spec = self.spec
        md = self.model_dir(m)
        md.mkdir(parents=True, exist_ok=True)
        (md / "model.json").write_text(json.dumps({"model": m, "look": faces.look(spec.category, m),
                                                    "description": faces.describe(spec.category, m)}, indent=2))
        bases = spec.bases()
        first_front = self.base_path(m, bases[0], "front_main")
        poses = ["front_main"] + [p for p in spec.poses if p != "front_main"]
        for base in bases:
            for pose in poses:
                if self.stop.is_set():
                    return
                path = self.base_path(m, base, pose)
                if stage != "recolor" and not self._shoot(m, base, pose, path, first_front):
                    continue
                if stage != "bases" and path.exists():
                    self._recolour_base(m, base, pose)

    def _shoot(self, m, base, pose, path: Path, first_front: Path) -> bool:
        """Make one base photo if missing. False if it couldn't be made."""
        if path.exists():
            self._event("ai_skipped", model=m, path=str(path))
            return True
        is_first = path.parent == first_front.parent
        own_front = path.parent / "front_main.png"
        if pose == "front_main":
            ref, refs = ("none", []) if is_first else ("face", [str(first_front)])
        else:
            ref, refs = ("front", [str(own_front)]) if is_first else ("face+front", [str(first_front), str(own_front)])
        if any(not Path(r).exists() for r in refs):
            self._event("errors", model=m, path=str(path), error="its reference front is missing")
            return False
        text = prompt(self.spec, m, base, pose, ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(self.retries):
            if self.stop.is_set():
                return False
            try:
                data = self.generate(text, refs)
                tmp = path.with_suffix(".tmp.png")
                tmp.write_bytes(data)
                self._check_base(tmp, base)  # the AI must have used both base colours
                tmp.replace(path)
                self._event("ai_done", model=m, path=str(path), base=base_dir(base), pose=pose)
                return True
            except Exception as exc:  # API errors, rate limits, wrong colours: retry with backoff
                last = exc
                time.sleep(min(30, 2 ** attempt) if attempt + 1 < self.retries else 0)
        self._event("errors", model=m, path=str(path), error=f"{type(last).__name__}: {last}")
        return False

    def _check_base(self, path: Path, base):
        import numpy as np
        from PIL import Image

        from .recolor import MaskError, garment_mask, rgb_to_lab, save_mask

        lab = rgb_to_lab(np.asarray(Image.open(path).convert("RGB")))
        top, _ = garment_mask(lab, TOP_BASES[base[0]][1])
        masks = {"top": top}
        if base[1]:
            bottom, _ = garment_mask(lab, BOTTOM_BASES[base[1]][1])
            bottom = np.minimum(bottom, 1 - top)
            if (bottom > 0.95).sum() < 0.002 * bottom.size:
                raise MaskError("the bottom isn't in its base colour")
            masks["bottom"] = bottom
        stem = path.name.replace(".tmp.png", "")
        for k, mk in masks.items():  # saved next to the base; edit them to fix an edge
            save_mask(path.parent / f"{stem}_{k}_mask.png", mk)

    def masks(self, m, base, pose):
        from PIL import Image

        from .recolor import load_mask

        src = self.base_path(m, base, pose)
        tm, bm = src.parent / f"{pose}_top_mask.png", src.parent / f"{pose}_bottom_mask.png"
        if not tm.exists() or (base[1] and not bm.exists()):
            self._check_base_existing(src, base)
        return Image.open(src), load_mask(tm), load_mask(bm) if base[1] else None

    def _check_base_existing(self, src: Path, base):
        tmp = src.with_suffix(".tmp.png")
        tmp.write_bytes(src.read_bytes())
        try:
            self._check_base(tmp, base)
        finally:
            tmp.unlink(missing_ok=True)

    def _recolour_base(self, m, base, pose):
        spec = self.spec
        todo = [(t, b) for t, b in spec.pairs() if spec.base_key(t, b) == base]
        if not todo:
            return
        try:
            img, tmask, bmask = self.masks(m, base, pose)
        except Exception as exc:
            self._event("errors", model=m, path=str(self.base_path(m, base, pose)), error=f"mask: {exc}")
            return
        for t, b in todo:
            if self.stop.is_set():
                return
            path = self.out_path(m, t, b, pose)
            if path.exists():
                self._event("recolour_skipped", model=m, path=str(path))
                continue
            try:
                self._recolour_one(img, tmask, bmask, base, t, b, path, pose)
                self._event("recoloured", model=m, path=str(path), top=t, bottom=b, pose=pose)
            except Exception as exc:
                self._event("errors", model=m, path=str(path), error=f"recolour: {exc}")

    def _recolour_one(self, img, tmask, bmask, base, top, bottom, path: Path, pose):
        from .recolor import fit_canvas, recolor_image, write_qc

        spec = self.spec
        thex = colour_hex(top, spec.fabric)
        out, de_t, _ = recolor_image(img, TOP_BASES[base[0]][1], thex, mask=tmask)
        qc = {"top": {"colour": top, "hex": thex, "deltaE": round(de_t, 2), "on_shade": de_t < 3}}
        if bottom:
            bhex = colour_hex(bottom, spec.fabric)
            out, de_b, _ = recolor_image(out, BOTTOM_BASES[base[1]][1], bhex, mask=bmask)
            qc["bottom"] = {"colour": bottom, "hex": bhex, "deltaE": round(de_b, 2), "on_shade": de_b < 3}
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path)
        if spec.canvas:
            fit_canvas(out, *spec.canvas).save(path.with_name(f"{pose}_{spec.canvas[0]}x{spec.canvas[1]}.jpg"), quality=95)
        qc_path = path.parent / "qc.json"
        rows = json.loads(qc_path.read_text()) if qc_path.exists() else {}
        rows[pose] = qc
        write_qc(qc_path, rows)

    # ---- results
    def results(self) -> List[dict]:
        """Every finished photo: model, top, bottom, pose, paths (relative to the batch folder)."""
        rows = []
        for md in sorted(self.root.glob("m[0-9][0-9][0-9][0-9]")):
            for pd in sorted(p for p in md.iterdir() if p.is_dir() and not p.name.startswith("_")):
                top, _, bottom = pd.name.partition("__")
                for img in sorted(pd.glob("*.png")):
                    canvas = sorted(pd.glob(f"{img.stem}_*x*.jpg"))
                    rows.append({"model": md.name, "top": top, "bottom": bottom or None, "pose": img.stem,
                                 "png": str(img.relative_to(self.root)),
                                 "jpg": str(canvas[0].relative_to(self.root)) if canvas else None})
        return rows
