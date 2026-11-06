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
        <out>/<job>/m0001/<Top colour>__<Bottom colour>/<pose>.png, <pose>_2000x2000.jpg, qc.json
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

TOPS = ["HalfSleeveTee", "FullSleeveTee", "SleevelessTee", "Hoodie", "Sweatshirt", "OversizedTee", "ZipHoodie"]
ADULT_ONLY_TOPS = {"OversizedTee"}  # drop-shoulder oversized tees are made for Men and Women only
KIDS_ONLY_TOPS = {"ZipHoodie"}      # full-zip hoodies are made for Kids Boys and Kids Girls only


def tops_for(category: str) -> list:
    """The top styles made for a category, in TOPS order."""
    left_out = ADULT_ONLY_TOPS if L.is_kids(category) else KIDS_ONLY_TOPS
    return [t for t in TOPS if t not in left_out]
BOTTOMS = ["Shorts", "Trackpants", "Jeans"]
FIXED_BOTTOMS = {"Jeans": "straight-leg light-wash blue denim jeans"}  # not recoloured
TOP_BASES = {"light": ("mint green", "#8FD3A8"), "dark": ("bottle green", "#1E4636")}
BOTTOM_BASES = {"light": ("lilac", "#C4A3EE"), "dark": ("deep purple", "#3B1F66")}
HANDS_FREE = ("Hands free: both hands relaxed and open, never touching the clothes or the body, not in pockets, "
              "not holding anything, arms kept a little away from the body so the whole garment is visible, "
              "including its side seams and hem")
HANDS_FREE_AVOID = "hands on the clothes, hands in pockets, hands tucked in, arms covering the garment, holding an object"
# Flash sometimes returns several images in one response (2 to 35 seen); each is billed, one is used.
ONE_IMAGE = "Output exactly one image: a single photograph of this one model, not a sequence, grid, collage or set of variations."
NO_HANDS_FREE = {"handpocket"}  # poses that need the hands on the clothes
NOT_MANNEQUIN = "mannequin, dummy, doll, ghost mannequin, faceless figure, plastic or wax figure, CGI character"
SHOTS = {"outfit": "Top and bottom (full body, both recoloured)", "top": "Top only (head to just above the knee)"}
SHOT_DEFAULTS: Dict[str, str] = {}  # the shot a top style starts with when it isn't "outfit" (every style is a full outfit now)
# the bottom each top style is worn with unless one is picked: shorts with short sleeves and sleeveless, else track pants
BOTTOM_DEFAULTS = {"HalfSleeveTee": "Shorts", "SleevelessTee": "Shorts", "OversizedTee": "Shorts"}


def default_shot(top: str) -> str:
    return SHOT_DEFAULTS.get(top, "outfit")


def default_bottom(top: str) -> str:
    return BOTTOM_DEFAULTS.get(top, "Trackpants")

TOP_ONLY_FRAMING = ("Three-quarter shot framed from the top of the head to just above the knees: knees, shins and feet are "
                    "out of frame, small margin above the head, the whole top fully visible, model fills about 90% of frame height")
FULL_FRAMING = "Full body head to shoes in frame with small margins, model about 85-90% of frame height"
CLOSE_UP = re.compile(r"close[- ]?up|detail shot|zoomed", re.I)
CLOSE_FRAMING = ("Close-up framing exactly as the pose describes, the garment detail sharp and filling most of the frame, "
                 "real model wearing it, eye-level camera, 3:4 portrait")
TOP_ONLY_NEW_PHOTO = "Create a new three-quarter photo framed from the top of the head to just above the knees (no legs below the knee, no feet). "
BOTTOM_LIKE = 0.06  # OKLab distance: pixels this close to a top-only shot's photographed bottom aren't the top
BOTTOM_TRIM_BAND = 0.06  # of photo height: only that close above the top's hem can pixels be trimmed as the bottom
BACKDROP_MAX_CHROMA, BACKDROP_MIN_LIGHT = 0.03, 0.65  # studio backdrop at the photo's top and upper sides
ZIP_GAP = 0.02  # of photo width: a gap this narrow inside the top (a black zip) doesn't make two figures
MIN_BASE_STRENGTH = 0.33  # a base garment keeps at least this share of its base colour's chroma (good ones: >= 0.38)
BASE_MB, PNG_MB, JPG_MB = 4.1, 4.3, 0.6  # measured sizes: AI base PNG, full-size photo PNG, 2000x2000 JPG
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def colour_hex(name: str, fabric: str) -> str:
    """A palette colour name (this fabric's palette first, then the other) or a #RRGGBB hex."""
    if not isinstance(name, str):
        raise ValueError(f"colours are names or #RRGGBB codes, not {name!r}")
    if HEX.match(name):
        return name.upper()
    for f in [fabric] + [x for x in L.COLOURS if x != fabric]:
        if name in L.COLOURS[f]:
            return L.COLOURS[f][name]
    raise ValueError(f"Unknown colour {name!r}: use a palette name or #RRGGBB")


def _base_hued(lab, base):
    """Pixels plainly in the base colour's hue (its folds included): never the photographed bottom."""
    import numpy as np

    C = np.hypot(lab[..., 1], lab[..., 2])
    dh = np.abs(np.angle(np.exp(1j * (np.arctan2(lab[..., 2], lab[..., 1]) - np.arctan2(base[2], base[1])))))
    return (dh < np.radians(40)) & (C > 0.015)


RETOUCH_BOTTOMS = ("Edit image 1. Change only the shorts or pants: remove the drawstring cords, their ends and eyelets, and remove "
                   "every pocket and pocket opening, so the waistband is a clean flat band covered in the same fabric and the "
                   "front and sides are completely plain and smooth, with only the outer side seam. The top stays untucked, its hem "
                   "hanging over the waistband (never tuck it in). Keep everything else exactly "
                   "the same: the child, face, hair, pose, hands, the top, all colours (the top's and the bottom's), lighting, "
                   "background, framing and image size. Return one photo.")
BOTTOM_WORD = {"Shorts": "shorts", "Trackpants": "track pants"}
BOTTOM_REF_LINE = (" Image {n} is a product photo of the exact {item} the model wears: copy its cut, length, plain covered "
                   "waistband and smooth flat front and sides exactly (no drawstring, no pockets), but in the bottom colour "
                   "stated above, not the colour in that photo, and worn by the model, not laid flat.")


def bottom_ref_path(out: Path, bottom: str) -> Path:
    """Where the shop's product photo of a bottom style is kept: <studio folder>/_refs/bottoms/<style>.jpg."""
    return Path(out) / "_refs" / "bottoms" / f"{bottom}.jpg"


# hair accessories in the clothes' colour get recoloured with them (or keep the old colour): none at all
HAIR_RULE = (" No hair accessories in any colour: no bows, ribbons, hair slides, flowers or scrunchies; hair held only by "
             "plain black or dark brown elastics where needed.")
UNTUCKED_RULE = (" The top is worn untucked: its hem hangs loosely OVER the waistband of the bottom, which is hidden under "
                 "the top's hem; the top is never tucked in.")
WHITE_TOP_BOTTOM = "Black"  # bottoms aren't made in white: a white top is paired with this bottom


ARCHIVED = ".archived"  # next to a base photo moved to Google Drive (kinseymodels2026/_bases/...) to free the disk


def archived(path: Path) -> bool:
    """The base photo is kept on Google Drive, not on the server: it is never made (or paid for) again."""
    return Path(str(path) + ARCHIVED).exists()


def fabric_twin(spec: "Spec", fabric: str) -> "Spec":
    """The batch that recolours `spec`'s models into `fabric`'s top colours from its base photos. The bottoms are
    the same range as the source batch's (bottom_fabric): each top colour with the bottom of the same name when
    that bottom is made (a co-ord, in the bottom fabric's shade), else with the black bottom; plus the batch's own
    combos whose top is in this fabric's list."""
    names = list(L.colours_for(fabric, spec.category))
    body = {k: v for k, v in asdict(spec).items() if k not in ("name", "also_fabrics", "base_from")}
    body.update(fabric=fabric, base_from=spec.name, also_fabrics=[])
    if spec.shot != "outfit":
        body.update(top_colours=names, colour_pairs=[], pairing="match")
        return Spec(**body)
    n = max(1, spec.models)
    # the bottoms stay the source batch's (one bottom range for every top fabric): a top colour gets the bottom of
    # the same name when that bottom is made (a co-ord), else the black one; plus the batch's combos that fit
    bottoms = [c for c in L.colours_for(spec.bfab, spec.category) if not is_white(c, spec.bfab)]
    body["bottom_fabric"] = spec.bfab
    pairs = [[c, c if (c in bottoms and not is_white(c, fabric)) else WHITE_TOP_BOTTOM, n] for c in names]
    seen = {(t, b) for t, b, _ in pairs}
    for t, b in spec.pairs():
        if b and t != b and t in names and b in bottoms and (t, b) not in seen:
            pairs.append([t, b, n])
            seen.add((t, b))
    body.update(colour_pairs=pairs, pairing="pairs", top_colours=[p[0] for p in pairs], bottom_colours=[])
    return Spec(**body)


def approve(root: Path, photos: list) -> int:
    """A person approves photos the QC agent held (paths like m0014/Red__Red/front_main_2000x2000.jpg):
    they are uploaded on the next Drive round."""
    from .recolor import write_qc

    root, n = Path(root).resolve(), 0
    for rel in photos:
        jpg = (root / str(rel)).resolve()
        if root not in jpg.parents or not jpg.name.endswith(".jpg"):
            continue  # only photos inside this batch
        qc_file = jpg.parent / "qc.json"
        pose = jpg.name.rsplit("_", 1)[0]
        try:
            rows = json.loads(qc_file.read_text())
        except (OSError, ValueError):
            continue
        if pose in rows:
            rows[pose].setdefault("check", {})["approved"] = True
            write_qc(qc_file, rows)
            n += 1
    return n


def is_white(name: str, fabric: str) -> bool:
    """White or an off-white (e.g. Cotton Mix #E7E7EC), not a light grey such as Melange or a pastel."""
    import numpy as np

    from .recolor import hex_to_lab

    lab = hex_to_lab(colour_hex(name, fabric))
    return float(lab[0]) >= 0.92 and float(np.hypot(lab[1], lab[2])) < 0.02


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
    pairing: str = "match"   # match: bottom = top colour; each: every top with every bottom; pairs: colour_pairs
    # Sets: exact (top colour, bottom colour, models) pairs, e.g. a combo set Black top + Melange bottom on the
    # first 45 models. The batch makes max(models) models; a pair goes on only its first `models` of them.
    colour_pairs: List[List] = field(default_factory=list)
    set_type: str = ""       # coord (same colour top and bottom), combo (contrast colours), or empty
    retouch_bottoms: bool = False  # Gemini: an edit pass takes drawstrings and pockets off plain shorts and pants
    drive_layout: str = "batch"    # Google Drive: "batch" folders, or the shop's "catalogue" folders (jagimage.catalogue)
    free_after_upload: bool = False  # take each finished JPG off the server once it is on Google Drive
    qc_agent: bool = False  # the QC agent checks base photos and finished photos; only passed photos are uploaded
    # the same models recoloured into these fabrics' colours too (no new AI photos): each becomes a twin batch
    # <..._PureCotton...> that reads this batch's base photos and goes to its own fabric folder on Drive
    also_fabrics: List[str] = field(default_factory=list)
    base_from: str = ""      # a twin batch: the batch whose base photos it recolours (it never makes its own)
    bottom_fabric: str = ""  # the bottoms' fabric when it isn't the top's (Pure Cotton tops, Cotton Mix bottoms): its colours
    shot: str = "outfit"     # outfit: full body, top and bottom recoloured; top: above the knee, only the top
    bottom_colour: str = "Navy"  # top-only shots: the bottom in this colour as the AI makes it (not recoloured)
    models: int = 1
    first_model: int = 1     # numbering, so a batch can be extended (301-600) later
    poses: List[str] = field(default_factory=lambda: ["front_main"])
    # Your own poses (name -> description), or new wording for a library pose such as front_main.
    custom_poses: Dict[str, str] = field(default_factory=dict)
    extra: str = ""          # extra instructions added to every photo's prompt
    hands_free: bool = True  # hands never touch the clothes (every pose except handpocket)
    batch: bool = False      # Gemini batch mode: half price, results within 24 hours
    keep_png: bool = True    # also keep a full-size PNG of every photo (4 MB each), besides the 2000x2000 JPG
    same_models: bool = False  # reuse models (faces) already made in this category, e.g. in another top style
    one_base: bool = False   # one light base shoot per model; dark colours are recoloured from it too (half the AI images)
    provider: str = "gemini"  # gemini, openai, meta, or runpod (your own GPU pod with an open model)
    quality: str = "medium"  # OpenAI: medium looks right for catalogue photos at about half the cost of high
    image_size: str = "2K"   # Gemini
    gemini_model: str = "flash"  # Gemini: flash (default, cheaper) or pro
    canvas: Optional[Tuple[int, int]] = (2000, 2000)  # marketplace JPG: 2000x2000 keeps zoom detail (Amazon's zoom size)
    background: str = ""     # a marketplace guide's backdrop, e.g. "pure white (#FFFFFF)"; empty: the studio grey
    guide: str = ""          # the marketplace guide these settings came from (shown on the dashboard), e.g. "Myntra"
    print_place: str = "chest_large"  # where prints go on the front photos (jagimage.prints.PLACES)
    print_dx: float = 0.0    # the whole batch's print moved right (+) / left, as a share of the chest width
    print_dy: float = 0.0    # ... and down (+) / up
    print_scale: float = 1.0  # ... and made bigger / smaller
    print_check: bool = False  # the AI print checker looks at every printed photo and fixes the placement
    name: str = ""

    def __post_init__(self):
        if self.category not in L.CATEGORIES:
            raise ValueError(f"category must be one of {', '.join(L.CATEGORIES)}")
        if self.top not in TOPS:
            raise ValueError(f"top must be one of {', '.join(TOPS)}")
        if self.top not in tops_for(self.category):
            raise ValueError("Oversized drop-shoulder T-shirts are for Men and Women only" if self.top in ADULT_ONLY_TOPS
                             else "Zip hoodies are for Kids Boys and Kids Girls only")
        if self.bottom not in BOTTOMS:
            raise ValueError(f"bottom must be one of {', '.join(BOTTOMS)}")
        if self.fabric not in L.FABRICS:
            raise ValueError(f"fabric must be one of {', '.join(L.FABRICS)}")
        if not self.top_colours:
            raise ValueError("pick at least one top colour")
        if self.colour_pairs:
            self._check_pairs()
        if self.pairing not in ("match", "each", "pairs"):
            raise ValueError("pairing must be match, each or pairs")
        if self.drive_layout not in ("batch", "catalogue"):
            raise ValueError("drive_layout must be batch or catalogue")
        if self.set_type not in ("", "coord", "combo"):
            raise ValueError("set_type must be coord or combo")
        if self.provider not in ("gemini", "openai", "meta", "runpod"):
            raise ValueError("provider must be gemini, openai, meta or runpod")
        if self.gemini_model not in ("flash", "pro"):
            raise ValueError("gemini_model must be flash or pro")
        if self.batch and self.provider not in ("gemini", "openai"):
            raise ValueError("Batch mode (half price) is available with Gemini and OpenAI")
        if self.shot not in SHOTS:
            raise ValueError("shot must be outfit (top and bottom) or top (top only)")
        if self.shot == "top":
            colour_hex(self.bottom_colour, self.fabric)
            if is_white(self.bottom_colour, self.fabric):
                raise ValueError("Bottoms aren't made in white: pick another bottom colour")
        elif self.pairing == "each" and not self.bottom_colours and self.bottom not in FIXED_BOTTOMS:
            raise ValueError("pick at least one bottom colour, or pair the bottom with the top colour")
        self.custom_poses = {slug(k).lower(): " ".join(str(v).split()) for k, v in (self.custom_poses or {}).items()}
        empty = [k for k, v in self.custom_poses.items() if not k or not v]
        if empty:
            raise ValueError("each custom pose needs a name and a description")
        self.extra = " ".join(str(self.extra or "").split())
        self.background, self.guide = " ".join(str(self.background).split()), " ".join(str(self.guide).split())
        from .prints import PLACES

        if self.print_place not in PLACES:
            raise ValueError(f"print_place must be one of {', '.join(PLACES)}")
        from .prints import adjust

        if not isinstance(self.print_check, bool):  # "false" or "no" would otherwise switch the paid AI checker on
            raise ValueError("print_check must be true or false")
        a = adjust(self.print_dx, self.print_dy, self.print_scale)
        self.print_dx, self.print_dy, self.print_scale = a["dx"], a["dy"], a["scale"]
        if len(self.background) > 200 or len(self.guide) > 80:
            raise ValueError("background must be at most 200 characters and guide at most 80")
        if len(self.extra) > 1000 or any(len(v) > 1000 for v in self.custom_poses.values()):
            raise ValueError("keep pose descriptions and extra instructions under 1000 characters")
        bad = [p for p in self.poses if p not in self.pose_texts()]
        if bad or "front_main" not in self.poses:
            raise ValueError("poses must include front_main and come from: " + ", ".join(L.POSES))
        try:
            self.models, self.first_model = int(self.models), int(self.first_model or 0)
        except (TypeError, ValueError):
            raise ValueError("models and the first model number must be whole numbers") from None
        if not 1 <= self.models <= 5000:
            raise ValueError("models must be between 1 and 5000")
        self.first_model = int(self.first_model)
        if self.first_model < 1 or self.first_model + int(self.models) - 1 > faces.capacity(self.category):
            raise ValueError(f"model numbers go from 1 to {faces.capacity(self.category)} for {self.category}")
        if self.bottom_fabric and self.bottom_fabric not in L.FABRICS:
            raise ValueError(f"bottom_fabric must be one of {', '.join(L.FABRICS)}")
        for c in self.top_colours:
            colour_hex(c, self.fabric)
        for c in self.bottom_colours:
            colour_hex(c, self.bfab)
        whites = [c for c in self.bottom_colours if is_white(c, self.bfab)]
        if whites:
            raise ValueError(f"Bottoms aren't made in white: take {', '.join(whites)} out of the bottom colours")
        if self.canvas:
            try:
                self.canvas = tuple(int(v) for v in self.canvas)
            except (TypeError, ValueError):
                raise ValueError("canvas must be width and height between 200 and 6000 px")
            if len(self.canvas) != 2 or not all(200 <= v <= 6000 for v in self.canvas):
                raise ValueError("canvas must be width and height between 200 and 6000 px")
        else:
            self.keep_png = True  # without a marketplace JPG the PNG is the only copy
        self.also_fabrics = [f for f in dict.fromkeys(self.also_fabrics or []) if f != self.fabric]
        if any(f not in L.FABRICS for f in self.also_fabrics):
            raise ValueError(f"also_fabrics: each must be one of {', '.join(L.FABRICS)}")
        if self.base_from and (not SAFE_NAME.match(self.base_from) or ".." in self.base_from):
            raise ValueError("base_from: a batch name")
        if not self.name:
            self.name = slug(f"{self.category}_{self.top}_{self.bottom}_{self.fabric}" + ("_top_only" if self.shot == "top" else "")
                             + (f"_{self.set_type.capitalize()}" if self.set_type else "")
                             + (f"_{self.guide}" if self.guide else ""))  # e.g. ..._Myntra: its own size and backdrop
        if not SAFE_NAME.match(self.name) or ".." in self.name:  # a folder inside the photos folder, nothing else
            raise ValueError("batch name: letters, digits, dot, dash and underscore only")

    @property
    def bfab(self) -> str:
        """The fabric whose colour list the bottoms' colours come from."""
        return self.bottom_fabric or self.fabric

    def pose_texts(self) -> Dict[str, str]:
        return {**L.POSES, **self.custom_poses}

    # ---- what the batch makes
    def _check_pairs(self):
        rows = []
        for p in self.colour_pairs:
            if not isinstance(p, (list, tuple)) or len(p) not in (2, 3):
                raise ValueError("each colour pair is [top colour, bottom colour] or [top, bottom, models]")
            t, b = str(p[0]), str(p[1])
            n = int(p[2]) if len(p) == 3 else int(self.models)
            if not 1 <= n <= int(self.models):
                raise ValueError(f"{t}/{b}: models must be between 1 and {self.models}")
            colour_hex(t, self.fabric), colour_hex(b, self.bfab)
            if is_white(b, self.bfab):
                raise ValueError(f"Bottoms aren't made in white: {t}/{b}")
            if [t, b] in [r[:2] for r in rows]:
                raise ValueError(f"{t}/{b} is listed twice")
            rows.append([t, b, n])
        if self.shot == "top" or self.bottom in FIXED_BOTTOMS:
            raise ValueError("Colour pairs need the full outfit shot with a recoloured bottom")
        self.colour_pairs, self.pairing = rows, "pairs"
        self.top_colours = list(dict.fromkeys(t for t, _, _ in rows))
        self.bottom_colours = []

    def pairs(self, m: Optional[int] = None) -> List[Tuple[str, Optional[str]]]:
        """(top colour, bottom colour) pairs; bottom None when it isn't recoloured (jeans, top-only shots).
        With colour_pairs and a model number m, only the pairs that model wears."""
        if self.colour_pairs:
            k = None if m is None else m - self.first_model + 1
            return [(t, b) for t, b, n in self.colour_pairs if k is None or k <= n]
        if self.bottom in FIXED_BOTTOMS or self.shot == "top":
            return [(t, None) for t in self.top_colours]
        if self.pairing == "match" or not self.bottom_colours:  # a white top goes with WHITE_TOP_BOTTOM
            return [(t, WHITE_TOP_BOTTOM if is_white(t, self.fabric) else t) for t in self.top_colours]
        return [(t, b) for t in self.top_colours for b in self.bottom_colours]

    def base_key(self, top: str, bottom: Optional[str]) -> Tuple[str, Optional[str]]:
        if self.one_base:
            return ("light", "light" if bottom else None)
        tb = base_of(colour_hex(top, self.fabric))
        if bottom == WHITE_TOP_BOTTOM and is_white(top, self.fabric):
            return (tb, tb)  # a white top's black bottom comes from the same light shoot: no extra AI photo
        return (tb, base_of(colour_hex(bottom, self.bfab)) if bottom else None)

    def bases(self, m: Optional[int] = None) -> List[Tuple[str, Optional[str]]]:
        """The base shoots each model (or model m) needs, the most used first (that one is made from the text)."""
        count: Dict[Tuple[str, Optional[str]], int] = {}
        for t, b in self.pairs(m):
            k = self.base_key(t, b)
            count[k] = count.get(k, 0) + 1
        return sorted(count, key=lambda k: (-count[k], str(k)))

    def retouch_poses(self) -> List[str]:
        """Poses whose base photo gets an edit pass that takes drawstrings and pockets off plain bottoms:
        Gemini draws them on kids' shorts and pants whatever the text says (the back view shows neither)."""
        if not self.retouch_bottoms or self.provider != "gemini" or self.bottom not in L.POSITIVE_BOTTOMS:
            return []
        return [p for p in self.poses if p != "back"]

    def model_numbers(self) -> List[int]:
        return list(range(self.first_model, self.first_model + int(self.models)))

    def plan(self) -> dict:
        n, poses = int(self.models), len(self.poses)
        edits = len(self.retouch_poses())  # the retouch pass is one more AI image per such pose
        if self.colour_pairs:  # pairs on only some of the models
            ai = sum(len(self.bases(m)) for m in self.model_numbers()) * (poses + edits)
            photos = sum(n for _, _, n in self.colour_pairs) * poses
        else:
            ai, photos = n * len(self.bases()) * (poses + edits), n * len(self.pairs()) * poses
        mb = ai * BASE_MB + photos * (JPG_MB + (PNG_MB if self.keep_png else 0))
        return {"models": n, "bases_per_model": len(self.bases()), "colour_pairs": len(self.pairs()),
                "poses": poses, "ai_images": ai, "recoloured_images": photos, "disk_gb": round(mb / 1000, 1)}


def base_dir(name) -> str:
    t, b = name
    return f"{t}-{b}" if b else t


def pair_dir(top: str, bottom: Optional[str]) -> str:
    return slug(top) + ("__" + slug(bottom) if bottom else "")


def judge_print(photo, art, mask, place: str, adj: dict, checker=None) -> dict:
    """Is this printed photo right? The firm rules on the measured placement (jagimage.prints.rule_check)
    decide first; the AI checker, when given, also looks at the photo (and is told the measurements).
    {ok, issues, dx, dy, scale (the fix), measured}"""
    from .prints import measure, rule_check

    info = measure(mask, art, place, **adj)
    issues, fix = rule_check(info, place)
    ok = not issues
    if checker:
        ai = checker(photo, art, info)
        issues += [i for i in ai["issues"] if i not in issues]
        ok = ok and ai["ok"]
        if fix == {"dx": 0.0, "dy": 0.0, "scale": 1.0}:  # the rules found nothing to move: the AI's fix
            fix = {k: ai[k] for k in ("dx", "dy", "scale")}
    return {"ok": ok, "issues": issues, **fix, "measured": info}


def backdrop(spec: Spec, pose: str) -> str:
    if spec.background:  # a marketplace guide's backdrop, for every pose
        return f"Plain seamless {spec.background.rstrip('.')} studio background, soft even shadow-free lighting."
    return L.BACKGROUNDS.get(pose, L.GREY)


def prompt(spec: Spec, model_no: int, base, pose: str, reference: str) -> str:
    """reference: 'none' (first base front: new face from the text), 'face' (other base fronts:
    image 1 is the model's first front), 'front' (poses of the first base: image 1 is this base's
    front), 'face+front' (poses of other bases: image 1 face, image 2 this base's front)."""
    tname, thex = TOP_BASES[base[0]]
    kids = L.is_kids(spec.category)
    top_text = L.GARMENTS[spec.top].replace("{COLOUR}", tname)
    top_only = spec.shot == "top"
    positive = spec.provider == "gemini" and spec.bottom in L.POSITIVE_BOTTOMS  # Gemini draws what the text names
    garments = {**L.GARMENTS, **(L.POSITIVE_BOTTOMS if positive else {})}
    if spec.bottom in FIXED_BOTTOMS:
        bottom_text, bcol = FIXED_BOTTOMS[spec.bottom], None
    elif top_only:  # the bottom only completes the look: the AI makes it in the chosen colour
        name = spec.bottom_colour if not HEX.match(spec.bottom_colour) else "solid colour " + spec.bottom_colour.upper()
        bottom_text, bcol = garments[spec.bottom].replace("{COLOUR}", name.lower()), None
    else:
        bname, bhex = BOTTOM_BASES[base[1]]
        bottom_text, bcol = garments[spec.bottom].replace("{COLOUR}", bname), (bname, bhex)
    shoes = "small white kids sneakers" if kids else "clean white low-top sneakers"
    lines = {"none": "", "face": L.REFERENCE_LINE, "front": L.FRONT_ONLY_LINE,
             "face+front": L.REFERENCE_LINE + L.MATCH_FRONT_LINE}
    colours = f"top {tname} ({thex})" + (f", bottom {bcol[0]} ({bcol[1]})" if bcol else "")
    rules = L.RULES.replace("Exact garment colour {COLOUR} ({HEX}),", f"Exact garment colours: {colours}, each")
    pose_text = spec.pose_texts()[pose]
    if CLOSE_UP.search(pose_text):  # a guide's detail shot (e.g. the neckline): framed as the pose says, not head to toe
        rules = rules.replace(FULL_FRAMING, CLOSE_FRAMING)
        lines = {k: v.replace("Create a new full-body photo. ", "Create a new close-up photo. ") for k, v in lines.items()}
    elif top_only:  # every part of the instruction says the same framing: nothing asks for legs or a full body
        rules = rules.replace(FULL_FRAMING, TOP_ONLY_FRAMING)
        lines = {k: v.replace("Create a new full-body photo. ", TOP_ONLY_NEW_PHOTO) for k, v in lines.items()}
        pose_text = pose_text.replace("weight slightly on one leg", "relaxed natural posture")
    text = (f"{lines[reference]}Real photograph of a real, living human model (not a mannequin, dummy, doll or 3D figure): "
            f"{faces.describe(spec.category, model_no, hair=reference == 'none')}, {pose_text}. "
            f"Wearing {top_text}, fabric: {L.FABRIC_TEXT[spec.fabric][L.FABRIC_KIND[spec.top]]}; "
            f"with {bottom_text}" + (f", fabric: {L.FABRIC_TEXT[spec.fabric]['bottom']}" if bcol else "") +
            ("" if top_only else f"; and {shoes}") +
            (". The top and the bottom are two clearly different colours, exactly as stated. " if bcol else ". ") +
            f"{backdrop(spec, pose)} {rules}")
    hands = spec.hands_free and pose not in NO_HANDS_FREE
    if hands:
        text += f" {HANDS_FREE}."
    text += f" {ONE_IMAGE}"
    if spec.extra:
        text += f" Also: {spec.extra.rstrip('.')}."
    if kids:
        text += L.KIDS_RULE
    text += HAIR_RULE
    product = "Trackpants" if spec.bottom == "Trackpants" else spec.top
    negative = L.build_negative(spec.category, product, pose)
    if not top_only:  # the top's hem over the waistband in every pose, as the shop sells them
        text += UNTUCKED_RULE
        negative += ", top tucked into the pants or shorts, waistband over the top"
    if spec.bottom == "Trackpants":
        text += L.JOGGER_RULE
    if positive:
        text += L.POSITIVE_BOTTOM_RULE
    elif spec.bottom in ("Shorts", "Trackpants"):  # plain bottoms: no drawstring, no pockets
        text += L.PLAIN_BOTTOM_RULE
        negative += ", " + L.NEGATIVE_PLAIN_BOTTOM
    if hands:
        negative += ", " + HANDS_FREE_AVOID
    if top_only:
        negative = negative.replace("cut-off feet, ", "") + ", full body shot, feet or shoes in frame, knees in frame"
    text = text + " Avoid: " + negative + ", " + NOT_MANNEQUIN + (", top and bottom in the same colour." if bcol else ".")
    if positive:  # nothing that names a pocket either, except a garment's own (a hoodie's kangaroo pocket)
        for phrase in (", no pocket", ", not in pockets", "chest pocket, ", "hands in pockets, "):
            text = text.replace(phrase, "")
    return child_safe(text) if kids and spec.provider == "openai" else text


def child_safe(text: str) -> str:
    """OpenAI's safety system refuses some children's photo requests (seen: a 4-5 year old girl in a tee and
    shorts) when the wording describes the child's body closely. For OpenAI, children are described plainly
    as a fully dressed child in a family-friendly clothing catalogue: no body proportions, lips or cheeks,
    and no adult-body words in the avoid list. The face stays the same through the reference photo."""
    for old, new in CHILD_SAFE:
        text = text.replace(old, new)
    return CHILD_SAFE_INTRO + text


CHILD_SAFE_INTRO = ("Wholesome, family-friendly children's clothing catalogue photo for an e-commerce store; "
                    "the child is fully dressed in the everyday clothes described. ")
CHILD_SAFE = [
    (", toddler proportions: baby face, large head relative to a small short body (head about one fifth of body height), "
     "short chubby limbs", ", a happy young child"),
    (", full little lips", ""), (", thin lips", ""), (", cupid's-bow lips", ""), (", rosy cheeks", ""),
    ("round face with chubby cheeks", "round face"),
    (L.NEGATIVE_KIDS, "adult, teenager, older child"),
    (L.KIDS_RULE, " The child looks about 4-5 years old."),
]


BILLING_PAGES = {"gemini": "AI Studio > Billing", "openai": "platform.openai.com > Billing", "meta": "dev.meta.ai > Billing"}
ENGINE_NAMES = {"gemini": "Gemini", "openai": "OpenAI", "meta": "Meta", "runpod": "RunPod"}


def account_problem(exc: Exception, provider: Optional[str] = None) -> Optional[str]:
    """A plain explanation when an AI error means the account can't make images at all, so the
    batch stops at once instead of failing every model. None for errors worth retrying. It names the
    engine and ends with the engine's own words, which tell e.g. a missing payment method from no credit."""
    text = f"{type(exc).__name__} {exc}".lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    who = ENGINE_NAMES.get(provider, "AI")
    said = f' {who} said: "{" ".join(str(exc).split())[:240]}"'
    if "runpod token not accepted" in text:
        return "The RunPod pod refused the token. Save the same token the pod was started with, then start again."
    if "runpod pod unreachable" in text:
        return ("The RunPod pod isn't answering: check it is running in the RunPod console (Pods) and the address "
                "ends in -8000.proxy.runpod.net, then start again: finished photos are kept.")
    if "credits are depleted" in text or "insufficient_quota" in text or "billing" in text and "402" in text or "error 402" in text or code == 402:
        where = BILLING_PAGES.get(provider) or ", ".join(f"{ENGINE_NAMES[k]}: {v}" for k, v in BILLING_PAGES.items())
        return (f"The {who} account is out of credits or billing isn't set up. Add credits ({where}), then start "
                "the batch again: finished photos are kept." + said)
    if code in (401, 403) or "api key not valid" in text or "invalid_api_key" in text or "incorrect api key" in text \
            or "permission_denied" in text or "error 401" in text or "error 403" in text:
        return (f"The {who} API key was refused (wrong, expired or without access to this model). Save a working "
                "key and start again." + said)
    return None


def is_rate_limit(exc: Exception) -> bool:
    text = f"{exc}".lower()
    return getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429 or "rate limit" in text \
        or ("resource_exhausted" in text and "credits" not in text)


def backdrop_problem(lab) -> Optional[str]:
    """Why an AI photo isn't on the plain light grey studio backdrop (the AI sometimes ignores it and
    puts the model outdoors), or None. Reads the top edge and the upper sides, where only backdrop is:
    tested studio photos stay below 0.008 colourfulness and above 0.84 lightness there."""
    import numpy as np

    h = lab.shape[0]
    edge = np.concatenate([lab[:12].reshape(-1, 3), lab[:int(h * 0.7), :12].reshape(-1, 3), lab[:int(h * 0.7), -12:].reshape(-1, 3)])
    colourful = float(np.percentile(np.hypot(edge[:, 1], edge[:, 2]), 90))
    light = float(np.median(edge[:, 0]))
    if colourful > BACKDROP_MAX_CHROMA or light < BACKDROP_MIN_LIGHT:
        return f"the background isn't the plain studio backdrop (colour {colourful:.3f}, lightness {light:.2f})"
    return None


def garment_problem(lab, masks: dict, base) -> Optional[str]:
    """Why the garments in an AI base photo can't be recoloured cleanly, or None.
    - More than one figure (Flash once drew a 4-panel product infographic): the top's mask then
      falls into several similar pieces; every real photo checked has one.
    - A garment much paler than its base colour (lilac trousers that came out nearly white): its
      edge can't be told from the light backdrop and the recolour leaves a ragged fringe."""
    import numpy as np
    from scipy import ndimage

    from .recolor import hex_to_lab

    top = masks["top"] > 0.5
    # a contrast zip or placket cuts the top into halves a few px apart; separate figures stand far apart
    joined = ndimage.binary_dilation(top, iterations=max(2, round(ZIP_GAP * top.shape[1] / 2)))
    pieces, n = ndimage.label(joined)
    if n > 1:
        sizes = np.sort(ndimage.sum(top, pieces, index=np.arange(1, n + 1)))[::-1]
        if (sizes >= 0.3 * sizes[0]).sum() > 1:
            return "the photo shows more than one figure (a collage or several views), not one model"
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    for part, key, bases in (("top", base[0], TOP_BASES), ("bottom", base[1], BOTTOM_BASES)):
        if part not in masks or not key:
            continue
        core = masks[part] > 0.95
        if not core.any():
            continue
        want = float(np.hypot(*hex_to_lab(bases[key][1])[1:]))
        got = float(np.median(chroma[core]))
        if got < MIN_BASE_STRENGTH * want:
            return f"the {part} came out too pale for a clean recolour ({got:.3f} colour strength, needs {MIN_BASE_STRENGTH * want:.3f})"
    return None


_LEDGER_LOCK = threading.Lock()
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,150}$")
DRIVE_LIST_LOCK = threading.Lock()  # guards each batch's _drive.txt (uploads listed by the Drive sync, redone photos removed)


class ModelLedger:
    """Which model numbers each category has used, across every batch in the output folder, so a
    face is never used twice: model n always has the same look, and each n belongs to one batch.
    Kept in <out>/_models.json; batches made before the ledger existed are read from their spec.json."""

    def __init__(self, out):
        self.path = Path(out) / "_models.json"
        self.out = Path(out)

    def ranges(self, category: str) -> List[dict]:
        rows = json.loads(self.path.read_text()) if self.path.exists() else []
        seen = {(r["batch"], r["first"]) for r in rows}
        for spec_file in self.out.glob("*/spec.json"):
            if spec_file.parent.name.startswith("_"):
                continue
            sp = json.loads(spec_file.read_text())
            key = (sp.get("name", spec_file.parent.name), sp["first_model"])
            if key not in seen:
                rows.append({"category": sp["category"], "batch": key[0], "first": sp["first_model"],
                             "last": sp["first_model"] + int(sp["models"]) - 1})
        return [r for r in rows if r["category"] == category]

    def next_free(self, category: str) -> int:
        return max([r["last"] for r in self.ranges(category)] + [0]) + 1

    def clash(self, spec: "Spec") -> Optional[dict]:
        """Another batch already using any of this spec's model numbers (re-running the same batch is fine)."""
        if spec.same_models:
            return None  # reusing faces on purpose (e.g. the same models in another top style)
        first, last = spec.first_model, spec.first_model + int(spec.models) - 1
        for r in self.ranges(spec.category):
            if r["batch"] != spec.name and r["first"] <= last and first <= r["last"]:
                return r
        return None

    def reserve(self, spec: "Spec"):
        row = {"category": spec.category, "batch": spec.name, "first": spec.first_model,
               "last": spec.first_model + int(spec.models) - 1}
        with _LEDGER_LOCK:  # batches may start at the same time: read-modify-write under a lock,
            rows = json.loads(self.path.read_text()) if self.path.exists() else []
            if row not in rows:   # and replace the file in one step so it's never half-written
                rows.append(row)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(f".{self.path.name}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(rows, indent=1))
                tmp.replace(self.path)


class Batch:
    """Runs a Spec: AI base shoots (Gemini/OpenAI) then exact-hex recolours. Safe to re-run: files
    that exist are skipped, so an interrupted batch continues where it stopped."""

    def __init__(self, spec: Spec, out="output/studio", generate: Optional[Callable[[str, List[str]], bytes]] = None,
                 workers=4, retries=3, on_event: Optional[Callable[[dict], None]] = None,
                 export: Optional[str] = None, export_what="jpg", redo: bool = False,
                 prints: Optional[Callable[[str, str, str], list]] = None, checker: Optional[Callable] = None,
                 inspector: Optional[Callable] = None, restore: Optional[Callable[[Path], bytes]] = None):
        """export: a second folder (e.g. your Google Drive folder) that gets a copy of each finished
        photo as <batch>/<Top>__<Bottom>/<model>_<pose>.jpg; export_what: jpg | png | both."""
        self.spec = spec
        self.export = Path(export).expanduser() if export else None
        if export_what not in ("jpg", "png", "both"):
            raise ValueError("export_what must be jpg, png or both")
        self.export_what = export_what
        self.root = Path(out) / spec.name
        self.generate = generate          # (prompt, reference paths) -> image bytes
        # batch mode: the engine's run_batch([(key, prompt, refs)], ...) -> {key: bytes | Exception}
        self.batch_runner = getattr(generate, "run_batch", None) if spec.batch else None
        self.workers = workers
        self.retries = retries
        self.redo = redo  # recolour again even where a finished photo exists (after recolouring improves)
        self.on_event = on_event or (lambda e: None)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.fatal = None  # set when the AI account itself can't be used (no credits, bad key)
        self.counts = {"ai_done": 0, "ai_skipped": 0, "recoloured": 0, "recolour_skipped": 0, "errors": 0, "qc_flagged": 0,
                       "qc_fixed": 0, "qc_retouched": 0, "retouch_skipped": 0, "qc_held": 0, "base_rejected": 0, "printed": 0, "print_fixed": 0, "print_flagged": 0}
        # prints(category, top style, colour) -> [(print name, PNG path)]: the shop's prints for that shirt
        self.prints = prints
        self.checker = checker  # checker(printed photo, artwork) -> {ok, issues, dx, dy, scale}: the AI print checker
        self._inspector = inspector  # the QC agent's eyes: ask(message, jpeg bytes) -> {ok, problems}
        self.restore = restore  # restore(base photo path) -> its bytes from Google Drive, for an archived base
        if hasattr(generate, "set_usage_log"):  # Google's billed token counts land next to the batch
            generate.set_usage_log(self.root / "_usage.jsonl")

    @property
    def inspector(self):
        if self._inspector is None:
            from .qcagent import default_ask

            self._inspector = default_ask()
        return self._inspector

    # ---- paths
    def model_dir(self, m: int) -> Path:
        return self.root / f"m{m:04d}"

    def base_path(self, m: int, base, pose: str) -> Path:
        if self.spec.base_from:  # a twin batch reads the base photos of the batch it was made from
            return self.root.parent / self.spec.base_from / f"m{m:04d}" / "_base" / base_dir(base) / f"{pose}.png"
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
        """stage: all | bases (AI only) | recolor (no AI calls) | prints (prints on the finished fronts, no AI) |
        qc (the QC agent re-checks the bases already made, then all: rejected bases are made again)."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "spec.json").write_text(json.dumps(asdict(self.spec), indent=2))
        if stage == "prints":
            self.run_prints()
            return self.counts
        if self.spec.base_from:  # a twin only recolours the base photos of its source batch
            stage = "recolor"
        if stage == "qc":  # the QC agent looks at the bases already made; rejected ones are made again
            if self.generate is None:
                raise RuntimeError("Re-checking bases needs the image engine (its API key)")
            self.requalify()
            try:
                return self.run("all")
            finally:
                self._restore_rejected()
        if stage != "recolor" and self.generate is None:
            raise RuntimeError("No image engine: pass generate=, or use stage='recolor'")
        if stage != "recolor" and self.spec.batch:
            if not self.batch_runner:
                raise RuntimeError("This engine has no batch mode")
            self._run_batched()
            if stage == "bases" or self.stop.is_set():
                return self.counts
            stage = "recolor"  # every base is made (or has failed): recolour them all
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            list(pool.map(lambda m: self.run_model(m, stage), self.spec.model_numbers()))
        if stage != "bases" and not self.stop.is_set():
            self._run_twins()
            if not self.stop.is_set() and not self.fatal:  # every photo made: its base photos may go to Drive
                (self.root / "_complete").write_text(json.dumps({"time": round(time.time()), "counts": self.counts}))
        return self.counts

    def _run_twins(self):
        """The same models recoloured into each of also_fabrics' colours, in a twin batch per fabric."""
        for fabric in self.spec.also_fabrics:
            if self.stop.is_set():
                return
            twin = Batch(fabric_twin(self.spec, fabric), out=self.root.parent, generate=None, workers=self.workers,
                         retries=self.retries, on_event=lambda e: self._event(e.pop("kind"), **e), redo=self.redo,
                         inspector=self._inspector, restore=self.restore)  # its photos count in this batch's totals
            twin.stop = self.stop
            twin.run("recolor")

    def requalify(self):
        """The QC agent checks every base photo already made. A rejected one is deleted (with its masks) so the
        run makes it again; when a front is rejected, the other poses of that shoot go too, so they match it."""
        from PIL import Image

        from .qcagent import check_base
        from .recolor import load_mask

        poses = ["front_main"] + [p for p in self.spec.poses if p != "front_main"]
        for m in self.spec.model_numbers():
            for base in self.spec.bases(m):
                for pose in poses:
                    if self.stop.is_set():
                        return
                    path = self.base_path(m, base, pose)
                    if not path.exists():
                        continue
                    masks = self._saved_masks(path.parent, pose, load_mask)
                    with Image.open(path) as img:
                        res = check_base(self.inspector, img.convert("RGB"), self.spec, pose, masks)
                    (path.parent / f"{pose}_base_qc.json").write_text(json.dumps(res, indent=1))
                    if res["ok"] is not False:
                        continue
                    self._event("base_rejected", model=m, path=str(path), problems=res["problems"])
                    for p in (poses if pose == "front_main" else [pose]):  # kept aside until a new one is made
                        for f in list(path.parent.glob(f"{p}.png")) + list(path.parent.glob(f"{p}_*mask*")):
                            f.replace(f.with_name(f.name + ".rejected"))
                    if pose == "front_main":
                        break

    def _restore_rejected(self):
        """A rejected base whose new one couldn't be made (no credits, stopped) is put back, so nothing is lost;
        the next qc run looks at it again. Where a new one was made, the rejected copy is deleted."""
        for old in self.root.glob("m[0-9][0-9][0-9][0-9]/_base/*/*.rejected"):
            new = old.with_name(old.name[:-len(".rejected")])
            if new.exists():
                old.unlink()
            else:
                old.replace(new)

    def _run_batched(self):
        """Batch mode: each round sends every base photo whose reference photos already exist as one
        Gemini or OpenAI batch job (half price). Round 1 makes each model's first front, round 2 the other fronts
        and the first base's poses, round 3 the remaining poses. Photos the AI got wrong go into the
        next round, up to `retries` times."""
        poses = ["front_main"] + [p for p in self.spec.poses if p != "front_main"]
        tries: Dict[str, int] = {}
        seen = set()
        while not self.stop.is_set():
            todo = []
            for m in self.spec.model_numbers():
                bases = self.spec.bases(m)
                first = self.identity_front(m, bases)
                for base in bases:
                    for pose in poses:
                        path = self.base_path(m, base, pose)
                        key = f"m{m:04d}|{base_dir(base)}|{pose}"
                        if path.exists() or archived(path):  # archived: on Drive, never paid for twice
                            if key not in seen and key not in tries:
                                self._event("ai_skipped", model=m, path=str(path))
                            seen.add(key)
                            continue
                        req = self._request(m, base, pose, path, first)
                        if req and tries.get(key, 0) < self.retries:
                            todo.append((key, m, base, pose, path) + req)
            if not todo:
                break
            n_round = 1 + max(tries.values(), default=0)
            self._event("batch", round=n_round, images=len(todo), state="sending")
            results = self.batch_runner(
                [(key, text, refs) for key, m, base, pose, path, text, refs in todo],
                on_status=lambda state, n=len(todo): self._event("batch", images=n, state=state),
                stop=self.stop, state_file=self.root / "_batch_jobs.json")
            for key, m, base, pose, path, text, refs in todo:
                tries[key] = tries.get(key, 0) + 1
                got = results.get(key)
                if got is None:
                    continue  # stopped, or the job is still with Google: picked up on the next start
                if isinstance(got, Exception):
                    fatal = account_problem(got, self.spec.provider)
                    if fatal:
                        self.fatal = fatal
                        self.stop.set()
                        self._event("errors", model=m, path=str(path), error=fatal)
                        return
                    if tries[key] >= self.retries:
                        self._event("errors", model=m, path=str(path), error=f"{type(got).__name__}: {got}")
                    continue
                try:
                    self._accept(path, got, base, m, pose)
                except Exception as exc:
                    if tries[key] >= self.retries:
                        self._event("errors", model=m, path=str(path), error=f"{type(exc).__name__}: {exc}")
            if all(results.get(t[0]) is None for t in todo):
                break  # nothing came back (stopped): don't spin

    def run_model(self, m: int, stage="all"):
        spec = self.spec
        md = self.model_dir(m)
        md.mkdir(parents=True, exist_ok=True)
        (md / "model.json").write_text(json.dumps({"model": m, "look": faces.look(spec.category, m),
                                                    "description": faces.describe(spec.category, m)}, indent=2))
        bases = spec.bases(m)
        first_front = self.identity_front(m, bases)
        poses = ["front_main"] + [p for p in spec.poses if p != "front_main"]
        for base in bases:
            for pose in poses:
                if self.stop.is_set():
                    return
                path = self.base_path(m, base, pose)
                if stage != "recolor" and not self._shoot(m, base, pose, path, first_front):
                    continue
                if stage != "bases" and (path.exists() or self._restore(m, base, pose, path)):
                    self._recolour_base(m, base, pose)

    def _restore(self, m, base, pose, path: Path) -> bool:
        """A base photo moved to Google Drive comes back when a photo made from it is missing (a colour or combo
        added later, or a redo); the Drive sync moves it out again once everything is uploaded."""
        if not archived(path) or self.restore is None:
            return False
        if all(self._done(self.out_path(m, t, b, pose), pose)
               for t, b in self.spec.pairs(m) if self.spec.base_key(t, b) == base):
            return False
        try:
            data = self.restore(path)
        except Exception as exc:
            self._event("errors", model=m, path=str(path), error=f"base on Drive: {type(exc).__name__}: {exc}"[:300])
            return False
        tmp = path.with_suffix(".restore.png")
        tmp.write_bytes(data)
        tmp.replace(path)
        Path(str(path) + ARCHIVED).unlink(missing_ok=True)
        (self.root.parent / path.relative_to(self.root.parent).parts[0] / "_complete").unlink(missing_ok=True)
        return True

    def face_file(self, m: int) -> Path:
        """Model m's face for the whole category: the first front ever made for this model in any
        batch, so the same person can wear every top style (half sleeve, hoodie, ...)."""
        return self.root.parent / "_faces" / self.spec.category / f"m{m:04d}.png"

    def identity_front(self, m: int, bases) -> Path:
        """The photo that defines model m's face: the first base front made for this model in this
        batch (adding colours later keeps the same person), else the model's face from another style
        of the same category, else the front of the base this run makes first (made from the text)."""
        made = [p for p in (self.model_dir(m) / "_base").glob("*/front_main.png")]
        face = self.face_file(m)
        if made:
            first = min(made, key=lambda p: p.stat().st_mtime)
            if not face.exists():
                self._register_face(m, first)
            return first
        if face.exists():
            return face
        return self.base_path(m, bases[0], "front_main")

    def _register_face(self, m: int, front: Path):
        face = self.face_file(m)
        if face.exists():
            return
        import shutil

        face.parent.mkdir(parents=True, exist_ok=True)
        tmp = face.with_name(f".{face.name}.{threading.get_ident()}.tmp")
        shutil.copy2(front, tmp)
        tmp.replace(face)

    def _request(self, m, base, pose, path: Path, first_front: Path):
        """(prompt, reference paths) for one base photo, or None while a reference isn't made yet."""
        is_first = path.parent == first_front.parent
        own_front = path.parent / "front_main.png"
        if pose == "front_main":
            ref, refs = ("none", []) if is_first else ("face", [str(first_front)])
        else:
            ref, refs = ("front", [str(own_front)]) if is_first else ("face+front", [str(first_front), str(own_front)])
        if any(not Path(r).exists() for r in refs):
            return None
        text = prompt(self.spec, m, base, pose, ref)
        product = self.bottom_ref()
        if product:  # a photo of the shop's own shorts or pants: the AI copies a picture better than words
            refs = refs + [str(product)]
            text += BOTTOM_REF_LINE.format(n=len(refs), item=BOTTOM_WORD.get(self.spec.bottom, "bottoms"))
        return text, refs

    def bottom_ref(self) -> Optional[Path]:
        """The shop's product photo of this batch's bottom (uploaded once in Settings), for full outfit shots."""
        if self.spec.shot != "outfit" or self.spec.bottom in FIXED_BOTTOMS or not L.is_kids(self.spec.category):
            return None  # the shop's photos are of its kids' shorts and pants
        f = bottom_ref_path(self.root.parent, self.spec.bottom)
        return f if f.exists() else None

    def _accept(self, path: Path, data: bytes, base, m, pose):
        """Keep an AI image only if it used the base colours (else raise, so it's asked for again)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.png")
        tmp.write_bytes(data)
        if pose in self.spec.retouch_poses() and self._needs_retouch(tmp):  # Gemini draws drawstrings and pockets anyway: edit them out
            try:
                tmp.write_bytes(self.generate(RETOUCH_BOTTOMS, [str(tmp)]))
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        try:
            self._check_base(tmp, base)
            if self.spec.qc_agent:
                self._qc_base(tmp, path, pose, m)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        if pose == "front_main":
            self._register_face(m, path)  # no-op once the model has a face in this category
        self._event("ai_done", model=m, path=str(path), base=base_dir(base), pose=pose)

    def _shoot(self, m, base, pose, path: Path, first_front: Path) -> bool:
        """Make one base photo if missing. False if it couldn't be made."""
        if path.exists() or archived(path):
            self._event("ai_skipped", model=m, path=str(path))
            return True
        req = self._request(m, base, pose, path, first_front)
        if req is None:
            self._event("errors", model=m, path=str(path), error="its reference front is missing")
            return False
        text, refs = req
        last = None
        for attempt in range(self.retries):
            if self.stop.is_set():
                return False
            try:
                self._accept(path, self.generate(text, refs), base, m, pose)
                return True
            except Exception as exc:  # API errors, rate limits, wrong colours: retry with backoff
                last = exc
                fatal = account_problem(exc, self.spec.provider)
                if fatal:  # no credits / bad key: every other request would fail the same way
                    self.fatal = fatal
                    self.stop.set()
                    self._event("errors", model=m, path=str(path), error=fatal)
                    return False
                wait = 20 * (attempt + 1) if is_rate_limit(exc) else 2 ** attempt
                time.sleep(min(60, wait) if attempt + 1 < self.retries else 0)
        self._event("errors", model=m, path=str(path), error=f"{type(last).__name__}: {last}")
        return False

    def _needs_retouch(self, tmp: Path) -> bool:
        """With the QC agent the drawstring edit (a full-price AI call) is only made when the agent sees a
        drawstring or pocket; without it, or when it can't tell, every photo is edited as before."""
        if not self.spec.qc_agent:
            return True
        from PIL import Image

        from .qcagent import BOTTOMS_POINT, inspect

        with Image.open(tmp) as img:
            res = inspect(self.inspector, img, [BOTTOMS_POINT])
        if res["ok"] is True:
            self._event("retouch_skipped", path=str(tmp))
            return False
        return True

    @staticmethod
    def _saved_masks(folder: Path, pose: str, load_mask):
        """(top, bottom) masks saved next to a base photo, or None."""
        t, b = folder / f"{pose}_top_mask.png", folder / f"{pose}_bottom_mask.png"
        if not t.exists():
            return None
        return load_mask(t), (load_mask(b) if b.exists() else None)

    def _qc_base(self, tmp: Path, path: Path, pose: str, m: int):
        """The QC agent looks at a new base photo; a fault recolouring can't fix sends it back to the AI."""
        from PIL import Image

        from .qcagent import check_base
        from .recolor import MaskError, load_mask

        masks = self._saved_masks(path.parent, pose, load_mask)
        with Image.open(tmp) as img:
            res = check_base(self.inspector, img.convert("RGB"), self.spec, pose, masks)
        (path.parent / f"{pose}_base_qc.json").write_text(json.dumps(res, indent=1))
        if res["ok"] is False:
            self._event("base_rejected", model=m, path=str(path), problems=res["problems"])
            raise MaskError("QC agent: " + "; ".join(res["problems"])[:400])

    def _check_base(self, path: Path, base):
        import numpy as np
        from PIL import Image

        from .recolor import MASK_VERSION, MaskError, garment_mask, rgb_to_lab, save_mask, split_outfit

        lab = rgb_to_lab(np.asarray(Image.open(path).convert("RGB")))
        why = backdrop_problem(lab)
        if why:
            raise MaskError(why)
        top, _ = garment_mask(lab, TOP_BASES[base[0]][1])
        if not base[1] and self.spec.bottom not in FIXED_BOTTOMS:
            top = self._trim_fixed_bottom(lab, top, base)
        masks = {"top": top}
        if base[1]:
            bottom, _ = garment_mask(lab, BOTTOM_BASES[base[1]][1], shoes=True)  # tinted shoes at the hem stay out
            # where the masks meet, each pixel belongs to the garment it's more part of, so no seam
            # pixel is left to neither (the waistband shadow keeping the base colour)
            bottom = np.where(bottom > top, bottom, np.minimum(bottom, 1 - top))
            top = np.where(top >= bottom, top, np.minimum(top, 1 - bottom))
            top, bottom = split_outfit(lab, top, bottom)  # a sleeve lit by the pants' colour stays sleeve
            masks["top"] = top
            if (bottom > 0.95).sum() < 0.002 * bottom.size:
                raise MaskError("the bottom isn't in its base colour")
            masks["bottom"] = bottom
        why = garment_problem(lab, masks, base)
        if why:
            raise MaskError(why)
        stem = path.name.replace(".tmp.png", "")
        for k, mk in masks.items():  # saved next to the base; edit them to fix an edge
            save_mask(path.parent / f"{stem}_{k}_mask.png", mk)
        (path.parent / f"{stem}_mask_version.txt").write_text(str(MASK_VERSION))

    def _trim_fixed_bottom(self, lab, top, base):
        """Top-only shots: a black or grey bottom can carry a faint tint of the base green near the
        seam, and the mask's shadow growth then reaches into it. Pixels in the bottom's own colour that
        connect to the bottom are taken out again; dark shadow inside the top is kept."""
        import numpy as np
        from scipy import ndimage

        from .recolor import hex_to_lab

        # the bottom as photographed (black shorts come out dark grey): sampled just below the top
        ys, xs = np.nonzero(top > 0.5)
        if not len(ys):
            return top
        x0, x1 = np.percentile(xs, 30), np.percentile(xs, 70)
        y0 = ys.max() + 10
        below = lab[y0:y0 + 50, int(x0):int(x1)].reshape(-1, 3)
        want = hex_to_lab(colour_hex(self.spec.bottom_colour, self.spec.fabric))
        if not len(below):
            return top
        seen = np.median(below, axis=0)
        if np.linalg.norm(seen - want) > 0.25:  # what's below isn't the bottom (e.g. bare legs): leave it
            return top
        like = np.linalg.norm(lab - seen, axis=-1) < BOTTOM_LIKE
        # never the base colour itself, and only near the hem: a navy bottom and the top's dark folds are
        # alike, and following them would reach up the side folds to the sleeves (left in bottle green)
        like &= ~_base_hued(lab, hex_to_lab(TOP_BASES[base[0]][1]))
        like[:max(0, ys.max() - int(BOTTOM_TRIM_BAND * lab.shape[0]))] = False
        seed = like & (top < 0.5)
        if not seed.any():
            return top
        bottomish = ndimage.binary_propagation(seed, mask=like)
        trimmed = (top > 0.5) & ~bottomish
        return np.minimum(top, ndimage.gaussian_filter(trimmed.astype(np.float32), 0.8))

    def masks(self, m, base, pose):
        from PIL import Image

        from .recolor import MASK_VERSION, load_mask

        src = self.base_path(m, base, pose)
        tm, bm = src.parent / f"{pose}_top_mask.png", src.parent / f"{pose}_bottom_mask.png"
        ver = src.parent / f"{pose}_mask_version.txt"
        old = not ver.exists() or ver.read_text().strip() != str(MASK_VERSION)  # made before masks improved
        if old or not tm.exists() or (base[1] and not bm.exists()):
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
        todo = [(t, b) for t, b in spec.pairs(m) if spec.base_key(t, b) == base]
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
            if self._done(path, pose):
                self._event("recolour_skipped", model=m, path=str(path))
                continue
            try:
                self._recolour_one(img, tmask, bmask, base, t, b, path, pose)
                if self.redo:
                    self._forget_upload(path, pose)
                self._event("recoloured", model=m, path=str(path), top=t, bottom=b, pose=pose)
            except Exception as exc:
                self._event("errors", model=m, path=str(path), error=f"recolour: {exc}")

    def _forget_upload(self, path: Path, pose: str):
        """A remade photo is uploaded to Google Drive again (replacing the old one there)."""
        done = self.root / "_drive.txt"
        c = self.spec.canvas
        if not c or not done.exists():
            return
        rel = path.with_name(f"{pose}_{c[0]}x{c[1]}.jpg").relative_to(self.root).as_posix()
        with DRIVE_LIST_LOCK:
            lines = done.read_text().splitlines()
            keep = [x for x in lines if x != rel and not x.startswith(rel + "#")]  # its single top too
            if len(keep) != len(lines):
                done.write_text("".join(f"{x}\n" for x in keep))
            freed = self.root / "_freed.txt"  # remade: it is on the server again until it is uploaded
            if freed.exists():
                left = [x for x in freed.read_text().splitlines() if x != rel]
                freed.write_text("".join(f"{x}\n" for x in left))

    def run_prints(self):
        """Every print the shop has for this category, top style and colour, on every finished front photo
        (front_main, and front_main_white when made): <pose>_print_<print name>.png and its marketplace JPG,
        next to the plain photo. No AI: the artwork is placed exactly (jagimage.prints)."""
        from .prints import PrintError, place_print
        from .recolor import fit_canvas

        if self.prints is None:
            raise RuntimeError("Print folders aren't set up (Settings → Print folders)")
        spec = self.spec
        cache: Dict[str, list] = {}
        for m in spec.model_numbers():
            for base in spec.bases(m):
                for pose in [p for p in ("front_main", "front_main_white") if p in spec.poses]:
                    todo = [(t, b) for t, b in spec.pairs(m) if spec.base_key(t, b) == base]
                    if not todo or not self.base_path(m, base, pose).exists():
                        continue
                    try:
                        img, tmask, bmask = self.masks(m, base, pose)
                    except Exception as exc:
                        self._event("errors", model=m, path=str(self.base_path(m, base, pose)), error=f"mask: {exc}")
                        continue
                    for t, b in todo:
                        if self.stop.is_set():
                            return
                        if t not in cache:
                            try:
                                cache[t] = self.prints(spec.category, spec.top, t)
                            except Exception as exc:  # e.g. Drive can't be read: no use trying every photo
                                raise RuntimeError(f"prints: {exc}") from exc
                        path = self.out_path(m, t, b, pose)
                        plain = None
                        for name, art_path in cache[t]:
                            dest = path.with_name(f"{pose}_print_{slug(name)}.png")
                            jpg = dest.with_name(f"{dest.stem}_{spec.canvas[0]}x{spec.canvas[1]}.jpg") if spec.canvas else None
                            if (dest.exists() or (jpg and jpg.exists())) and not self.redo:
                                continue
                            try:
                                if plain is None:
                                    plain = self._recoloured(img, tmask, bmask, base, t, b)
                                from .prints import open_image
                                art = open_image(str(art_path))
                                out, check = self._print_and_check(plain, tmask, art, m, dest)
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                if spec.keep_png:
                                    out.save(dest)
                                if jpg:
                                    fit_canvas(out, *spec.canvas).save(jpg, quality=95)
                                if check:
                                    self._print_qc(dest, check)
                                if self.redo:
                                    self._forget_upload(dest, dest.stem)
                                self._event("printed", model=m, path=str(dest), top=t, bottom=b, pose=pose, print=name)
                            except (PrintError, OSError) as exc:
                                self._event("errors", model=m, path=str(dest), error=f"print {name}: {exc}")

    def print_targets(self) -> List[dict]:
        """The finished front photos a print can go on: [{model, top, bottom, pose}] (for the preview)."""
        spec, rows = self.spec, []
        for m in spec.model_numbers():
            for pose in [p for p in ("front_main", "front_main_white") if p in spec.poses]:
                for t, b in spec.pairs(m):
                    if self.base_path(m, spec.base_key(t, b), pose).exists():
                        rows.append({"model": m, "top": t, "bottom": b, "pose": pose})
        return rows

    def preview_print(self, target: dict, art, place: str, dx=0.0, dy=0.0, scale=1.0):
        """One printed photo, made in memory only: how the print sits before the whole batch is printed.
        Returns (photo, the shirt's mask)."""
        from .prints import place_print

        m, t, b, pose = target["model"], target["top"], target["bottom"], target["pose"]
        base = self.spec.base_key(t, b)
        img, tmask, bmask = self.masks(m, base, pose)
        plain = self._recoloured(img, tmask, bmask, base, t, b)
        return place_print(plain, tmask, art, place, dx=dx, dy=dy, scale=scale), tmask

    PRINT_FIXES = 2  # the checker may move or resize a print this many times, then it is flagged for a person

    def _print_and_check(self, plain, tmask, art, m, dest):
        """Place the print; with the AI checker on, show it each result and apply its fix (a move or a
        resize) until it is happy, at most PRINT_FIXES times. Returns (photo, check record or None)."""
        from .prints import adjust, place_print

        spec = self.spec
        adj = adjust(spec.print_dx, spec.print_dy, spec.print_scale)
        out = place_print(plain, tmask, art, spec.print_place, **adj)
        if not self.checker:
            return out, None
        record = {"rounds": [], "ok": False}
        for n in range(self.PRINT_FIXES + 1):
            try:
                res = judge_print(out, art, tmask, spec.print_place, adj, self.checker)
            except Exception as exc:  # the checker can't be reached: keep the print, say it wasn't checked
                record.update(error=f"{type(exc).__name__}: {exc}"[:300])
                self._event("errors", model=m, path=str(dest), error=f"print check: {exc}")
                break
            record["rounds"].append({"ok": res["ok"], "issues": res["issues"], "at": dict(adj)})
            moved = abs(res["dx"]) > 0.005 or abs(res["dy"]) > 0.005 or abs(res["scale"] - 1) > 0.01
            if res["ok"] or not moved or n == self.PRINT_FIXES:
                record["ok"] = res["ok"]
                break
            adj = adjust(adj["dx"] + res["dx"], adj["dy"] + res["dy"], adj["scale"] * res["scale"])
            out = place_print(plain, tmask, art, spec.print_place, **adj)
        record["adjust"] = adj
        record["fixed"] = len(record["rounds"]) > 1
        if record["fixed"]:
            self._event("print_fixed", model=m, path=str(dest))
        if not record["ok"]:
            self._event("print_flagged", model=m, path=str(dest), issues=record["rounds"][-1]["issues"] if record["rounds"] else [])
        return out, record

    @staticmethod
    def _print_qc(dest: Path, record: dict):
        """Each printed photo's check, next to it in prints_qc.json (shown in the Print placement tab)."""
        f = dest.parent / "prints_qc.json"
        rows = json.loads(f.read_text()) if f.exists() else {}
        rows[dest.stem] = record
        f.write_text(json.dumps(rows, indent=1))

    def _recoloured(self, img, tmask, bmask, base, top, bottom):
        from .recolor import recolor_image

        spec = self.spec
        out, _, _ = recolor_image(img, TOP_BASES[base[0]][1], colour_hex(top, spec.fabric), mask=tmask, others=bmask if bottom else None)
        if bottom:
            out, _, _ = recolor_image(out, BOTTOM_BASES[base[1]][1], colour_hex(bottom, spec.bfab), mask=bmask, others=tmask)
        return out

    def _recolour_one(self, img, tmask, bmask, base, top, bottom, path: Path, pose):
        import numpy as np

        from .recolor import fit_canvas, quality_check, recolor_image, write_qc

        spec = self.spec
        thex = colour_hex(top, spec.fabric)
        bhex = colour_hex(bottom, spec.bfab) if bottom else None

        def paint(tm, bm):
            out, de_t, _ = recolor_image(img, TOP_BASES[base[0]][1], thex, mask=tm, others=bm if bottom else None)
            qc = {"top": {"colour": top, "hex": thex, "deltaE": round(de_t, 2), "on_shade": de_t < 3}}
            if bottom:
                out, de_b, _ = recolor_image(out, BOTTOM_BASES[base[1]][1], bhex, mask=bm, others=tm)
                qc["bottom"] = {"colour": bottom, "hex": bhex, "deltaE": round(de_b, 2), "on_shade": de_b < 3}
            return out, qc

        def judge(out, tm, bm, qc):
            masks = [tm] + ([bm] if bottom else [])
            bases = [TOP_BASES[base[0]][1]] + ([BOTTOM_BASES[base[1]][1]] if bottom else [])
            check = quality_check(img, out, masks, bases, [thex] + ([bhex] if bottom else []))
            problems = list(check["problems"]) + [f"{k} colour off by ΔE {v['deltaE']}" for k, v in qc.items() if not v["on_shade"]]
            return check, problems

        out, qc = paint(tmask, bmask)
        # the quality check every finished photo gets: colour accuracy, bleeding, base colour left
        check, problems = judge(out, tmask, bmask, qc)
        fixed, held, agent = [], False, None
        if problems and spec.qc_agent:
            from .qcagent import FIX_ROUNDS, fix_photo, inspect, photo_points

            for _ in range(FIX_ROUNDS):  # the QC agent's automatic fixes, then the pixel check again
                def grow(extra, tm=tmask, bm=bmask):
                    # leftover base colour joins the garment it touches most (top unless it is in the bottom)
                    nonlocal tmask, bmask, qc
                    if bottom and bm is not None:
                        in_b = extra & (np.asarray(bm) > 0.05)
                        bmask = np.maximum(bm, in_b.astype(np.float32))
                        extra = extra & ~in_b
                    tmask = np.maximum(tm, extra.astype(np.float32))
                    o, qc = paint(tmask, bmask)
                    return o
                before = (check["bleed_px"], check["base_left_px"])
                out = fix_photo(img, out, check, grow)
                check, problems = judge(out, tmask, bmask, qc)
                fixed.append(f"bleed {before[0]}→{check['bleed_px']} px, base colour {before[1]}→{check['base_left_px']} px")
                if not problems:
                    break
            if problems:  # still counted: the agent's own retouch on the server (no AI call), then the pixel check again
                from .qcagent import local_fix

                def repaint(ms):
                    nonlocal tmask, bmask, qc
                    tmask, bmask = ms[0], (ms[1] if bottom else bmask)
                    o, qc = paint(tmask, bmask)
                    return o
                before = (check["bleed_px"], check["base_left_px"])
                kept = (out, tmask, bmask, qc, check, problems)
                out, ms = local_fix(img, out, [tmask] + ([bmask] if bottom else []), [thex] + ([bhex] if bottom else []), repaint)
                tmask, bmask = ms[0], (ms[1] if bottom else bmask)
                check, problems = judge(out, tmask, bmask, qc)
                if check["bleed_px"]:  # the retouch recolours afresh: skin and backdrop get their own pixels back again
                    out = fix_photo(img, out, {"_bleed": check["_bleed"]}, None)
                    check, problems = judge(out, tmask, bmask, qc)
                if check["bleed_px"] + check["base_left_px"] > sum(before):  # the retouch made it worse: keep the fix
                    out, tmask, bmask, qc, check, problems = kept
                fixed.append(f"retouch: bleed {before[0]}→{check['bleed_px']} px, base colour {before[1]}→{check['base_left_px']} px")
                self._event("qc_retouched", model=path.parent.parent.name, path=str(path))
            if problems:  # still counted after fixing: the agent looks whether a shopper would see it
                agent = inspect(self.inspector, out, photo_points(top, bottom, check["base_left_px"] > 0, check["bleed_px"] > 0,
                                                                [thex] + ([bhex] if bottom else [])))
                held = agent["ok"] is not True
            if fixed:
                self._event("qc_fixed", model=path.parent.parent.name, path=str(path))
            if held:
                self._event("qc_held", model=path.parent.parent.name, path=str(path), problems=agent["problems"])
        qc["check"] = {"passed": not problems or (agent or {}).get("ok") is True, "problems": problems,
                       "bleed_px": check["bleed_px"], "base_left_px": check["base_left_px"]}
        if spec.qc_agent:
            qc["check"].update(held=held, fixed=fixed, agent=agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec.keep_png:
            out.save(path)
        if spec.canvas:
            fit_canvas(out, *spec.canvas).save(path.with_name(f"{pose}_{spec.canvas[0]}x{spec.canvas[1]}.jpg"), quality=95)
        qc_path = path.parent / "qc.json"
        rows = json.loads(qc_path.read_text()) if qc_path.exists() else {}
        rows[pose] = qc
        write_qc(qc_path, rows)
        marks = path.with_name(f"{pose}_qc.jpg")  # where the problems are, for the dashboard
        if problems:
            self._qc_marks(out, check, marks)
            self._event("qc_flagged", model=path.parent.parent.name, path=str(path), problems=problems)
        else:
            marks.unlink(missing_ok=True)
        self._export(path, pose)

    @staticmethod
    def _qc_marks(img, check, dest: Path):
        """A small copy of the photo with bled pixels in cyan and leftover base colour in magenta."""
        import numpy as np
        from PIL import Image
        from scipy import ndimage

        a = np.asarray(img.convert("RGB")).copy()
        a[ndimage.binary_dilation(check["_bleed"], iterations=2)] = (0, 230, 255)
        a[ndimage.binary_dilation(check["_left"], iterations=2)] = (255, 0, 200)
        im = Image.fromarray(a)
        im.thumbnail((800, 800))
        im.save(dest, quality=85)

    def _export(self, path: Path, pose: str):
        if not self.export:
            return
        import shutil

        model = path.parent.parent.name
        dest = self.export / self.spec.name / path.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        files = []
        if self.export_what in ("jpg", "both") and self.spec.canvas:
            files.append(path.with_name(f"{pose}_{self.spec.canvas[0]}x{self.spec.canvas[1]}.jpg"))
        if (self.export_what in ("png", "both") or not self.spec.canvas) and path.exists():
            files.append(path)
        for f in files:
            if f.exists():
                shutil.copy2(f, dest / f"{model}_{pose}{f.suffix}")

    # ---- results
    def _done(self, path: Path, pose: str) -> bool:
        """Nothing to make for this photo. redo True: everything is made again; redo "held": only the photos the
        QC agent held (not approved by a person) are made and judged again."""
        if self.redo is True:
            return False
        if not self.made(path, pose):
            return False
        if self.redo == "held":
            from .qcagent import passed

            qc = path.parent / "qc.json"
            try:
                row = (json.loads(qc.read_text()).get(pose) or {}).get("check") or {}
            except (OSError, ValueError):
                return True
            return not (row.get("held") and not passed(qc, pose))
        return True

    def made(self, path: Path, pose: str) -> bool:
        """A recoloured photo exists: its PNG, or only its marketplace JPG when PNGs aren't kept, or that JPG
        went to Google Drive and was taken off the server."""
        c = self.spec.canvas
        if path.exists():
            return True
        if not c:
            return False
        jpg = path.with_name(f"{pose}_{c[0]}x{c[1]}.jpg")
        if jpg.exists():
            return True
        if not hasattr(self, "_freed"):
            from .catalogue import freed
            self._freed = freed(self.root)
        return jpg.relative_to(self.root).as_posix() in self._freed

    def results(self) -> List[dict]:
        """Every finished photo: model, top, bottom, pose, paths (relative to the batch folder; png is
        None when only the marketplace JPG was kept)."""
        rows = []
        size = re.compile(r"^(.*)_\d+x\d+$")
        for md in sorted(self.root.glob("m[0-9][0-9][0-9][0-9]")):
            for pd in sorted(p for p in md.iterdir() if p.is_dir() and not p.name.startswith("_")):
                top, _, bottom = pd.name.partition("__")
                pq = pd / "prints_qc.json"
                pchecks = json.loads(pq.read_text()) if pq.exists() else {}
                poses = {f.stem for f in pd.glob("*.png") if not f.name.startswith(".")}
                poses |= {size.match(f.stem).group(1) for f in pd.glob("*_*x*.jpg") if size.match(f.stem) and not f.name.startswith(".")}
                for pose in sorted(poses):
                    png = pd / f"{pose}.png"
                    # this pose's own JPG, not a printed one (front_main_print_237_...) that starts the same
                    canvas = sorted(c for c in pd.glob(f"{pose}_*x*.jpg") if size.match(c.stem) and size.match(c.stem).group(1) == pose)
                    row = {"model": md.name, "top": top, "bottom": bottom or None, "pose": pose,
                           "png": str(png.relative_to(self.root)) if png.exists() else None,
                           "jpg": str(canvas[0].relative_to(self.root)) if canvas else None}
                    if pose in pchecks:  # the AI print checker's verdict on a printed photo
                        c = pchecks[pose]
                        row["print_check"] = {"ok": c.get("ok"), "fixed": c.get("fixed"), "error": c.get("error"),
                                              "issues": (c.get("rounds") or [{}])[-1].get("issues", [])}
                    rows.append(row)
        return rows
