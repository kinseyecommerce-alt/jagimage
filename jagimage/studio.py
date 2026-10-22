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
HANDS_FREE = ("Hands free: both hands relaxed and open, never touching the clothes or the body, not in pockets, "
              "not holding anything, arms kept a little away from the body so the whole garment is visible, "
              "including its side seams and hem")
HANDS_FREE_AVOID = "hands on the clothes, hands in pockets, hands tucked in, arms covering the garment, holding an object"
NO_HANDS_FREE = {"handpocket"}  # poses that need the hands on the clothes
NOT_MANNEQUIN = "mannequin, dummy, doll, ghost mannequin, faceless figure, plastic or wax figure, CGI character"
SHOTS = {"outfit": "Top and bottom (full body, both recoloured)", "top": "Top only (head to just above the knee)"}
TOP_ONLY_FRAMING = ("Three-quarter shot framed from the top of the head to just above the knees: knees, shins and feet are "
                    "out of frame, small margin above the head, the whole top fully visible, model fills about 90% of frame height")
FULL_FRAMING = "Full body head to shoes in frame with small margins, model about 85-90% of frame height"
BOTTOM_LIKE = 0.06  # OKLab distance: pixels this close to a top-only shot's photographed bottom aren't the top
BASE_MB, PNG_MB, JPG_MB = 4.1, 4.3, 0.25  # measured sizes: AI base PNG, full-size photo PNG, 1500x1500 JPG
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
    keep_png: bool = True    # also keep a full-size PNG of every photo (4 MB each), besides the 1500x1500 JPG
    same_models: bool = False  # reuse models (faces) already made in this category, e.g. in another top style
    one_base: bool = False   # one light base shoot per model; dark colours are recoloured from it too (half the AI images)
    provider: str = "gemini"  # gemini, openai, or runpod (your own GPU pod with an open model)
    quality: str = "high"    # OpenAI
    image_size: str = "2K"   # Gemini
    gemini_model: str = "flash"  # Gemini: flash (default, cheaper) or pro
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
        if self.provider not in ("gemini", "openai", "runpod"):
            raise ValueError("provider must be gemini, openai or runpod")
        if self.gemini_model not in ("flash", "pro"):
            raise ValueError("gemini_model must be flash or pro")
        if self.batch and self.provider != "gemini":
            raise ValueError("Batch mode (half price) is available with Gemini")
        if self.shot not in SHOTS:
            raise ValueError("shot must be outfit (top and bottom) or top (top only)")
        if self.shot == "top":
            colour_hex(self.bottom_colour, self.fabric)
        elif self.pairing == "each" and not self.bottom_colours and self.bottom not in FIXED_BOTTOMS:
            raise ValueError("pick at least one bottom colour, or pair the bottom with the top colour")
        self.custom_poses = {slug(k).lower(): " ".join(str(v).split()) for k, v in (self.custom_poses or {}).items()}
        empty = [k for k, v in self.custom_poses.items() if not k or not v]
        if empty:
            raise ValueError("each custom pose needs a name and a description")
        self.extra = " ".join(str(self.extra or "").split())
        if len(self.extra) > 1000 or any(len(v) > 1000 for v in self.custom_poses.values()):
            raise ValueError("keep pose descriptions and extra instructions under 1000 characters")
        bad = [p for p in self.poses if p not in self.pose_texts()]
        if bad or "front_main" not in self.poses:
            raise ValueError("poses must include front_main and come from: " + ", ".join(L.POSES))
        if not 1 <= int(self.models) <= 5000:
            raise ValueError("models must be between 1 and 5000")
        self.first_model = int(self.first_model)
        if self.first_model < 1 or self.first_model + int(self.models) - 1 > faces.capacity(self.category):
            raise ValueError(f"model numbers go from 1 to {faces.capacity(self.category)} for {self.category}")
        for c in self.top_colours + self.bottom_colours:
            colour_hex(c, self.fabric)
        if self.canvas:
            self.canvas = tuple(int(v) for v in self.canvas)
        else:
            self.keep_png = True  # without a marketplace JPG the PNG is the only copy
        if not self.name:
            self.name = slug(f"{self.category}_{self.top}_{self.bottom}_{self.fabric}" + ("_top_only" if self.shot == "top" else ""))

    def pose_texts(self) -> Dict[str, str]:
        return {**L.POSES, **self.custom_poses}

    # ---- what the batch makes
    def pairs(self) -> List[Tuple[str, Optional[str]]]:
        """(top colour, bottom colour) pairs; bottom None when it isn't recoloured (jeans, top-only shots)."""
        if self.bottom in FIXED_BOTTOMS or self.shot == "top":
            return [(t, None) for t in self.top_colours]
        if self.pairing == "match" or not self.bottom_colours:
            return [(t, t) for t in self.top_colours]
        return [(t, b) for t in self.top_colours for b in self.bottom_colours]

    def base_key(self, top: str, bottom: Optional[str]) -> Tuple[str, Optional[str]]:
        if self.one_base:
            return ("light", "light" if bottom else None)
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
        ai, photos = n * len(self.bases()) * poses, n * len(self.pairs()) * poses
        mb = ai * BASE_MB + photos * (JPG_MB + (PNG_MB if self.keep_png else 0))
        return {"models": n, "bases_per_model": len(self.bases()), "colour_pairs": len(self.pairs()),
                "poses": poses, "ai_images": ai, "recoloured_images": photos, "disk_gb": round(mb / 1000, 1)}


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
    top_only = spec.shot == "top"
    if spec.bottom in FIXED_BOTTOMS:
        bottom_text, bcol = FIXED_BOTTOMS[spec.bottom], None
    elif top_only:  # the bottom only completes the look: the AI makes it in the chosen colour
        name = spec.bottom_colour if not HEX.match(spec.bottom_colour) else "solid colour " + spec.bottom_colour.upper()
        bottom_text, bcol = L.GARMENTS[spec.bottom].replace("{COLOUR}", name.lower()), None
    else:
        bname, bhex = BOTTOM_BASES[base[1]]
        bottom_text, bcol = L.GARMENTS[spec.bottom].replace("{COLOUR}", bname), (bname, bhex)
    shoes = "small white kids sneakers" if kids else "clean white low-top sneakers"
    lines = {"none": "", "face": L.REFERENCE_LINE, "front": L.FRONT_ONLY_LINE,
             "face+front": L.REFERENCE_LINE + L.MATCH_FRONT_LINE}
    colours = f"top {tname} ({thex})" + (f", bottom {bcol[0]} ({bcol[1]})" if bcol else "")
    rules = L.RULES.replace("Exact garment colour {COLOUR} ({HEX}),", f"Exact garment colours: {colours}, each")
    if top_only:
        rules = rules.replace(FULL_FRAMING, TOP_ONLY_FRAMING)
    text = (f"{lines[reference]}Real photograph of a real, living human model (not a mannequin, dummy, doll or 3D figure): "
            f"{faces.describe(spec.category, model_no, hair=reference == 'none')}, {spec.pose_texts()[pose]}. "
            f"Wearing {top_text}, fabric: {L.FABRIC_TEXT[spec.fabric][L.FABRIC_KIND[spec.top]]}; "
            f"with {bottom_text}" + (f", fabric: {L.FABRIC_TEXT[spec.fabric]['bottom']}" if bcol else "") +
            ("" if top_only else f"; and {shoes}") +
            (". The top and the bottom are two clearly different colours, exactly as stated. " if bcol else ". ") +
            f"{L.BACKGROUNDS.get(pose, L.GREY)} {rules}")
    hands = spec.hands_free and pose not in NO_HANDS_FREE
    if hands:
        text += f" {HANDS_FREE}."
    if spec.extra:
        text += f" Also: {spec.extra.rstrip('.')}."
    if kids:
        text += L.KIDS_RULE
    product = "Trackpants" if spec.bottom == "Trackpants" else spec.top
    negative = L.build_negative(spec.category, product, pose)
    if spec.bottom == "Trackpants":
        text += L.JOGGER_RULE
    if hands:
        negative += ", " + HANDS_FREE_AVOID
    if top_only:
        negative = negative.replace("cut-off feet, ", "") + ", full body shot, feet or shoes in frame, knees in frame"
    return text + " Avoid: " + negative + ", " + NOT_MANNEQUIN + (", top and bottom in the same colour." if bcol else ".")


def account_problem(exc: Exception) -> Optional[str]:
    """A plain explanation when an AI error means the account can't make images at all, so the
    batch stops at once instead of failing every model. None for errors worth retrying."""
    text = f"{type(exc).__name__} {exc}".lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if "credits are depleted" in text or "insufficient_quota" in text or "billing" in text and "402" in text or code == 402:
        return ("The AI account is out of credits or billing isn't set up. Add credits (Gemini: AI Studio > "
                "Billing; OpenAI: platform.openai.com > Billing), then start the batch again: finished photos are kept.")
    if code in (401, 403) or "api key not valid" in text or "invalid_api_key" in text or "incorrect api key" in text \
            or "permission_denied" in text:
        return "The API key was refused (wrong, expired or without access to this model). Save a working key and start again."
    if "runpod token not accepted" in text:
        return "The RunPod pod refused the token. Save the same token the pod was started with, then start again."
    if "runpod pod unreachable" in text:
        return ("The RunPod pod isn't answering: check it is running in the RunPod console (Pods) and the address "
                "ends in -8000.proxy.runpod.net, then start again: finished photos are kept.")
    return None


def is_rate_limit(exc: Exception) -> bool:
    text = f"{exc}".lower()
    return getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429 or "rate limit" in text \
        or ("resource_exhausted" in text and "credits" not in text)


_LEDGER_LOCK = threading.Lock()


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
                 export: Optional[str] = None, export_what="jpg"):
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
        self.on_event = on_event or (lambda e: None)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.fatal = None  # set when the AI account itself can't be used (no credits, bad key)
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
        if stage != "recolor" and self.spec.batch:
            if not self.batch_runner:
                raise RuntimeError("This engine has no batch mode")
            self._run_batched()
            if stage == "bases" or self.stop.is_set():
                return self.counts
            stage = "recolor"  # every base is made (or has failed): recolour them all
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            list(pool.map(lambda m: self.run_model(m, stage), self.spec.model_numbers()))
        return self.counts

    def _run_batched(self):
        """Batch mode: each round sends every base photo whose reference photos already exist as one
        Gemini batch job (half price). Round 1 makes each model's first front, round 2 the other fronts
        and the first base's poses, round 3 the remaining poses. Photos the AI got wrong go into the
        next round, up to `retries` times."""
        poses = ["front_main"] + [p for p in self.spec.poses if p != "front_main"]
        tries: Dict[str, int] = {}
        seen = set()
        while not self.stop.is_set():
            todo = []
            for m in self.spec.model_numbers():
                bases = self.spec.bases()
                first = self.identity_front(m, bases)
                for base in bases:
                    for pose in poses:
                        path = self.base_path(m, base, pose)
                        key = f"m{m:04d}|{base_dir(base)}|{pose}"
                        if path.exists():
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
                    fatal = account_problem(got)
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
        bases = spec.bases()
        first_front = self.identity_front(m, bases)
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
        return prompt(self.spec, m, base, pose, ref), refs

    def _accept(self, path: Path, data: bytes, base, m, pose):
        """Keep an AI image only if it used the base colours (else raise, so it's asked for again)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.png")
        tmp.write_bytes(data)
        try:
            self._check_base(tmp, base)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        if pose == "front_main":
            self._register_face(m, path)  # no-op once the model has a face in this category
        self._event("ai_done", model=m, path=str(path), base=base_dir(base), pose=pose)

    def _shoot(self, m, base, pose, path: Path, first_front: Path) -> bool:
        """Make one base photo if missing. False if it couldn't be made."""
        if path.exists():
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
                fatal = account_problem(exc)
                if fatal:  # no credits / bad key: every other request would fail the same way
                    self.fatal = fatal
                    self.stop.set()
                    self._event("errors", model=m, path=str(path), error=fatal)
                    return False
                wait = 20 * (attempt + 1) if is_rate_limit(exc) else 2 ** attempt
                time.sleep(min(60, wait) if attempt + 1 < self.retries else 0)
        self._event("errors", model=m, path=str(path), error=f"{type(last).__name__}: {last}")
        return False

    def _check_base(self, path: Path, base):
        import numpy as np
        from PIL import Image

        from .recolor import MaskError, garment_mask, rgb_to_lab, save_mask

        lab = rgb_to_lab(np.asarray(Image.open(path).convert("RGB")))
        top, _ = garment_mask(lab, TOP_BASES[base[0]][1])
        if not base[1] and self.spec.bottom not in FIXED_BOTTOMS:
            top = self._trim_fixed_bottom(lab, top)
        masks = {"top": top}
        if base[1]:
            bottom, _ = garment_mask(lab, BOTTOM_BASES[base[1]][1])
            # where the masks meet, each pixel belongs to the garment it's more part of, so no seam
            # pixel is left to neither (the waistband shadow keeping the base colour)
            bottom = np.where(bottom > top, bottom, np.minimum(bottom, 1 - top))
            top = np.where(top >= bottom, top, np.minimum(top, 1 - bottom))
            masks["top"] = top
            if (bottom > 0.95).sum() < 0.002 * bottom.size:
                raise MaskError("the bottom isn't in its base colour")
            masks["bottom"] = bottom
        stem = path.name.replace(".tmp.png", "")
        for k, mk in masks.items():  # saved next to the base; edit them to fix an edge
            save_mask(path.parent / f"{stem}_{k}_mask.png", mk)

    def _trim_fixed_bottom(self, lab, top):
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
        seed = like & (top < 0.5)
        if not seed.any():
            return top
        bottomish = ndimage.binary_propagation(seed, mask=like)
        trimmed = (top > 0.5) & ~bottomish
        return np.minimum(top, ndimage.gaussian_filter(trimmed.astype(np.float32), 0.8))

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
            if self.made(path, pose):
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
        out, de_t, _ = recolor_image(img, TOP_BASES[base[0]][1], thex, mask=tmask, others=bmask if bottom else None)
        qc = {"top": {"colour": top, "hex": thex, "deltaE": round(de_t, 2), "on_shade": de_t < 3}}
        if bottom:
            bhex = colour_hex(bottom, spec.fabric)
            out, de_b, _ = recolor_image(out, BOTTOM_BASES[base[1]][1], bhex, mask=bmask, others=tmask)
            qc["bottom"] = {"colour": bottom, "hex": bhex, "deltaE": round(de_b, 2), "on_shade": de_b < 3}
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec.keep_png:
            out.save(path)
        if spec.canvas:
            fit_canvas(out, *spec.canvas).save(path.with_name(f"{pose}_{spec.canvas[0]}x{spec.canvas[1]}.jpg"), quality=95)
        qc_path = path.parent / "qc.json"
        rows = json.loads(qc_path.read_text()) if qc_path.exists() else {}
        rows[pose] = qc
        write_qc(qc_path, rows)
        self._export(path, pose)

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
    def made(self, path: Path, pose: str) -> bool:
        """A recoloured photo exists: its PNG, or only its marketplace JPG when PNGs aren't kept."""
        c = self.spec.canvas
        return path.exists() or bool(c and path.with_name(f"{pose}_{c[0]}x{c[1]}.jpg").exists())

    def results(self) -> List[dict]:
        """Every finished photo: model, top, bottom, pose, paths (relative to the batch folder; png is
        None when only the 1500x1500 JPG was kept)."""
        rows = []
        size = re.compile(r"^(.*)_\d+x\d+$")
        for md in sorted(self.root.glob("m[0-9][0-9][0-9][0-9]")):
            for pd in sorted(p for p in md.iterdir() if p.is_dir() and not p.name.startswith("_")):
                top, _, bottom = pd.name.partition("__")
                poses = {f.stem for f in pd.glob("*.png") if not f.name.startswith(".")}
                poses |= {size.match(f.stem).group(1) for f in pd.glob("*_*x*.jpg") if size.match(f.stem) and not f.name.startswith(".")}
                for pose in sorted(poses):
                    png = pd / f"{pose}.png"
                    canvas = sorted(pd.glob(f"{pose}_*x*.jpg"))
                    rows.append({"model": md.name, "top": top, "bottom": bottom or None, "pose": pose,
                                 "png": str(png.relative_to(self.root)) if png.exists() else None,
                                 "jpg": str(canvas[0].relative_to(self.root)) if canvas else None})
        return rows
