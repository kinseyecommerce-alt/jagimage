"""Print run: plain T-shirt photos from one Google Drive folder (any photos: the Studio's model photos, flat lays,
hanger shots) and prints from another become printed product photos, one folder per SKU, saved to Google Drive
and Dropbox, with a manifest the Listing tab writes marketplace listings from.

- The tee photos are grouped by folder (one folder = one product: a style in a colour). Within a folder, photos
  of the same model or piece share a name stem (m0001_front.jpg / m0001_back.jpg).
- The vision AI looks at each front photo once: is it a T-shirt from the front, where is the shirt and its chest,
  what colour is it. The software finds the shirt's exact outline (jagimage.segment) and prints the artwork on
  it as drawn (jagimage.prints), checks the placement by its firm rules and corrects it.
- Which prints go on which colours: a print whose ink would hardly show on the shirt (a navy print on a navy
  tee) is left out by measurement; the AI then judges the rest the way a buyer would (once per print and colour).
- Each print on each product is one SKU, named by the shop's SKU pattern."""

import io
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from . import library as L
from .prints import PLACES, PrintError, adjust, artwork, measure, open_image, place_print, rule_check
from .recolor import hex_to_lab, rgb_to_lab
from .segment import SegmentError, shirt_mask

SHIRT_SYSTEM = """You look at one product photo for a clothing shop's software. Say whether it shows a T-shirt-type top
(T-shirt, sleeveless tee, oversized tee, sweatshirt or hoodie) and from which side, and where it is.
view: "front" (the front of the top faces the camera: worn by a model or laid flat), "back", or "other" (side,
close-up, folded, no top). box: the top's bounding box [x0, y0, x1, y1] as shares of the photo's width and height
(0..1), sleeves included. chest: [x, y], a point in the middle of the chest of the top (on the fabric, not on a
hand, a print or a logo). colour: the top's colour in one or two plain words (e.g. "navy blue"), and colour_hex
its typical colour as #RRGGBB, read off the lit fabric. plain: true when the front has no print, logo or
pattern. style: one of HalfSleeveTee, FullSleeveTee, SleevelessTee, OversizedTee, Sweatshirt, Hoodie, ZipHoodie.
wearer: Men, Women, Kids_Boys, Kids_Girls or "" when no person (a flat lay)."""

SHIRT_SCHEMA = {
    "type": "object",
    "properties": {"is_top": {"type": "boolean"}, "view": {"type": "string"}, "plain": {"type": "boolean"},
                   "box": {"type": "array", "items": {"type": "number"}}, "chest": {"type": "array", "items": {"type": "number"}},
                   "colour": {"type": "string"}, "colour_hex": {"type": "string"}, "style": {"type": "string"},
                   "wearer": {"type": "string"}},
    "required": ["is_top", "view", "box", "chest", "colour", "colour_hex", "style"],
}

SUIT_SYSTEM = """You are a clothing shop's merchandiser. Image 1 is a print (artwork) a printer will screen-print on
T-shirts. Say whether it will look good, clearly visible and sellable, on a T-shirt of the given colour: think of
contrast (a dark print on a dark shirt disappears), colour harmony and who buys that colour. Also describe the
print for a product listing: theme (e.g. "dinosaur", "typography quote", "anime", "floral") and a short description
of what it shows, and the text in it if any, exactly. ok: true when it suits the colour."""

SUIT_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}, "theme": {"type": "string"},
                   "description": {"type": "string"}, "text": {"type": "string"}},
    "required": ["ok", "reason", "theme", "description"],
}

STYLE_CODES = {"HalfSleeveTee": "HS", "FullSleeveTee": "FS", "SleevelessTee": "SL", "OversizedTee": "OS",
               "Sweatshirt": "SW", "Hoodie": "HD", "ZipHoodie": "ZH"}
CATEGORY_CODES = {"Men": "M", "Women": "W", "Kids_Boys": "KB", "Kids_Girls": "KG"}
FABRIC_CODES = {"CottonMix": "CM", "PureCotton": "PC"}
# folder names in the shop's catalogue layout (and common variants) that tell what a folder holds
STYLE_WORDS = [("zip hood", "ZipHoodie"), ("sweatshirt", "Sweatshirt"), ("hood", "Hoodie"), ("sleeveless", "SleevelessTee"),
               ("full sleeve", "FullSleeveTee"), ("half sleeve", "HalfSleeveTee"), ("oversized", "OversizedTee"),
               ("vest", "SleevelessTee"), ("t-shirt", "HalfSleeveTee"), ("tshirt", "HalfSleeveTee"), ("tee", "HalfSleeveTee")]
CATEGORY_WORDS = [("cottonmens", "Men"), ("girl", "Kids_Girls"), ("boy", "Kids_Boys"), ("women", "Women"), ("woman", "Women"), ("ladies", "Women"),
                  ("men", "Men"), ("man", "Men")]
FABRIC_WORDS = [("pure cotton", "PureCotton"), ("cotton mix", "CottonMix"), ("cottonmens", "PureCotton")]
SKIP_FOLDERS = ("combo sets", "_bases", "printed")  # combo-set photos show two colours; bases are not finished photos
DEFAULT_PATTERN = "{BRAND}-{CAT}-{STYLE}-{COL3}-{PRINT}"
LOW_CONTRAST = 0.12   # OKLab distance under which ink hardly shows on the shirt
HIDDEN_SHARE = 0.6    # a print with more than this share of its ink in low contrast is left out without asking


def guess(path: List[str]) -> dict:
    """What the folder names say: category, style, fabric and colour (colour = a shop colour named in the path)."""
    low = [p.lower() for p in path]
    joined = " / ".join(low)
    out = {}
    for words, key in ((CATEGORY_WORDS, "category"), (STYLE_WORDS, "style"), (FABRIC_WORDS, "fabric")):
        for w, v in words:
            if re.search(r"\b" + re.escape(w), joined):
                out[key] = v
                break
    names = sorted({c for cols in L.COLOURS.values() for c in cols}, key=len, reverse=True)
    for p in reversed(path):
        hit = next((c for c in names if p.strip().lower() == c.lower()), None)
        if hit:
            out["colour"] = hit
            break
    return out


def colour_hex_for(name: str, fabric: str = "") -> Optional[str]:
    for fab in ([fabric] if fabric in L.COLOURS else []) + list(L.COLOURS):
        for c, h in L.COLOURS[fab].items():
            if c.lower() == str(name or "").lower():
                return h
    return None


def nearest_colour(hex_: str) -> str:
    """The shop colour closest to a colour the AI read off a photo."""
    try:
        t = hex_to_lab(hex_)
    except Exception:
        return ""
    best, dist = "", 9.0
    for cols in L.COLOURS.values():
        for c, h in cols.items():
            d = float(np.linalg.norm(hex_to_lab(h) - t))
            if d < dist:
                best, dist = c, d
    return best if dist < 0.12 else ""


def code(text: str, n: int = 0) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "", str(text or "")).upper()
    return s[:n] if n else s


def make_sku(pattern: str, fields: dict) -> str:
    """The shop's SKU pattern with its fields filled in: {BRAND} {CAT} {CATEGORY} {STYLE} {STYLE_NAME} {FABRIC}
    {COLOUR} {COL3} {PRINT} {N}. The result has only letters, digits and dashes."""
    pattern = pattern or DEFAULT_PATTERN

    def put(m):
        k = m.group(1).upper()
        return str(fields.get(k, ""))
    sku = re.sub(r"\{([A-Za-z_0-9]+)\}", put, pattern)
    sku = re.sub(r"[^A-Za-z0-9_-]+", "-", sku).strip("-")
    sku = re.sub(r"-{2,}", "-", sku)
    if not sku:
        raise ValueError("The SKU pattern gives an empty SKU")
    return sku[:60]


def sku_fields(brand: str, product: dict, print_name: str, n: int) -> dict:
    return {"BRAND": code(brand) or "SPB", "CAT": CATEGORY_CODES.get(product.get("category", ""), "U"),
            "CATEGORY": code(product.get("category", "")), "STYLE": STYLE_CODES.get(product.get("style", ""), "TS"),
            "STYLE_NAME": code(product.get("style", "")), "FABRIC": FABRIC_CODES.get(product.get("fabric", ""), ""),
            "COLOUR": code(product.get("colour", "")), "COL3": code(product.get("colour", ""), 3),
            "PRINT": code(print_name, 16), "N": f"{n:04d}"}


def ink_hidden_share(art: Image.Image, shirt_hex: str) -> float:
    """The share of the print's ink that would hardly show on a shirt of this colour."""
    a = artwork(art)
    a.thumbnail((300, 300))
    rgba = np.asarray(a)
    ink = rgba[..., 3] > 128
    if not ink.any():
        return 1.0
    lab = rgb_to_lab(rgba[..., :3])[ink]
    return float((np.linalg.norm(lab - hex_to_lab(shirt_hex), axis=-1) < LOW_CONTRAST).mean())


def _jpeg(img: Image.Image, side: int) -> tuple:
    im = img.convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


def _art_jpeg(art: Image.Image, side: int = 512) -> tuple:
    a = artwork(art)
    bg = Image.new("RGB", a.size, (200, 200, 200))
    bg.paste(a, mask=a.getchannel("A"))
    return _jpeg(bg, side)


def stem_of(name: str) -> str:
    """m0001_front.jpg and m0001_back.jpg share the stem m0001: photos of the same model or piece."""
    base = re.sub(r"\.[A-Za-z0-9]+$", "", name)
    stem = base
    while True:  # m0001_front_main_2000x2000 -> m0001
        cut = re.sub(r"[ _-]*(front|back|main|rear|side|\d+x\d+)$", "", stem, flags=re.I)
        if cut == stem or not cut:
            return stem
        stem = cut


def is_back_name(name: str) -> bool:
    return bool(re.search(r"(^|[ _-])(back|rear)([ _.-]|$)", name.lower()))


class PrintRun:
    """One print run, in a background thread. status() is what the Print placement tab shows."""

    def __init__(self, drive, ask: Callable, data: Path, tees_folder: str, prints_folder: str, dropbox=None,
                 place: str = "chest_large", per_sku: int = 5, brand: str = "SPB", pattern: str = DEFAULT_PATTERN,
                 limit: int = 0, label: str = "", checker: Optional[Callable] = None):
        if place not in PLACES:
            raise ValueError(f"place must be one of {', '.join(PLACES)}")
        self.id = uuid.uuid4().hex[:8]
        self.drive, self.ask, self.dropbox = drive, ask, dropbox
        self.data = Path(data)
        self.tees_folder, self.prints_folder = tees_folder, prints_folder
        self.place, self.per_sku = place, max(1, min(int(per_sku or 5), 12))
        self.brand, self.pattern, self.limit = brand or "SPB", pattern or DEFAULT_PATTERN, max(0, int(limit or 0))
        self.label, self.checker = label, checker  # checker: the AI print checker (Agent.check_print), or rules only
        self.counts = {"tee_photos": 0, "products": 0, "prints": 0, "skus": 0, "photos": 0, "left_out": 0,
                       "fixed": 0, "errors": 0, "dropbox": 0}
        self.log: List[dict] = []
        self.skus: List[dict] = []
        self.error: Optional[str] = None
        self.stage = "starting"
        self.started, self.finished = time.time(), None
        self.stop = threading.Event()
        self.looks: Dict[str, dict] = {}
        self.suits: Dict[tuple, dict] = {}
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"printrun-{self.id}")

    # ---- what the tab shows
    def status(self) -> dict:
        return {"id": self.id, "label": self.label, "stage": self.stage, "counts": dict(self.counts), "error": self.error,
                "running": self.finished is None, "elapsed": round((self.finished or time.time()) - self.started),
                "skus": [{k: s[k] for k in ("sku", "category", "style", "colour", "print", "photos")} for s in self.skus[-200:]],
                "log": self.log[-40:], "manifest": f"/api/printruns/{self.id}/manifest"}

    def _note(self, kind: str, **kw):
        self.log.append({"kind": kind, "t": round(time.time() - self.started, 1), **kw})
        del self.log[:-400]

    # ---- the steps
    def _look(self, key: str, img: Image.Image) -> dict:
        if key not in self.looks:
            ans = self.ask("Look at this product photo.", SHIRT_SYSTEM, SHIRT_SCHEMA, [_jpeg(img, 1024)])
            self.looks[key] = ans if isinstance(ans, dict) else {}
        return self.looks[key]

    def _suits(self, pname: str, art: Image.Image, colour: str, hex_: str) -> dict:
        k = (pname, colour.lower())
        if k not in self.suits:
            hidden = ink_hidden_share(art, hex_)
            if hidden > HIDDEN_SHARE:
                self.suits[k] = {"ok": False, "reason": f"{round(hidden * 100)}% of the print would hardly show on {colour}",
                                 "theme": "", "description": ""}
            else:
                ans = self.ask(f"T-shirt colour: {colour} ({hex_}).", SUIT_SYSTEM, SUIT_SCHEMA, [_art_jpeg(art)])
                self.suits[k] = {"ok": bool(ans.get("ok")), "reason": " ".join(str(ans.get("reason") or "").split())[:200],
                                 "theme": " ".join(str(ans.get("theme") or "").split())[:60],
                                 "description": " ".join(str(ans.get("description") or "").split())[:300],
                                 "text": " ".join(str(ans.get("text") or "").split())[:120]}
        return self.suits[k]

    def _placement(self, photo: Image.Image, mask: np.ndarray, art: Image.Image) -> dict:
        """The print placement agent, once per SKU on its first photo: the firm rules, then the AI looks at the
        printed photo next to the artwork (too high? too big? off-centre?) and moves or resizes it, at most twice.
        The other photos of the SKU get the same placement."""
        from .studio import judge_print

        adj = adjust()
        for _ in range(3):
            try:
                out = place_print(photo, mask, art, self.place, **adj)
                verdict = judge_print(out, art, mask, self.place, adj, self.checker)
            except PrintError:
                return adj
            except Exception as exc:  # the AI checker couldn't run: the rules' placement stands
                self._note("error", error=f"print agent: {type(exc).__name__}: {exc}"[:200])
                return adj
            if verdict["ok"] or (verdict["dx"], verdict["dy"], verdict["scale"]) == (0.0, 0.0, 1.0):
                return adj
            adj = adjust(adj["dx"] + verdict["dx"], adj["dy"] + verdict["dy"], adj["scale"] * verdict["scale"])
            self.counts["fixed"] += 1
            self._note("moved", why="; ".join(verdict.get("issues") or [])[:200], **adj)
        return adj

    def _place(self, photo: Image.Image, mask: np.ndarray, art: Image.Image, adj: Optional[dict] = None) -> Image.Image:
        return place_print(photo, mask, art, self.place, **(adj or adjust()))

    def _products(self, tees: list) -> List[dict]:
        """The tee photos grouped into products (one folder each), front photos first, at most per_sku models."""
        groups: Dict[tuple, dict] = {}
        for f in tees:
            if any(w in p.lower() for p in f["path"] for w in SKIP_FOLDERS):
                continue
            g = groups.setdefault(tuple(f["path"]), {"path": f["path"], "files": [], **guess(f["path"])})
            g["files"].append(f)
        out = []
        for g in groups.values():
            stems: Dict[str, dict] = {}
            for f in sorted(g["files"], key=lambda f: f["name"]):
                s = stems.setdefault(stem_of(f["name"]), {"front": None, "back": None, "other": []})
                if is_back_name(f["name"]):
                    s["back"] = s["back"] or f
                elif s["front"] is None:
                    s["front"] = f
                else:
                    s["other"].append(f)
            g["models"] = [v for v in stems.values() if v["front"]]
            out.append(g)
        return out

    def _run(self):
        try:
            self._go()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"[:400]
            self._note("error", error=self.error)
        finally:
            self.finished = time.time()
            self.stage = "stopped" if self.stop.is_set() else ("failed" if self.error else "done")
            self._save()

    def _go(self):
        self.stage = "reading Drive"
        tees = self.drive.list_tree(self.tees_folder)
        prints = [f for f in self.drive.list_tree(self.prints_folder, depth=2)]
        self.counts["tee_photos"], self.counts["prints"] = len(tees), len(prints)
        if not tees:
            raise RuntimeError("No pictures in the T-shirt folder (or the app can't see it: share it with the connected Google account)")
        if not prints:
            raise RuntimeError("No pictures in the prints folder")
        cache = self.data / "print_cache"
        cache.mkdir(parents=True, exist_ok=True)
        arts = []
        for p in prints:
            local = cache / f"{p['id']}_{re.sub(r'[^0-9A-Za-z]', '', p.get('modifiedTime', ''))[:14]}{Path(p['name']).suffix.lower() or '.png'}"
            if not local.exists():
                local.write_bytes(self.drive.download(p["id"]))
            try:
                arts.append((Path(p["name"]).stem, open_image(str(local))))
            except PrintError as exc:
                self._note("error", file=p["name"], error=str(exc))
                self.counts["errors"] += 1
        products = self._products(tees)
        self.counts["products"] = len(products)
        out_root = self.data / "printed" / self.id
        n = 0
        for prod in products:
            if self.stop.is_set() or (self.limit and len(self.skus) >= self.limit):
                break
            self.stage = f"{' / '.join(prod['path']) or 'tees'}"
            fronts = []  # (stem entry, photo, mask, look)
            for m in prod["models"]:
                if len(fronts) >= self.per_sku or self.stop.is_set():
                    break
                f = m["front"]
                try:
                    photo = open_image(self.drive.download(f["id"])).convert("RGB")
                    look = self._look(f["id"], photo)
                    if not look.get("is_top") or str(look.get("view", "")).lower() != "front":
                        self._note("skipped", file=f["name"], why="not a front view of a top")
                        continue
                    if look.get("plain") is False:
                        self._note("skipped", file=f["name"], why="already printed")
                        continue
                    mask = shirt_mask(photo, look.get("box") or [], look.get("chest") or [])
                    fronts.append((m, photo, mask, look))
                except (PrintError, SegmentError) as exc:
                    self._note("error", file=f["name"], error=str(exc))
                    self.counts["errors"] += 1
            if not fronts:
                continue
            look = fronts[0][3]
            prod.setdefault("style", look.get("style") if look.get("style") in STYLE_CODES else "HalfSleeveTee")
            prod.setdefault("category", look.get("wearer") if look.get("wearer") in CATEGORY_CODES else "")
            if "colour" not in prod:
                prod["colour"] = nearest_colour(look.get("colour_hex", "")) or " ".join(str(look.get("colour") or "").title().split())[:30]
            hex_ = colour_hex_for(prod["colour"], prod.get("fabric", "")) or str(look.get("colour_hex") or "#808080")
            for pname, art in arts:
                if self.stop.is_set() or (self.limit and len(self.skus) >= self.limit):
                    break
                verdict = self._suits(pname, art, prod["colour"], hex_)
                if not verdict["ok"]:
                    self.counts["left_out"] += 1
                    self._note("left_out", colour=prod["colour"], print=pname, why=verdict["reason"])
                    continue
                n += 1
                sku = make_sku(self.pattern, sku_fields(self.brand, prod, pname, n))
                files, adj = [], None
                for i, (m, photo, mask, _) in enumerate(fronts):
                    try:
                        if adj is None:
                            adj = self._placement(photo, mask, art)
                        files.append(self._place(photo, mask, art, adj))
                    except PrintError as exc:
                        self._note("error", sku=sku, error=str(exc))
                        self.counts["errors"] += 1
                        continue
                    if m["back"] and len(files) < self.per_sku * 2:
                        try:
                            files.append(open_image(self.drive.download(m["back"]["id"])).convert("RGB"))
                        except PrintError:
                            pass
                if not files:
                    continue
                self._save_sku(out_root, sku, prod, pname, verdict, hex_, files)
        self.stage = "done"

    def _save_sku(self, out_root: Path, sku: str, prod: dict, pname: str, verdict: dict, hex_: str, photos: list):
        folder = out_root / sku
        folder.mkdir(parents=True, exist_ok=True)
        images = []
        for i, im in enumerate(photos, 1):
            name = f"{sku}_{i}.jpg"
            local = folder / name
            im.save(local, "JPEG", quality=95)
            row = {"name": name}
            try:
                self.drive.upload(local, ["Printed", sku], name)
                row["drive"] = True
            except Exception as exc:
                self._note("error", sku=sku, error=f"Drive: {exc}"[:200])
                self.counts["errors"] += 1
            if self.dropbox is not None and self.dropbox.connected:
                try:
                    path = self.dropbox.upload(local.read_bytes(), sku, name)
                    row["dropbox"] = path
                    row["url"] = self.dropbox.link(path)
                    self.counts["dropbox"] += 1
                except Exception as exc:
                    self._note("error", sku=sku, error=f"Dropbox: {exc}"[:600])
                    self.counts["errors"] += 1
            images.append(row)
            self.counts["photos"] += 1
            if i == 1:  # the server keeps only a small first photo, for the Listing tab's AI to look at
                small = im.convert("RGB")
                small.thumbnail((1024, 1024))
                small.save(folder / "listing.jpg", "JPEG", quality=88)
            local.unlink(missing_ok=True)
        self.skus.append({"sku": sku, "category": prod.get("category", ""), "style": prod.get("style", ""),
                          "fabric": prod.get("fabric", ""), "colour": prod.get("colour", ""), "colour_hex": hex_,
                          "print": pname, "theme": verdict.get("theme", ""), "print_description": verdict.get("description", ""),
                          "print_text": verdict.get("text", ""), "photos": len(images), "images": images,
                          "folder": " / ".join(prod["path"])})
        self.counts["skus"] += 1
        self._note("sku", sku=sku, photos=len(images))
        self._save()

    def _save(self):
        f = self.data / "printruns" / f"{self.id}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"id": self.id, "label": self.label, "started": self.started, "finished": self.finished,
                                 "stage": self.stage, "counts": self.counts, "error": self.error, "place": self.place,
                                 "skus": self.skus}, indent=1))

    def start(self):
        self.thread.start()
        return self


def saved_runs(data: Path) -> List[dict]:
    """The manifests of earlier print runs, newest first (the Listing tab picks SKUs from them)."""
    d = Path(data) / "printruns"
    rows = []
    for f in d.glob("*.json") if d.exists() else []:
        try:
            rows.append(json.loads(f.read_text()))
        except ValueError:
            continue
    return sorted(rows, key=lambda r: -float(r.get("started") or 0))
