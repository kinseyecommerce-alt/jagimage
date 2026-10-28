"""Print designs: original T-shirt print artwork in the styles that sell online in India (motivational
typography, retro sunsets, line art, desi slogans, dinosaurs and space for kids ...), made by the AI as
print-ready PNGs and kept in design sets that the Print placement tab can use like a print folder.

The designs are new artwork in popular styles. They never copy an existing design, brand, logo, film or
cartoon character, team or celebrity: those belong to someone else and get listings taken down."""

import io
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

# generic themes that sell well on Indian marketplaces (Myntra, Amazon, Flipkart, Meesho), per audience
THEMES: Dict[str, List[str]] = {
    "adults": ["Motivational typography", "Minimal line art", "Retro sunset and waves", "Travel and mountains",
               "Bike and road trip", "Gym and fitness", "Coffee and chai lover", "Abstract geometric",
               "Botanical and floral", "Vintage college-style lettering", "Desi slang and humour", "Tamil slogans",
               "Hindi slogans", "Wildlife and nature", "Space and astronaut", "Music and headphones",
               "Cricket fan (no teams or logos)", "Japanese-style wave and sun"],
    "kids": ["Dinosaurs", "Space and rockets", "Cars, trucks and trains", "Cute animals", "Unicorns and rainbows",
             "Ocean and sea creatures", "Superhero-style (original, no known heroes)", "Robots", "Little explorer",
             "Butterflies and flowers", "Fun Tamil words", "Fun Hindi words", "Birthday and celebration",
             "Football and sports"],
}
TEES = {"dark": "dark T-shirts (black, navy, bottle green): light and bright ink colours, no black in the design",
        "light": "light T-shirts (white, light grey, pastels): dark and bright ink colours, no white in the design",
        "both": "both dark and light T-shirts: mid-tone ink colours that show on black and on white"}
LANGUAGES = {"English": "short English text", "Tamil": "short text in Tamil script", "Hindi": "short text in Hindi (Devanagari)",
             "Mixed": "a mix of English, Tamil and Hindi text across the designs", "None": "no text at all: pure illustration"}
MAX_DESIGNS = 40

BRIEF_SYSTEM = """You are the print designer of SPB Textile, a T-shirt maker in Tirupur. Plan ORIGINAL T-shirt print
designs in the styles that sell well online in India (Myntra, Amazon, Flipkart, Meesho) for the audience, themes,
T-shirt colours and language given. Rules:
- Every design is new artwork. Never copy or imitate an existing design, brand, logo, trademark, team, film,
  cartoon or game character, celebrity, or song lyric; no religious or political symbols; nothing offensive.
- Print-ready: one centred graphic for the chest, flat solid colours (at most 5), clean bold shapes that screen
  print well, no photo, no tiny details, no gradients that need many colours.
- Text, when asked for, is short (1-6 words), spelt correctly, catchy and positive; give it exactly in "text".
- Make the designs clearly different from each other (subject, layout, colours, lettering style).
For each design give: name (2-4 words), theme (one of the given themes), text ("" for none), and prompt: a
detailed instruction for an image AI describing the artwork only (subject, composition, style, ink colours,
lettering style and the exact text), without mentioning a T-shirt mockup or a model."""

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {"designs": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "theme": {"type": "string"}, "text": {"type": "string"}, "prompt": {"type": "string"}},
        "required": ["name", "theme", "text", "prompt"]}}},
    "required": ["designs"],
}


def audience(category: str) -> str:
    return "kids" if category.startswith("Kids") else "adults"


def art_prompt(brief: dict, tees: str) -> str:
    """The image AI's instruction for one design: the artwork alone, isolated, ready to print."""
    bg = "pure black" if tees == "dark" else "pure white"
    text = f' The only text, spelt exactly: "{brief["text"]}".' if brief.get("text") else " No text, no letters."
    return (f"Original T-shirt print artwork for screen printing: {brief['prompt'].rstrip('.')}.{text} "
            f"Flat vector-style illustration with solid ink colours (at most 5), crisp clean edges, bold readable shapes, "
            f"one centred graphic. Inks suited to {TEES[tees]}. Isolated on a plain {bg} background "
            "(transparent if possible) with empty space around it. No T-shirt, no mockup, no model, no frame, no border, "
            "no shadow, no watermark, no signature, no brand names, logos, trademarks, celebrities or cartoon characters.")


def clean_briefs(ans: dict, themes: List[str], count: int) -> List[dict]:
    out, seen = [], set()
    for d in (ans.get("designs") or [])[:count]:
        if not isinstance(d, dict):
            continue
        name = " ".join(str(d.get("name") or "").split())[:40]
        prompt = " ".join(str(d.get("prompt") or "").split())[:1200]
        if not name or not prompt or name.lower() in seen:
            continue
        seen.add(name.lower())
        theme = str(d.get("theme") or "")
        out.append({"name": name, "theme": theme if theme in themes else (themes[0] if themes else theme[:60]),
                    "text": " ".join(str(d.get("text") or "").split())[:60], "prompt": prompt})
    return out


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:50] or "design"


class DesignStore:
    """Design sets on disk: <root>/<set id>/meta.json and one PNG per design."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.Lock()

    def folder(self, set_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]{4,40}", set_id or ""):
            raise FileNotFoundError(set_id)
        return self.root / set_id

    def sets(self) -> List[dict]:
        rows = []
        for meta in self.root.glob("*/meta.json") if self.root.exists() else []:
            try:
                m = json.loads(meta.read_text())
            except ValueError:
                continue
            m["items"] = [i for i in m.get("items", []) if (meta.parent / i["file"]).exists()]
            rows.append(m)
        return sorted(rows, key=lambda m: -m.get("created", 0))

    def create(self, meta: dict) -> dict:
        set_id = f"{slug(meta.get('category', 'set')).lower().replace('_', '-')}-{uuid.uuid4().hex[:6]}"
        meta = {**meta, "id": set_id, "created": time.time(), "items": []}
        d = self.folder(set_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta

    def add(self, set_id: str, brief: dict, png: bytes) -> dict:
        d = self.folder(set_id)
        with self.lock:
            meta = json.loads((d / "meta.json").read_text())
            name = slug(brief["name"])
            file = f"{name}.png"
            n = 2
            while (d / file).exists():
                file, n = f"{name}_{n}.png", n + 1
            (d / file).write_bytes(png)
            item = {**brief, "file": file}
            meta["items"].append(item)
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            return item

    def remove(self, set_id: str, file: Optional[str] = None):
        d = self.folder(set_id)
        with self.lock:
            if file is None:
                for f in d.glob("*"):
                    f.unlink()
                d.rmdir()
                return
            if "/" in file or not file.endswith(".png"):
                raise FileNotFoundError(file)
            (d / file).unlink(missing_ok=True)
            meta = json.loads((d / "meta.json").read_text())
            meta["items"] = [i for i in meta["items"] if i["file"] != file]
            (d / "meta.json").write_text(json.dumps(meta, indent=1))

    def files(self, set_id: str) -> list:
        """[(design name, PNG path)] of one set, for printing."""
        d = self.folder(set_id)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(set_id)
        meta = json.loads((d / "meta.json").read_text())
        return [(Path(i["file"]).stem, d / i["file"]) for i in meta["items"] if (d / i["file"]).exists()]


def finish_art(raw: bytes) -> bytes:
    """The AI's picture as a print PNG: its plain background taken out (when the AI couldn't make it
    transparent) and trimmed to the design with a small margin."""
    from PIL import Image

    from .prints import artwork

    art = artwork(Image.open(io.BytesIO(raw)))
    box = art.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not box:
        raise ValueError("the AI returned an empty picture")
    pad = int(0.03 * max(art.size))
    art = art.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(art.width, box[2] + pad), min(art.height, box[3] + pad)))
    buf = io.BytesIO()
    art.save(buf, "PNG")
    return buf.getvalue()


class DesignJob:
    """Makes the planned designs one after another into a new design set."""

    def __init__(self, store: DesignStore, meta: dict, briefs: List[dict], make: Callable[[str], bytes]):
        self.id = uuid.uuid4().hex[:8]
        self.store, self.briefs, self.make = store, briefs, make
        self.set = store.create(meta)
        self.done, self.failed, self.errors = 0, 0, []
        self.stop_flag = threading.Event()
        self.finished = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        tees = self.set.get("tees", "both")
        for b in self.briefs:
            if self.stop_flag.is_set():
                break
            try:
                self.store.add(self.set["id"], b, finish_art(self.make(art_prompt(b, tees))))
                self.done += 1
            except Exception as exc:
                self.failed += 1
                self.errors.append(f"{b['name']}: {type(exc).__name__}: {exc}"[:300])
        self.finished = time.time()

    def status(self) -> dict:
        return {"id": self.id, "set": self.set["id"], "total": len(self.briefs), "done": self.done, "failed": self.failed,
                "errors": self.errors[-5:], "running": self.finished is None}
