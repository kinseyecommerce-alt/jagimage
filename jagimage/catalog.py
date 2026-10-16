"""Kids' tee catalogue: front shot + consistent pose shots for every style x colour.

For each SKU (gender, style, colour) the front photo is generated first; every
pose is then made by editing from that front photo (plus the optional face
reference), so the same child, garment and studio carry across all shots.
Output: <out>/<Gender>/<Style>/<Colour>/01-front.png, 02-threequarter.png, ...
"""

import base64
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator, List, Optional

CATEGORIES = [
    ("Boys", "HalfSleeveTee", "plain half-sleeve crew-neck t-shirt"),
    ("Boys", "FullSleeveTee", "plain full-sleeve crew-neck t-shirt with ribbed cuffs"),
    ("Boys", "SleevelessTee", "plain sleeveless crew-neck vest with neat bound armholes"),
    ("Girls", "HalfSleeveTee", "plain half-sleeve crew-neck t-shirt"),
    ("Girls", "FullSleeveTee", "plain full-sleeve crew-neck t-shirt with ribbed cuffs"),
]

COLOURS = {
    "Boys": ["Orange", "Navy", "Green", "Yellow", "Red", "Black", "White"],
    "Girls": ["Pink", "Lavender", "Orange", "Navy", "Green", "Yellow", "Red", "Black", "White"],
}

COL = {
    "Pink": ("#FFA3B5", "soft rose pink #FFA3B5 (Pantone 1765C), clean pastel pink, not salmon, not peach"),
    "Lavender": ("#C7B2DE", "soft lavender #C7B2DE (Pantone 2071C), light cool purple, not grey, not blue"),
    "Orange": ("#FF671F", "bright orange #FF671F (Pantone 165C), vivid, not red, not rust"),
    "Navy": ("#0C2340", "deep navy blue #0C2340 (Pantone 289C), rich dark navy, NOT grey-blue, not faded, not black"),
    "Green": ("#6CC24A", "bright leaf green #6CC24A (Pantone 360C), fresh saturated green, NOT olive, not dull, not mint"),
    "Yellow": ("#FDDA24", "sunny yellow #FDDA24 (Pantone 115C), bright lemon-golden, not mustard"),
    "Red": ("#CD001A", "true red #CD001A (Pantone 3546C), not orange, not maroon, not pink"),
    "Black": ("#000000", "pure jet black #000000, deep true black, NOT charcoal, not faded grey; knit visible only in soft highlights"),
    "White": ("#FFFFFF", "bright optical white #FFFFFF, clean neutral white, NO grey cast, no blue or yellow tint, knit shown by soft shadows"),
}

POSES = {
    "02-threequarter": "turned three-quarter to the left, looking at camera, one hand lightly holding the shorts side, other arm relaxed and bent, mid-step stance, exactly two arms and two hands",
    "03-side": "in full side profile facing right, head turned slightly toward camera, arms relaxed with natural bend, walking step, exactly two arms and two hands, rear arm hidden behind body",
    "04-back": "seen from the back, full back of the top visible, looking back over the shoulder at camera with a smile, arms relaxed with slight bend, exactly two arms and two hands",
    "05-handpocket": "facing camera at a slight angle, both hands in shorts pockets, relaxed confident lean, cheerful smile",
}

STUDIO = "Smooth plain seamless light grey studio background with no texture, specks or gradient banding, soft even studio lighting, gentle shadows, true-to-life natural colours, photorealistic Myntra e-commerce catalogue photo, 85mm lens look, full body head to shoes with clear empty space above the head and below the shoes, garment sharp. Exactly one child, two arms, two hands, two legs. No logos, no text, no print on the top, no watermark."

FRONT = "01-front"


# --- Bottoms. ASSUMPTION: full-sleeve tees are styled with joggers, everything else with
# shorts; bottoms are navy (charcoal under navy/black tops). Replace with your own rules.
def pants(s):
    return s["style"] == "FullSleeveTee"


def legs(s):
    shade = "charcoal grey" if s["colour"] in ("Navy", "Black") else "navy blue"
    if pants(s):
        return f"Paired with plain {shade} cotton jogger pants with elastic cuffs and clean white sneakers. "
    return f"Paired with plain {shade} cotton knee-length shorts and clean white sneakers. "


# --- Prompts (as supplied) ---
def _sh(s, txt):
    if not pants(s):
        return txt
    return (txt.replace("shorts side", "jogger side").replace("shorts pockets", "jogger pockets")
            .replace("shorts pocket", "jogger pocket").replace("top and shorts", "top and joggers")
            .replace("pose, shorts,", "pose, joggers,"))


def kid_text(s):
    i = s["ident"]
    return (f"a {i['age']}-year-old {i['region']} Indian {'boy' if s['gender'] == 'Boys' else 'girl'}, {i['skin']} skin, {i['shape']}, "
            f"{i['hair']}, {' and '.join(i['feat'])}, {i['expr']}")


def garment(s):
    return (f"Wearing a {s['desc']}, 100% pure cotton, regular fit, colour {COL[s['colour']][1]}. Clearly visible fine single-jersey knit texture, matte dry cotton, "
            f"no sheen, soft natural folds, ribbed crew collar, crisp double-needle stitching visible on hems, solid uniform dye, clean fabric with no lint or specks. "
            + legs(s))


def front_prompt(s):
    lead = ("Use the reference portrait ONLY for the child's face identity (same face, features, skin tone, hairstyle); new full-body photo. " if s["face_ref"] else "")
    return _sh(s, lead + f"Photo of {kid_text(s)}, standing facing camera, one hand casually in shorts pocket, other arm relaxed with a soft bend, weight on one leg. ") + garment(s) + STUDIO


def pose_prompt(s, p):
    return _sh(s, "Use the reference photos: keep exactly the same child (same face, hair, skin tone), same top and shorts, same studio. The same child " + POSES[p] + ". ") + garment(s) + STUDIO


# --- SKUs and generation ---
def load_kids(path: str) -> dict:
    """JSON keyed by gender: {"Boys": {age, region, skin, shape, hair, feat: [...], expr, face_ref?}, ...}"""
    kids = json.loads(Path(path).read_text())
    base = Path(path).parent
    for ident in kids.values():
        if ident.get("face_ref"):
            ident["face_ref"] = str((base / ident["face_ref"]).resolve())
    return kids


def skus(kids: dict, gender=None, style=None, colour=None) -> Iterator[dict]:
    for g, st, desc in CATEGORIES:
        if g not in kids or (gender and g != gender) or (style and st != style):
            continue
        for c in COLOURS[g]:
            if colour and c != colour:
                continue
            ident = kids[g]
            yield {"gender": g, "style": st, "desc": desc, "colour": c,
                   "ident": ident, "face_ref": ident.get("face_ref")}


class CatalogAgent:
    def __init__(self, client=None, model: Optional[str] = None, output_dir="output",
                 quality="high", size="1024x1536", fidelity: Optional[str] = None):
        from .agent import DEFAULT_MODEL
        import os

        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model or os.environ.get("JAGIMAGE_MODEL", DEFAULT_MODEL)
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.size = size
        self.fidelity = fidelity

    def folder(self, s) -> Path:
        return self.output_dir / s["gender"] / s["style"] / s["colour"]

    def _call(self, prompt: str, refs: List[str]) -> bytes:
        params = dict(model=self.model, prompt=prompt, size=self.size, quality=self.quality,
                      background="opaque", output_format="png", n=1)
        if refs:
            if self.fidelity:
                params["input_fidelity"] = self.fidelity
            with ExitStack() as stack:
                files = [stack.enter_context(open(r, "rb")) for r in refs]
                result = self.client.images.edit(image=files, **params)
        else:
            result = self.client.images.generate(**params)
        return base64.b64decode(result.data[0].b64_json)

    def run_sku(self, s, poses=None, force=False) -> Iterator[tuple]:
        """Yields (shot name, path, 'made'|'skipped'). Existing files are kept unless force=True,
        so an interrupted run resumes where it stopped."""
        folder = self.folder(s)
        folder.mkdir(parents=True, exist_ok=True)
        front = folder / f"{FRONT}.png"
        if force or not front.exists():
            refs = [s["face_ref"]] if s["face_ref"] else []
            front.write_bytes(self._call(front_prompt(s), refs))
            yield FRONT, front, "made"
        else:
            yield FRONT, front, "skipped"
        for p in poses or POSES:
            path = folder / f"{p}.png"
            if not force and path.exists():
                yield p, path, "skipped"
                continue
            refs = [str(front)] + ([s["face_ref"]] if s["face_ref"] else [])
            path.write_bytes(self._call(pose_prompt(s, p), refs))
            yield p, path, "made"
