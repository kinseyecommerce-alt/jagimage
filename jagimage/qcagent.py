"""The QC agent: checks every photo before it may go to Google Drive, and fixes what it can.

1. Base photos (the AI's shoot, before any recolouring): an AI vision check for what recolouring can't
   fix later: a tucked-in top, drawstrings or pockets on the bottoms, a white hood lining, hair bows
   in the clothes' colour, broken hands or faces. A failing base is asked for again.
2. Recoloured photos: the pixel check (colour bleed, base colour left). Bleed is put back to the photo's
   own pixels and leftover base colour is painted, then the photo is checked again. Problems still
   counted after that are shown to the vision check, which tells real faults (a grey strip on a sleeve)
   from specks nobody can see.
3. Only photos that passed (or that a person approved) are uploaded; the rest are held for review."""

import io
import json
from typing import Callable, List, Optional

import numpy as np
from PIL import Image
from scipy import ndimage

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}}},
          "required": ["ok", "problems"]}
SYSTEM = ("You are the quality inspector of a kids' and adults' clothing catalogue. You look at one AI-made photo and "
          "check ONLY the points you are given. Be strict about the clothes and fair about the rest: tiny specks a "
          "shopper would not notice at normal size are not problems. Answer as JSON: ok (true only if every point "
          "passes) and problems (one short plain sentence per failed point, saying where it is; empty when ok).")
BOTTOMS_POINT = "The shorts or pants have NO drawstring cords and NO pockets or pocket openings."
VIEW_HEIGHT = 1280  # px: photos are shown to the vision check at this height
FIX_ROUNDS = 1      # automatic fixes per recoloured photo before it is re-checked


def _jpeg(img: Image.Image, height: int = VIEW_HEIGHT) -> bytes:
    img = img.convert("RGB")
    if img.height > height:
        img = img.resize((round(img.width * height / img.height), height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def default_ask() -> Callable:
    """The app's AI (Gemini first, else OpenAI), as ask(message, image_bytes) -> {ok, problems}."""
    from .cloud import Agent

    agent = Agent()

    def ask(message: str, image: bytes) -> dict:
        return agent._ask(message, system=SYSTEM, schema=SCHEMA, images=[(image, "image/jpeg")])
    return ask


def base_points(spec, pose: str) -> List[str]:
    """What a base photo must get right, for this batch and pose."""
    from . import library as L

    points = []
    outfit = spec.shot == "outfit"
    if outfit:
        points.append("The top is worn untucked: its hem hangs over the waistband of the shorts or pants. FAIL if the top "
                      "is tucked in or the waistband sits on top of the top's hem.")
        if spec.bottom in ("Shorts", "Trackpants"):
            points.append(BOTTOMS_POINT)
    if spec.top in ("Hoodie", "ZipHoodie"):
        points.append("The inside of the hood is the same colour as the hoodie. FAIL if the hood lining is white or grey.")
    if spec.category == "Kids_Girls":
        points.append("No hair bows, ribbons or clips in the colour of the clothes (plain black hair elastics are fine).")
    points.append("Each garment is one plain, even colour with no print, logo or patches of another colour.")
    points.append("Natural body: two arms, hands with five fingers, no extra or merged limbs" +
                  ("" if pose == "back" else ", a natural undistorted face") + ".")
    if L.is_kids(spec.category):
        points.append("The model is a small child, not a teenager or adult.")
    return points


def photo_points(top: str, bottom: Optional[str], left: bool, bleed: bool) -> List[str]:
    clothes = f"the {top} top" + (f" and the {bottom} bottom" if bottom else "")
    points = [f"The clothes were recoloured to {clothes}. Every part of each garment is that colour, with its natural "
              "folds and shading. FAIL for any visible patch, strip or edge of a different colour on the clothes "
              "(grey, white, black, mint green, lilac, or the other garment's colour), for example along the inside "
              "of a sleeve, at the waistband, the cuffs or the hood."]
    if bleed:
        points.append("The clothes' colour has not spilled onto the skin, hair, shoes or the background. FAIL for a "
                      "visible coloured fringe or blotch beside the clothes.")
    return points


def inspect(ask: Callable, img: Image.Image, points: List[str]) -> dict:
    """{ok, problems} from the vision check, or {ok: None, problems: [why]} when it couldn't run."""
    message = "Check this photo for these points:\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))
    try:
        ans = ask(message, _jpeg(img))
    except Exception as exc:  # no key, no credits, network: the check didn't run
        return {"ok": None, "problems": [f"QC agent unavailable: {type(exc).__name__}: {str(exc)[:160]}"]}
    if not isinstance(ans, dict) or not isinstance(ans.get("ok"), bool):
        return {"ok": None, "problems": ["QC agent gave no clear answer"]}
    problems = [str(p) for p in ans.get("problems") or [] if str(p).strip()]
    ok = ans["ok"] and not problems  # "ok" with a problem listed is not ok
    return {"ok": ok, "problems": [] if ok else (problems or ["failed a check"])}


def fix_photo(before: Image.Image, after: Image.Image, check: dict, recolour: Callable) -> Image.Image:
    """The automatic fixes for what the pixel check found.
    bleed: skin or backdrop that took on the new colour gets its own pixels back (softly, so no hard edge).
    left:  garment pixels still in the base colour are added to the garment and the photo recoloured again;
           recolour(extra_mask) does that."""
    out = after
    left = np.asarray(check.get("_left", np.zeros(1))).astype(bool)
    if left.any():
        out = recolour(ndimage.binary_dilation(left, iterations=1))
    bleed = np.asarray(check.get("_bleed", np.zeros(1))).astype(bool)
    if bleed.any():
        a = ndimage.gaussian_filter(ndimage.binary_dilation(bleed, iterations=1).astype(np.float32), 0.8)[..., None]
        b = np.asarray(before.convert("RGB"), np.float32)
        o = np.asarray(out.convert("RGB"), np.float32)
        out = Image.fromarray(np.clip(np.round(o * (1 - a) + b * a), 0, 255).astype(np.uint8))
    return out


def fix_prompt(problems: List[str], top: str, thex: str, bottom: Optional[str], bhex: Optional[str]) -> str:
    colours = f"the top {top} ({thex})" + (f" and the bottom {bottom} ({bhex})" if bottom else "")
    return ("Edit image 1, a clothing catalogue photo. Fix ONLY these colour faults on the clothes: " + "; ".join(problems) +
            f". Paint those areas in the garment's own colour, {colours}, with the same natural folds, shading and knit "
            "texture as the rest of that garment, so every part of each garment is one even colour. Do not change the "
            "garment shapes or details, the child, face, hair, hands, shoes, background, framing or image size. Return one photo.")


def blend_fix(out: Image.Image, edited: Image.Image, masks: list, reach: int = 6) -> Image.Image:
    """Keeps the AI edit only on the clothes (and a few px around them, where a stray colour sat); the face,
    hair, hands and backdrop stay exactly as they were."""
    edited = edited.convert("RGB").resize(out.size, Image.LANCZOS)
    g = np.zeros(out.size[::-1], bool)
    for m in masks:
        g |= np.asarray(m) > 0.3
    a = ndimage.gaussian_filter(ndimage.binary_dilation(g, iterations=reach).astype(np.float32), 1.5)[..., None]
    o, e = np.asarray(out.convert("RGB"), np.float32), np.asarray(edited, np.float32)
    return Image.fromarray(np.clip(np.round(o * (1 - a) + e * a), 0, 255).astype(np.uint8))


def passed(qc_file, pose: str) -> bool:
    """A finished photo may be uploaded: it passed the QC agent or a person approved it."""
    try:
        row = json.loads(open(qc_file).read()).get(pose) or {}
    except (OSError, ValueError):
        return False
    c = row.get("check") or {}
    return bool(c.get("approved") or (c.get("passed") and not c.get("held")))
