"""The QC agent: checks every photo before it may go to Google Drive, and fixes what it can.

1. Base photos (the AI's shoot, before any recolouring): an AI vision check for what recolouring can't
   fix later: a tucked-in top, drawstrings or pockets on the bottoms, a white hood lining, hair bows
   in the clothes' colour, broken hands or faces. A failing base is asked for again.
2. Recoloured photos: the pixel check (colour bleed, base colour left). Bleed is put back to the photo's
   own pixels and leftover base colour is painted; what is still counted gets the agent's own retouch on
   the server (local_fix: missed fabric joins its garment, wrong shades take the garment's colour, bled
   colour is replaced by clean skin or backdrop), all without an AI call. Problems still counted after
   that are shown to the vision check, which tells real faults from specks nobody can see.
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


SHADE_HUE_MAX = 25      # degrees: a garment pixel further than this from the target's hue is a wrong shade
SHADE_CHROMA_MIN = 0.5  # ...or with less than this share of a colourful target's colour strength (a grey patch)
NEUTRAL_CHROMA = 0.035  # on black, white and grey targets: a pixel with more colour than this is a tinted patch


def local_fix(before: Image.Image, after: Image.Image, masks: list, targets: list, recolour: Callable):
    """The QC agent's own retouch, on the server (no AI call): for colour bleed and wrong shades only.
    1. Where top and bottom meet, the line between them is redrawn along the photo's edges (split_outfit), so
       fabric no mask took there (a grey strip along a sleeve) joins its garment; the photo is recoloured.
    2. Garment pixels whose shade is off the target (another hue, a grey or tinted patch) get the target's
       colour with their own lightness, so folds and knit stay.
    3. Colour bled onto skin or backdrop (as they were in the photo) gets those pixels back.
    recolour(list of masks) -> recoloured image. Returns (photo, masks)."""
    from .recolor import backdrop_mask, lab_to_rgb, render_target, rgb_to_lab, split_outfit

    lab0 = rgb_to_lab(np.asarray(before.convert("RGB")))
    H = lab0.shape[0]
    ms = [np.asarray(m, np.float32) for m in masks]
    regrown = False
    if len(ms) == 2:  # where top and bottom meet, the line follows the photo's edges (not the colour)
        t2, b2 = split_outfit(lab0, ms[0], ms[1])
        regrown = bool(((t2 > 0.5) != (ms[0] > 0.5)).any() or ((b2 > 0.5) != (ms[1] > 0.5)).any())
        ms = [t2, b2]
    if regrown:
        after = recolour(ms)
    lab = rgb_to_lab(np.asarray(after.convert("RGB")))
    C, Hh = np.hypot(lab[..., 1], lab[..., 2]), np.arctan2(lab[..., 2], lab[..., 1])
    for m, th in zip(ms, targets):  # 2: wrong shades inside each garment get its colour, keeping their lightness
        T = render_target(th)
        ct = float(np.hypot(T[1], T[2]))
        inside = m > 0.9
        if ct > 0.04:
            off = np.abs(np.angle(np.exp(1j * (Hh - np.arctan2(T[2], T[1]))))) > np.radians(SHADE_HUE_MAX)
            wrong = inside & (off | (C < SHADE_CHROMA_MIN * ct))
        else:
            wrong = inside & (C > NEUTRAL_CHROMA)
        wrong = ndimage.binary_opening(wrong, iterations=1)  # a patch, not single noisy pixels
        if wrong.any():
            a = ndimage.gaussian_filter(ndimage.binary_dilation(wrong, iterations=1).astype(np.float32), 1.0) * inside
            lab[..., 1] = lab[..., 1] * (1 - a) + T[1] * a
            lab[..., 2] = lab[..., 2] * (1 - a) + T[2] * a
    rgb = lab_to_rgb(lab).astype(np.float32)
    union = np.zeros(lab0.shape[:2], bool)
    for m in ms:
        union |= m > 0.5
    C0 = np.hypot(lab0[..., 1], lab0[..., 2])
    skin0 = (lab0[..., 1] > 0.015) & (lab0[..., 2] > -0.015) & (C0 > 0.03)
    ring = ndimage.binary_dilation(union, iterations=10) & ~ndimage.binary_dilation(union, iterations=2)
    moved = np.linalg.norm(lab - lab0, axis=-1) > 0.04
    bleed = ring & moved & (skin0 | backdrop_mask(lab0))  # only what was skin or backdrop in the photo
    if bleed.any():  # 3: skin and backdrop beside the clothes get their own pixels back
        src = np.asarray(before.convert("RGB"), np.float32)
        a = ndimage.gaussian_filter(bleed.astype(np.float32), 0.7)[..., None]
        rgb = rgb * (1 - a) + src * a
    return Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8)), ms


def passed(qc_file, pose: str) -> bool:
    """A finished photo may be uploaded: it passed the QC agent or a person approved it."""
    try:
        row = json.loads(open(qc_file).read()).get(pose) or {}
    except (OSError, ValueError):
        return False
    c = row.get("check") or {}
    return bool(c.get("approved") or (c.get("passed") and not c.get("held")))
