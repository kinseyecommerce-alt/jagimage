"""SPB Textile image prompt library: Men, Women, Kids Boys, Kids Girls x 7 products x
Pure Cotton / Cotton Mix x colour x 6 poses.

Prompts are assembled from the library's parts with the master template:
    {REFERENCE_LINE}Photo of {MODEL}, {POSE}. Wearing {GARMENT}, fabric: {FABRIC}, paired with {PAIRING}. {BACKGROUND} {RULES}
Identity flow: front_main (and front_main_white) use the face reference only; every
other pose sends image 1 = face reference, image 2 = the approved front_main.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

CATEGORIES = ["Men", "Women", "Kids_Boys", "Kids_Girls"]
PRODUCTS = ["HalfSleeveTee", "FullSleeveTee", "SleevelessTee", "Shorts", "Trackpants", "Hoodie", "Sweatshirt"]
FABRICS = {"PureCotton": "Pure Cotton (100% cotton)", "CottonMix": "Cotton Mix (PC, poly-cotton blend)"}

COLOURS = {
    "PureCotton": {
        "Black": "#000000", "White": "#FFFFFF", "Light Pink": "#FFA3B5", "Red": "#CD001A", "Navy": "#0C2340",
        "Green": "#6CC24A", "Lavender": "#C7B2DE", "Orange": "#FF671F", "Yellow": "#FDDA24",
    },
    "CottonMix": {
        "Black": "#151515", "Blue": "#004DBE", "Maroon": "#490D14", "Melange": "#C9C7C9", "White": "#E7E7EC",
        "Red": "#CF0B15", "Yellow": "#F9A712", "Light Pink": "#EBBECE", "Dark Pink": "#E6276A",
        "Navy Blue": "#1F3A5F", "Olive": "#556B2F", "Beige": "#D8C8A8", "Grey": "#8E8E90",
        "Royal Blue": "#2A4BA8", "Military Green": "#4B5320", "Pink": "#E8A0B8", "Green": "#2E7D32",
    },
}

MODELS = {
    "Men": "an Indian man aged 22-32, lean-athletic or regular build, fair to wheatish even skin, short textured crop with tapered sides, neat light stubble, friendly natural expression",
    "Women": "an Indian woman aged 21-30, regular slim build, fair to wheatish even skin, natural fresh makeup, nude lip, hair in soft loose waves or a sleek low ponytail kept off the neckline, small stud earrings only",
    "Kids_Boys": "a small 4-5 year old Indian boy, toddler proportions: baby face, round chubby cheeks, soft small nose, big bright eyes, large head relative to a small short body (head about one fifth of body height), short chubby limbs, fair to wheatish even skin, short neat kids haircut, cheerful innocent smile",
    "Kids_Girls": "a small 4-5 year old Indian girl, toddler proportions: baby face, round chubby cheeks, soft small nose, big bright eyes, large head relative to a small short body (head about one fifth of body height), short chubby limbs, fair to wheatish even skin, hairstyle exactly as in the reference (e.g. short bob, pigtails or ponytail) kept identical in every pose, cheerful innocent smile",
}
GIRLS_HAIR_EXAMPLE = "(e.g. short bob, pigtails or ponytail)"

FABRIC_TEXT = {
    "PureCotton": {
        "top": "100% pure cotton single jersey knit, 180-200 GSM combed cotton, fine visible knit texture, soft matte finish, neat visible stitching at neckband, shoulder seams and hems",
        "fleece": "100% pure cotton loopback French Terry fleece, 280-320 GSM, smooth matte face, soft structured drape, neat visible stitching",
        "bottom": "100% pure cotton French Terry knit, smooth matte face, soft drape, neat visible stitching at waistband and side seams",
    },
    "CottonMix": {
        "top": "Cotton Mix poly-cotton single jersey knit, 160-180 GSM, smooth fine knit with a very subtle sheen, crisp shape, neat visible stitching at neckband, shoulder seams and hems",
        "fleece": "Cotton Mix poly-cotton brushed fleece, 280-300 GSM, smooth even face, structured shape, neat visible stitching",
        "bottom": "Cotton Mix poly-cotton French Terry knit, smooth even surface, slight structured drape, neat visible stitching at waistband and side seams",
    },
}
FABRIC_KIND = {"HalfSleeveTee": "top", "FullSleeveTee": "top", "SleevelessTee": "top", "Shorts": "bottom",
               "Trackpants": "bottom", "Hoodie": "fleece", "Sweatshirt": "fleece"}

GARMENTS = {
    "HalfSleeveTee": "a plain {COLOUR} regular-fit crew-neck half-sleeve t-shirt, 1x1 ribbed neckband, set-in shoulder seam at the natural shoulder, sleeves ending mid-bicep, straight hem at the hip, worn untucked, no print, no logo, no pocket",
    "FullSleeveTee": "a plain {COLOUR} regular-fit crew-neck full-sleeve t-shirt, ribbed neckband, sleeves ending at the wrist bone with a plain flat hemmed sleeve end, straight hem at the hip, worn untucked, no print, no logo, no pocket",
    "SleevelessTee": "a plain {COLOUR} regular-fit crew-neck sleeveless vest/tank tee with neat bound armholes, straight hem, worn untucked, no print, no logo, no pocket",
    "Shorts": "plain {COLOUR} regular-fit knit shorts ending just above the knee, covered elastic waistband with flat drawcord, two side seam pockets, straight flat hem, no logo",
    "Trackpants": "plain {COLOUR} regular-fit trackpants/joggers, full length, covered elastic waistband with flat drawcord, two side seam pockets, straight-leg with a plain straight hem at the ankle (NO ribbed cuffs, no elastic gathering), no logo",
    "Hoodie": "a plain {COLOUR} regular-fit pullover hoodie, lined double-layer hood with flat drawcords, front kangaroo pocket, ribbed cuffs and ribbed hem band, no print, no logo",
    "Sweatshirt": "a plain {COLOUR} regular-fit crew-neck sweatshirt, ribbed neckband, ribbed cuffs and ribbed hem band, set-in sleeves, no print, no logo, no pocket",
}
KIDS_FULLSLEEVE_SET = "a plain {COLOUR} crew-neck full-sleeve t-shirt with plain flat hemmed sleeve ends, worn with matching {COLOUR} Cotton Mix French Terry full-length joggers with a plain straight hem at the ankle (NO ribbed cuffs, no elastic gathering)"

PAIRING = {
    ("Men", "top"): "straight-leg light-wash blue denim jeans (dark tops) or black cotton trousers (light tops), clean white low-top sneakers",
    ("Men", "bottom"): "a plain white crew-neck t-shirt tucked loosely, clean white low-top sneakers",
    ("Women", "top"): "straight-leg light-wash blue denim jeans (dark tops) or black trousers (light tops), clean white low-top sneakers",
    ("Women", "bottom"): "a plain white crew-neck t-shirt, clean white low-top sneakers",
    ("Kids", "top"): "matching-colour Cotton Mix French Terry shorts, small white kids sneakers",
    ("Kids", "bottom"): "a plain white kids crew-neck t-shirt, small white kids sneakers",
}
# The library's Kids FullSleeveTee prompts say "worn with ... joggers" and then "paired with
# matching-colour ... shorts". The set already includes the joggers, so only shoes are added here.
KIDS_FULLSLEEVE_PAIRING = "small white kids sneakers"

FRONT_POSE = "standing straight, facing the camera squarely, weight slightly on one leg, both arms relaxed and natural with a soft bend at the elbow and a small visible gap between arm and torso, open relaxed hands, looking at the camera with a gentle smile"
POSES = {
    "front_main": FRONT_POSE,
    "front_main_white": FRONT_POSE,
    "threequarter": "body turned about 45 degrees to the left, head and face turned the same direction as the body and gently looking at the camera, arms relaxed with a natural bend, exactly two arms and two hands",
    "side": "TRUE side profile facing right: whole body AND head turned 90 degrees in the same direction, face seen in clean profile (one eye, nose and chin silhouette), head NOT turned back toward the camera, arms relaxed hanging naturally with a slight bend, rear arm partly hidden behind the body, same hairstyle as reference seen from the side",
    "back": "TRUE rear view: model standing with the back fully to the camera, head facing straight away from the camera, NO face visible, not looking over the shoulder, arms relaxed at the sides with a slight bend, full back of the garment visible, hairstyle exactly the same as the reference seen from behind (same length, colour, parting, ties/accessories)",
    "handpocket": "facing the camera at a slight angle, one hand casually in the side/front pocket, the other arm relaxed with a soft bend, relaxed weight shift, gentle smile",
}
FRONTS = ("front_main", "front_main_white")

GREY_FRONT = "Plain seamless light grey studio background (#EDEDED), soft even shadow-free lighting."
BACKGROUNDS = {
    "front_main": GREY_FRONT,
    "front_main_white": "Pure white seamless studio background (RGB 255,255,255), soft even shadow-free lighting, marketplace main-image compliant.",
}
GREY = "Plain seamless light grey studio background (#EDEDED), soft even lighting."

REFERENCE_LINE = "Use the reference image ONLY for identity: keep exactly the same face, facial features, skin tone, hairstyle, hair colour and hair length. Create a new full-body photo. "
MATCH_FRONT_LINE = "Also match image 2 (approved front of the same model and colour) for garment, colour and hairstyle. "

RULES = ("Exact garment colour {COLOUR} ({HEX}), uniform and true under neutral daylight-balanced 5500K studio light, no colour cast. "
         "Sharp HD face in crisp focus with natural skin texture, clear eyes, lashes and brows. "
         "Skin tone even and identical across face, ears, neck, hands and arms; no shadow under the chin or on the neck; natural neck-to-collar transition. "
         "Natural anatomy, correct face orientation, no twisted or distorted face, exactly two arms, two hands with five fingers. "
         "Full body head to shoes in frame with small margins, model about 85-90% of frame height, eye-level camera, 3:4 portrait, "
         "photorealistic e-commerce catalogue photo, no text, no logo, no watermark, no props, compliant with Amazon, Flipkart, Myntra and Meesho.")
KIDS_RULE = " The child must clearly look 4-5 years old with toddler proportions."
JOGGER_RULE = " Joggers end in a plain straight hem, no ribbed cuffs."

NEGATIVE = ("blurry face, soft face, low resolution, waxy plastic skin, over-smoothed skin, grey or discoloured ears, dark neck shadow, "
            "black shadow under chin, mismatched face and body skin tone, orange neck, twisted face, head turned opposite to body, "
            "distorted face, extra limbs, extra fingers, missing fingers, fused hands, deformed hands, stiff arms pinned to body, "
            "arms glued to sides, military stance, wrong garment colour, colour shift, faded colour, print, logo, graphic, text, "
            "watermark, chest pocket, extra layers, undershirt, wrinkled messy garment, cropped head, cut-off feet, busy background, "
            "props, gradient background, harsh shadows, vignette, collage, frame, border, cartoon, illustration, 3D render, CGI")
NEGATIVE_KIDS = "adult face, teenager, older child, mature proportions, long legs, slim adult body, makeup on child"
NEGATIVE_JOGGERS = "ribbed cuffs on joggers, elastic ankle cuffs"
NEGATIVE_POSE = {
    "side": "face looking at camera, frontal face on side pose, head twisted toward camera",
    "back": "visible face, looking over shoulder, face in back view, different hairstyle",
}


def is_kids(category: str) -> bool:
    return category.startswith("Kids")


def has_joggers(category: str, product: str) -> bool:
    return product == "Trackpants" or (is_kids(category) and product == "FullSleeveTee")


# Used instead of REFERENCE_LINE + MATCH_FRONT_LINE when there is no face photo: the
# approved front (the only reference image) carries the model's identity.
FRONT_ONLY_LINE = ("Use the reference image (approved front of the same model and colour) for identity: keep exactly the same "
                   "face, facial features, skin tone, hairstyle, hair colour and hair length, and the same garment and colour. "
                   "Create a new full-body photo. ")


def model_text(category: str, hairstyle: Optional[str] = None, look: Optional[str] = None, has_reference=True) -> str:
    text = MODELS[category]
    if category == "Kids_Girls":
        if hairstyle:
            text = text.replace(GIRLS_HAIR_EXAMPLE, f"({hairstyle})")
        if not has_reference:
            text = text.replace("hairstyle exactly as in the reference ", "hairstyle ")
    if look:
        text += f", {look}"
    return text


def build_prompt(category, product, fabric, pose, colour=None, hairstyle=None, look=None, face=True, hex_=None) -> str:
    """Prompt for one cell of the library. Leaves {COLOUR}/{HEX} in place when colour is None.
    face=False: no face photo, so fronts invent a new model and poses copy it from the front."""
    kind = FABRIC_KIND[product]
    kids = is_kids(category)
    front = pose in FRONTS
    if kids and product == "FullSleeveTee":
        garment, pairing = KIDS_FULLSLEEVE_SET, KIDS_FULLSLEEVE_PAIRING
    else:
        garment = GARMENTS[product]
        pairing = PAIRING[("Kids" if kids else category, "bottom" if kind == "bottom" else "top")]
    if face:
        reference = REFERENCE_LINE + ("" if front else MATCH_FRONT_LINE)
    else:
        reference = "" if front else FRONT_ONLY_LINE
    model = model_text(category, hairstyle, look, has_reference=face or not front)
    text = (f"{reference}Photo of {model}, {POSES[pose]}. Wearing {garment}, "
            f"fabric: {FABRIC_TEXT[fabric][kind]}, paired with {pairing}. {BACKGROUNDS.get(pose, GREY)} {RULES}")
    if kids:
        text += KIDS_RULE
    if has_joggers(category, product):
        text += JOGGER_RULE
    if colour:
        text = text.replace("{COLOUR}", colour).replace("{HEX}", hex_ or COLOURS[fabric][colour])
    return text


def build_negative(category, product, pose) -> str:
    parts = [NEGATIVE]
    if is_kids(category):
        parts.append(NEGATIVE_KIDS)
    if has_joggers(category, product):
        parts.append(NEGATIVE_JOGGERS)
    if pose in NEGATIVE_POSE:
        parts.append(NEGATIVE_POSE[pose])
    return ", ".join(parts)


def export(path: str) -> dict:
    """Write prompts.json: category_product -> fabric -> pose -> {prompt, negative} (with {COLOUR}/{HEX})."""
    data = {
        f"{c}_{p}": {
            f: {pose: {"prompt": build_prompt(c, p, f, pose), "negative": build_negative(c, p, pose)} for pose in POSES}
            for f in FABRICS
        }
        for c in CATEGORIES for p in PRODUCTS
    }
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def load_models(path: Optional[str]) -> Dict[str, List[dict]]:
    """JSON keyed by category, each a list of models: {"Men": {"faces": [{"face": "faces/men1.jpg"},
    {"face": "faces/men2.jpg"}, {"look": "oval face, thick wavy hair"}]}}. Each entry may have
    face (photo), hairstyle and look. An entry without a face photo makes a brand-new face.
    Old style {"face_ref": ..., "hairstyle": ...} is still accepted. No file = new faces everywhere."""
    raw = json.loads(Path(path).read_text()) if path else {}
    base = Path(path).parent if path else Path(".")
    models = {}
    for category in CATEGORIES:
        cfg = raw.get(category, {})
        entries = cfg.get("faces") or ([{"face": cfg["face_ref"], "hairstyle": cfg.get("hairstyle")}] if cfg.get("face_ref") else [{}])
        pool = []
        for e in entries:
            e = {"face": e} if isinstance(e, str) else dict(e)
            if e.get("face"):
                e["face"] = str((base / e["face"]).resolve())
            pool.append(e)
        models[category] = pool
    return models


def items(categories=None, products=None, fabrics=None, colours=None) -> Iterator[dict]:
    for c in categories or CATEGORIES:
        for p in products or PRODUCTS:
            for f in fabrics or FABRICS:
                for colour in COLOURS[f]:
                    if colours and colour not in colours:
                        continue
                    yield {"category": c, "product": p, "fabric": f, "colour": colour}


def model_index(it: dict, face_by="sku") -> int:
    """Position of this SKU (or product+fabric) in the full, unfiltered category list, so the
    same SKU always gets the same model whatever filters a run uses."""
    keys = []
    for p in PRODUCTS:
        for f in FABRICS:
            if face_by == "product":
                keys.append((p, f))
            else:
                keys.extend((p, f, c) for c in COLOURS[f])
    key = (it["product"], it["fabric"]) if face_by == "product" else (it["product"], it["fabric"], it["colour"])
    return keys.index(key)


def folder_for(out: str, it: dict) -> Path:
    name = it.get("folder") or it["colour"].replace(" ", "_")
    return Path(out) / it["category"] / it["product"] / it["fabric"] / name


# ---- Two-base workflow: AI shoots each style in a light and a dark base colour, then every
# palette colour is made from the nearer base with exact-hex recolouring (jagimage.recolor).
# The bases are mid-saturation greens: easy to find in the photo (no green in skin, hair,
# grey studio, denim or black trousers) while keeping all the fold and knit detail.
BASES = {"light": ("Mint", "#8FD3A8"), "dark": ("Bottle Green", "#1E4636")}
LIGHT_FROM = 0.55  # targets with OKLab lightness at or above this use the light base


def base_item(it: dict, which: str, bases=None) -> dict:
    name, hex_ = (bases or BASES)[which]
    return {"category": it["category"], "product": it["product"], "fabric": it["fabric"],
            "colour": name, "hex": hex_, "base": which, "folder": f"_base_{which}"}


def base_for(it: dict) -> str:
    from .recolor import lightness

    return "light" if lightness(COLOURS[it["fabric"]][it["colour"]]) >= LIGHT_FROM else "dark"


def recolor_item(it: dict, poses: List[str], out="output", bases=None, mode="match", force=False,
                 canvas=None) -> Iterator[tuple]:
    """Make one palette colour from its base photos. Yields (pose, path, status, ΔE or None).
    A hand-corrected <pose>_mask.png in the base folder is used when present."""
    from PIL import Image

    from .recolor import WHITE_FROM, MaskError, fit_canvas, lightness, load_mask, recolor_image, save_mask, write_qc

    which = base_for(it)
    b = base_item(it, which, bases)
    bfolder, folder = folder_for(out, b), folder_for(out, it)
    folder.mkdir(parents=True, exist_ok=True)
    target = COLOURS[it["fabric"]][it["colour"]]
    qc_path = folder / "qc.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    for pose in poses:
        path, src = folder / f"{pose}.png", bfolder / f"{pose}.png"
        if path.exists() and not force:
            yield pose, path, "skipped", None
            continue
        if not src.exists():
            yield pose, path, f"waiting for {which} base", None
            continue
        mpath = bfolder / f"{pose}_mask.png"
        try:
            img, de, mask = recolor_image(Image.open(src), b["hex"], target, mode,
                                          mask=load_mask(mpath) if mpath.exists() else None)
        except MaskError as exc:
            yield pose, path, f"needs a mask: {exc}", None
            continue
        if not mpath.exists():
            save_mask(mpath, mask)
        img.save(path)
        if canvas:  # marketplace copy, e.g. front_main_1500x1500.jpg
            fit_canvas(img, *canvas).save(folder / f"{pose}_{canvas[0]}x{canvas[1]}.jpg", quality=95)
        qc[pose] = {"from": f"{which} base {b['hex']}", "target": target, "deltaE": round(de, 2), "on_shade": de < 3}
        if lightness(target) > WHITE_FROM:
            qc[pose]["note"] = "near-white: highlights are on the hex; folds show as shadow, so the average is slightly darker"
        write_qc(qc_path, qc)
        yield pose, path, "made", de


class LibraryRunner:
    """Generates library images. stage='fronts' makes front_main + front_main_white for review;
    stage='poses' makes the other poses from approved fronts; stage='all' does both.
    Each SKU gets its own model from the category's pool (rotating), kept across its 6 poses."""

    def __init__(self, agent, models: Dict[str, List[dict]], out="output", negative=True, face_by="sku", same_child=True):
        self.agent = agent  # anything with _call(prompt, refs) -> bytes (CatalogAgent)
        self.models = models
        self.out = out
        self.negative = negative
        self.face_by = face_by
        self.same_child = same_child

    def _light_front(self, it) -> Path:
        return Path(self.out) / it["category"] / it["product"] / it["fabric"] / "_base_light" / "front_main.png"

    def identity(self, it) -> dict:
        """The model for this SKU. Once a front exists, the identity saved with it wins, so
        poses always match the front even if the pool changes later."""
        saved = folder_for(self.out, it) / "model.json"
        if saved.exists():
            return json.loads(saved.read_text())
        pool = self.models.get(it["category"]) or [{}]
        if it.get("base"):
            light = pool[(model_index(it, "product") * 2) % len(pool)]
            if it["base"] == "light" or not self.same_child:
                return light if it["base"] == "light" else pool[(model_index(it, "product") * 2 + 1) % len(pool)]
            # same child in both bases: the dark shoot copies the light shoot's model. With no
            # face photo, the approved light-base front becomes the face reference.
            ident = dict(light)
            if not ident.get("face"):
                ident["face"] = str(self._light_front(it))
            return ident
        return pool[model_index(it, self.face_by) % len(pool)]

    def prompt(self, it, pose, ident=None) -> str:
        ident = ident if ident is not None else self.identity(it)
        text = build_prompt(it["category"], it["product"], it["fabric"], pose, it["colour"],
                            ident.get("hairstyle"), ident.get("look"), face=bool(ident.get("face")), hex_=it.get("hex"))
        if self.negative:  # the image API has no negative-prompt field, so it is appended as an instruction
            text += " Avoid: " + build_negative(it["category"], it["product"], pose) + "."
        return text

    def run(self, it, poses: List[str], stage="all", force=False) -> Iterator[tuple]:
        folder = folder_for(self.out, it)
        folder.mkdir(parents=True, exist_ok=True)
        front = folder / "front_main.png"
        if force and front.exists() and "front_main" in poses:
            (folder / "model.json").unlink(missing_ok=True)  # remaking the front may pick a new model
        if it.get("base") == "dark" and self.same_child and not (folder / "model.json").exists() \
                and not self._light_front(it).exists() and not (self.models.get(it["category"]) or [{}])[0].get("face"):
            for pose in poses:
                yield pose, folder / f"{pose}.png", "waiting for the light base front"
            return
        ident = self.identity(it)
        face = ident.get("face")
        for pose in poses:
            is_front = pose in FRONTS
            if (stage == "fronts" and not is_front) or (stage == "poses" and is_front):
                continue
            path = folder / f"{pose}.png"
            if path.exists() and not force:
                yield pose, path, "skipped"
                continue
            if is_front:
                refs = [face] if face else []
            elif front.exists():
                refs = ([face] if face else []) + [str(front)]
            else:
                yield pose, path, "waiting for front_main"
                continue
            path.write_bytes(self.agent._call(self.prompt(it, pose, ident), refs))
            if pose == "front_main":
                (folder / "model.json").write_text(json.dumps(ident, indent=2))
            yield pose, path, "made"
