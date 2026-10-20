import json

import pytest

from jagimage import library as L
from jagimage.cli import main

# Verbatim copies from the SPB Textile prompt library document.
MEN_HALF_PC_FRONT = "Use the reference image ONLY for identity: keep exactly the same face, facial features, skin tone, hairstyle, hair colour and hair length. Create a new full-body photo. Photo of an Indian man aged 22-32, lean-athletic or regular build, fair to wheatish even skin, short textured crop with tapered sides, neat light stubble, friendly natural expression, standing straight, facing the camera squarely, weight slightly on one leg, both arms relaxed and natural with a soft bend at the elbow and a small visible gap between arm and torso, open relaxed hands, looking at the camera with a gentle smile. Wearing a plain {COLOUR} regular-fit crew-neck half-sleeve t-shirt, 1x1 ribbed neckband, set-in shoulder seam at the natural shoulder, sleeves ending mid-bicep, straight hem at the hip, worn untucked, no print, no logo, no pocket, fabric: 100% pure cotton single jersey knit, 180-200 GSM combed cotton, fine visible knit texture, soft matte finish, neat visible stitching at neckband, shoulder seams and hems, paired with straight-leg light-wash blue denim jeans (dark tops) or black cotton trousers (light tops), clean white low-top sneakers. Plain seamless light grey studio background (#EDEDED), soft even shadow-free lighting. Exact garment colour {COLOUR} ({HEX}), uniform and true under neutral daylight-balanced 5500K studio light, no colour cast. Sharp HD face in crisp focus with natural skin texture, clear eyes, lashes and brows. Skin tone even and identical across face, ears, neck, hands and arms; no shadow under the chin or on the neck; natural neck-to-collar transition. Natural anatomy, correct face orientation, no twisted or distorted face, exactly two arms, two hands with five fingers. Full body head to shoes in frame with small margins, model about 85-90% of frame height, eye-level camera, 3:4 portrait, photorealistic e-commerce catalogue photo, no text, no logo, no watermark, no props, compliant with Amazon, Flipkart, Myntra and Meesho."

WOMEN_TRACK_PC_BACK = "Use the reference image ONLY for identity: keep exactly the same face, facial features, skin tone, hairstyle, hair colour and hair length. Create a new full-body photo. Also match image 2 (approved front of the same model and colour) for garment, colour and hairstyle. Photo of an Indian woman aged 21-30, regular slim build, fair to wheatish even skin, natural fresh makeup, nude lip, hair in soft loose waves or a sleek low ponytail kept off the neckline, small stud earrings only, TRUE rear view: model standing with the back fully to the camera, head facing straight away from the camera, NO face visible, not looking over the shoulder, arms relaxed at the sides with a slight bend, full back of the garment visible, hairstyle exactly the same as the reference seen from behind (same length, colour, parting, ties/accessories). Wearing plain {COLOUR} regular-fit trackpants/joggers, full length, covered elastic waistband with flat drawcord, two side seam pockets, straight-leg with a plain straight hem at the ankle (NO ribbed cuffs, no elastic gathering), no logo, fabric: 100% pure cotton French Terry knit, smooth matte face, soft drape, neat visible stitching at waistband and side seams, paired with a plain white crew-neck t-shirt, clean white low-top sneakers. Plain seamless light grey studio background (#EDEDED), soft even lighting. Exact garment colour {COLOUR} ({HEX}), uniform and true under neutral daylight-balanced 5500K studio light, no colour cast. Sharp HD face in crisp focus with natural skin texture, clear eyes, lashes and brows. Skin tone even and identical across face, ears, neck, hands and arms; no shadow under the chin or on the neck; natural neck-to-collar transition. Natural anatomy, correct face orientation, no twisted or distorted face, exactly two arms, two hands with five fingers. Full body head to shoes in frame with small margins, model about 85-90% of frame height, eye-level camera, 3:4 portrait, photorealistic e-commerce catalogue photo, no text, no logo, no watermark, no props, compliant with Amazon, Flipkart, Myntra and Meesho. Joggers end in a plain straight hem, no ribbed cuffs."

BOYS_HOODIE_CM_HANDPOCKET = "Use the reference image ONLY for identity: keep exactly the same face, facial features, skin tone, hairstyle, hair colour and hair length. Create a new full-body photo. Also match image 2 (approved front of the same model and colour) for garment, colour and hairstyle. Photo of a small 4-5 year old Indian boy, toddler proportions: baby face, round chubby cheeks, soft small nose, big bright eyes, large head relative to a small short body (head about one fifth of body height), short chubby limbs, fair to wheatish even skin, short neat kids haircut, cheerful innocent smile, facing the camera at a slight angle, one hand casually in the side/front pocket, the other arm relaxed with a soft bend, relaxed weight shift, gentle smile. Wearing a plain {COLOUR} regular-fit pullover hoodie, lined double-layer hood with flat drawcords, front kangaroo pocket, ribbed cuffs and ribbed hem band, no print, no logo, fabric: Cotton Mix poly-cotton brushed fleece, 280-300 GSM, smooth even face, structured shape, neat visible stitching, paired with matching-colour Cotton Mix French Terry shorts, small white kids sneakers. Plain seamless light grey studio background (#EDEDED), soft even lighting. Exact garment colour {COLOUR} ({HEX}), uniform and true under neutral daylight-balanced 5500K studio light, no colour cast. Sharp HD face in crisp focus with natural skin texture, clear eyes, lashes and brows. Skin tone even and identical across face, ears, neck, hands and arms; no shadow under the chin or on the neck; natural neck-to-collar transition. Natural anatomy, correct face orientation, no twisted or distorted face, exactly two arms, two hands with five fingers. Full body head to shoes in frame with small margins, model about 85-90% of frame height, eye-level camera, 3:4 portrait, photorealistic e-commerce catalogue photo, no text, no logo, no watermark, no props, compliant with Amazon, Flipkart, Myntra and Meesho. The child must clearly look 4-5 years old with toddler proportions."

GIRLS_SLEEVELESS_PC_WHITE = "Use the reference image ONLY for identity: keep exactly the same face, facial features, skin tone, hairstyle, hair colour and hair length. Create a new full-body photo. Photo of a small 4-5 year old Indian girl, toddler proportions: baby face, round chubby cheeks, soft small nose, big bright eyes, large head relative to a small short body (head about one fifth of body height), short chubby limbs, fair to wheatish even skin, hairstyle exactly as in the reference (e.g. short bob, pigtails or ponytail) kept identical in every pose, cheerful innocent smile, standing straight, facing the camera squarely, weight slightly on one leg, both arms relaxed and natural with a soft bend at the elbow and a small visible gap between arm and torso, open relaxed hands, looking at the camera with a gentle smile. Wearing a plain {COLOUR} regular-fit crew-neck sleeveless vest/tank tee with neat bound armholes, straight hem, worn untucked, no print, no logo, no pocket, fabric: 100% pure cotton single jersey knit, 180-200 GSM combed cotton, fine visible knit texture, soft matte finish, neat visible stitching at neckband, shoulder seams and hems, paired with matching-colour Cotton Mix French Terry shorts, small white kids sneakers. Pure white seamless studio background (RGB 255,255,255), soft even shadow-free lighting, marketplace main-image compliant. Exact garment colour {COLOUR} ({HEX}), uniform and true under neutral daylight-balanced 5500K studio light, no colour cast. Sharp HD face in crisp focus with natural skin texture, clear eyes, lashes and brows. Skin tone even and identical across face, ears, neck, hands and arms; no shadow under the chin or on the neck; natural neck-to-collar transition. Natural anatomy, correct face orientation, no twisted or distorted face, exactly two arms, two hands with five fingers. Full body head to shoes in frame with small margins, model about 85-90% of frame height, eye-level camera, 3:4 portrait, photorealistic e-commerce catalogue photo, no text, no logo, no watermark, no props, compliant with Amazon, Flipkart, Myntra and Meesho. The child must clearly look 4-5 years old with toddler proportions."

BOYS_FULL_PC_FRONT = "Use the reference image ONLY for identity: keep exactly the same face, facial features, skin tone, hairstyle, hair colour and hair length. Create a new full-body photo. Photo of a small 4-5 year old Indian boy, toddler proportions: baby face, round chubby cheeks, soft small nose, big bright eyes, large head relative to a small short body (head about one fifth of body height), short chubby limbs, fair to wheatish even skin, short neat kids haircut, cheerful innocent smile, standing straight, facing the camera squarely, weight slightly on one leg, both arms relaxed and natural with a soft bend at the elbow and a small visible gap between arm and torso, open relaxed hands, looking at the camera with a gentle smile. Wearing a plain {COLOUR} crew-neck full-sleeve t-shirt with plain flat hemmed sleeve ends, worn with matching {COLOUR} Cotton Mix French Terry full-length joggers with a plain straight hem at the ankle (NO ribbed cuffs, no elastic gathering), fabric: 100% pure cotton single jersey knit, 180-200 GSM combed cotton, fine visible knit texture, soft matte finish, neat visible stitching at neckband, shoulder seams and hems, paired with matching-colour Cotton Mix French Terry shorts, small white kids sneakers. Plain seamless light grey studio background (#EDEDED), soft even shadow-free lighting. Exact garment colour {COLOUR} ({HEX}), uniform and true under neutral daylight-balanced 5500K studio light, no colour cast. Sharp HD face in crisp focus with natural skin texture, clear eyes, lashes and brows. Skin tone even and identical across face, ears, neck, hands and arms; no shadow under the chin or on the neck; natural neck-to-collar transition. Natural anatomy, correct face orientation, no twisted or distorted face, exactly two arms, two hands with five fingers. Full body head to shoes in frame with small margins, model about 85-90% of frame height, eye-level camera, 3:4 portrait, photorealistic e-commerce catalogue photo, no text, no logo, no watermark, no props, compliant with Amazon, Flipkart, Myntra and Meesho. The child must clearly look 4-5 years old with toddler proportions. Joggers end in a plain straight hem, no ribbed cuffs."

BOYS_FULL_NEGATIVE = "blurry face, soft face, low resolution, waxy plastic skin, over-smoothed skin, grey or discoloured ears, dark neck shadow, black shadow under chin, mismatched face and body skin tone, orange neck, twisted face, head turned opposite to body, distorted face, extra limbs, extra fingers, missing fingers, fused hands, deformed hands, stiff arms pinned to body, arms glued to sides, military stance, wrong garment colour, colour shift, faded colour, print, logo, graphic, text, watermark, chest pocket, extra layers, undershirt, wrinkled messy garment, cropped head, cut-off feet, busy background, props, gradient background, harsh shadows, vignette, collage, frame, border, cartoon, illustration, 3D render, CGI, adult face, teenager, older child, mature proportions, long legs, slim adult body, makeup on child, ribbed cuffs on joggers, elastic ankle cuffs"


@pytest.mark.parametrize("args,expected", [
    (("Men", "HalfSleeveTee", "PureCotton", "front_main"), MEN_HALF_PC_FRONT),
    (("Women", "Trackpants", "PureCotton", "back"), WOMEN_TRACK_PC_BACK),
    (("Kids_Boys", "Hoodie", "CottonMix", "handpocket"), BOYS_HOODIE_CM_HANDPOCKET),
    (("Kids_Girls", "SleevelessTee", "PureCotton", "front_main_white"), GIRLS_SLEEVELESS_PC_WHITE),
])
def test_matches_library_document(args, expected):
    assert L.build_prompt(*args) == expected


def test_kids_fullsleeve_set_drops_contradictory_shorts():
    expected = BOYS_FULL_PC_FRONT.replace(
        "paired with matching-colour Cotton Mix French Terry shorts, small white kids sneakers",
        "paired with small white kids sneakers")
    assert L.build_prompt("Kids_Boys", "FullSleeveTee", "PureCotton", "front_main") == expected


def test_negatives():
    assert L.build_negative("Kids_Boys", "FullSleeveTee", "front_main") == BOYS_FULL_NEGATIVE
    assert L.build_negative("Men", "Hoodie", "back").endswith("CGI, visible face, looking over shoulder, face in back view, different hairstyle")


def test_colour_fill_and_girls_hairstyle():
    p = L.build_prompt("Kids_Girls", "HalfSleeveTee", "CottonMix", "side", colour="Navy Blue", hairstyle="two pigtails with pink ties")
    assert "Exact garment colour Navy Blue (#1F3A5F)" in p and "{COLOUR}" not in p
    assert "hairstyle exactly as in the reference (two pigtails with pink ties)" in p


def test_counts_and_export(tmp_path):
    assert len(list(L.items())) == 4 * 7 * (9 + 17)
    data = L.export(str(tmp_path / "prompts.json"))
    assert len(data) == 28 and set(data["Men_Hoodie"]) == {"PureCotton", "CottonMix"}
    assert set(data["Men_Hoodie"]["CottonMix"]) == set(L.POSES)
    assert json.loads((tmp_path / "prompts.json").read_text())["Men_HalfSleeveTee"]["PureCotton"]["front_main"]["prompt"] == MEN_HALF_PC_FRONT


class FakeAgent:
    def __init__(self):
        self.calls = []

    def _call(self, prompt, refs):
        self.calls.append((prompt, refs))
        return b"png"


def test_staged_run_uses_face_then_front(tmp_path):
    face = tmp_path / "men.jpg"
    face.write_bytes(b"x")
    agent = FakeAgent()
    runner = L.LibraryRunner(agent, {"Men": [{"face": str(face)}]}, out=str(tmp_path / "out"))
    it = {"category": "Men", "product": "Hoodie", "fabric": "CottonMix", "colour": "Navy Blue"}
    poses = list(L.POSES)

    assert [s for _, _, s in runner.run(it, poses, stage="poses")] == ["waiting for front_main"] * 4
    fronts = list(runner.run(it, poses, stage="fronts"))
    assert [p for p, _, _ in fronts] == ["front_main", "front_main_white"]
    assert all(refs == [str(face)] for _, refs in agent.calls)
    list(runner.run(it, poses, stage="poses"))
    front = str(tmp_path / "out/Men/Hoodie/CottonMix/Navy_Blue/front_main.png")
    assert agent.calls[2][1] == [str(face), front]
    assert agent.calls[2][0].endswith("different hairstyle.") is False  # threequarter has no back extras
    assert " Avoid: blurry face" in agent.calls[0][0]
    assert {s for _, _, s in runner.run(it, poses)} == {"skipped"}


def test_cli_library_dry_run_and_export(tmp_path, capsys):
    assert main(["library", "--category", "Men", "--product", "Shorts", "--fabric", "PureCotton",
                 "--colour", "Red", "--poses", "front_main", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 SKUs" in out and "Exact garment colour Red (#CD001A)" in out
    assert main(["library", "--export", str(tmp_path / "p.json")]) == 0
    assert (tmp_path / "p.json").exists()


def test_each_sku_gets_a_different_model_and_keeps_it(tmp_path):
    faces = []
    for i in range(3):
        f = tmp_path / f"m{i}.jpg"
        f.write_bytes(b"x")
        faces.append(f"m{i}.jpg")
    (tmp_path / "models.json").write_text(json.dumps({"Men": {"faces": faces}}))
    models = L.load_models(str(tmp_path / "models.json"))
    runner = L.LibraryRunner(FakeAgent(), models, out=str(tmp_path / "out"))
    skus = list(L.items(["Men"], ["Hoodie"], ["PureCotton"]))[:3]
    assert len({runner.identity(it)["face"] for it in skus}) == 3
    # same SKU -> same model regardless of filters
    one = next(L.items(["Men"], ["Hoodie"], ["PureCotton"], ["Red"]))
    assert runner.identity(one) == runner.identity([s for s in L.items(["Men"], ["Hoodie"], ["PureCotton"]) if s["colour"] == "Red"][0])
    # the model saved with the front wins for the poses, even if the pool changes
    list(runner.run(one, ["front_main"]))
    saved = runner.identity(one)["face"]
    runner.models = {"Men": [{"face": "other.jpg"}]}
    assert runner.identity(one)["face"] == saved
    # product mode: one model across colours
    prod = L.LibraryRunner(None, models, out=str(tmp_path / "o2"), face_by="product")
    assert len({prod.identity(it)["face"] for it in skus}) == 1


def test_no_face_photo_makes_new_face_then_copies_it(tmp_path):
    agent = FakeAgent()
    runner = L.LibraryRunner(agent, L.load_models(None), out=str(tmp_path))
    it = {"category": "Kids_Girls", "product": "Shorts", "fabric": "CottonMix", "colour": "Pink"}
    list(runner.run(it, ["front_main", "back"]))
    (front_prompt, front_refs), (back_prompt, back_refs) = agent.calls
    assert front_refs == [] and front_prompt.startswith("Photo of a small 4-5 year old Indian girl")
    assert "exactly as in the reference" not in front_prompt.split(" Avoid:")[0]
    assert back_refs == [str(tmp_path / "Kids_Girls/Shorts/CottonMix/Pink/front_main.png")]
    assert back_prompt.startswith(L.FRONT_ONLY_LINE)


def test_look_is_added_to_model_description():
    p = L.build_prompt("Men", "Hoodie", "CottonMix", "front_main", "Grey", look="oval face, thick wavy hair", face=False)
    assert "friendly natural expression, oval face, thick wavy hair, standing straight" in p
