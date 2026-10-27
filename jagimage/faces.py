"""A different model for every number: studio batches ask for e.g. 300 models, and each one gets
its own combination of face shape, skin tone, hair, eyes and expression. Model i always gets the
same look, so a batch can be resumed or extended without the faces changing."""

from typing import Dict, List

# Who the model is, without the features that vary (those come from FEATURES).
PEOPLE = {
    "Men": "an Indian man, natural friendly expression",
    "Women": "an Indian woman, natural fresh makeup, nude lip, small stud earrings only",
    "Kids_Boys": ("a small 4-5 year old Indian boy, toddler proportions: baby face, large head relative "
                  "to a small short body (head about one fifth of body height), short chubby limbs"),
    "Kids_Girls": ("a small 4-5 year old Indian girl, toddler proportions: baby face, large head relative "
                   "to a small short body (head about one fifth of body height), short chubby limbs"),
}

# Fair models only (the shop's choice). Six shades of fair, so faces still differ from model to model.
SKIN = ["fair even skin", "fair even skin with a warm golden undertone", "fair even skin with a soft pink undertone",
        "light fair even skin", "fair even skin with a neutral beige undertone", "fair even skin with a light peachy glow"]
NOSE = ["small button nose", "soft rounded nose", "slightly upturned nose", "small straight nose"]
KID_CHEEKS = ["rosy cheeks", "deep dimples", "a few light freckles across the nose", "smooth plain cheeks"]
KID_LIPS = ["full little lips", "thin lips", "cupid's-bow lips"]
ADULT_NOSE = ["straight nose", "slightly rounded nose", "narrow high-bridged nose", "broad soft nose"]
ADULT_LIPS = ["full lips", "medium lips", "thin lips"]
EYES = ["big round dark-brown eyes", "almond-shaped dark eyes", "large bright eyes with long lashes",
        "deep-set brown eyes", "wide-set expressive eyes"]

MARKS = ["no distinguishing marks", "a small beauty mole on one cheek", "a dimple in the chin", "deep dimples when smiling",
         "a few faint freckles across the nose", "a small mole above the lip"]
KID_MARKS = ["no distinguishing marks", "deep dimples when smiling", "a few faint freckles across the nose",
             "a tiny mole on one cheek", "a small gap between the front teeth"]

# Each feature is written so the AI can see it in a front photo; subtle ones (lips, nose) on their own
# leave the AI drawing the same "default" face, so the strong ones (hair, face shape, beard, age) lead.
FEATURES: Dict[str, Dict[str, List[str]]] = {
    "Kids_Boys": {
        "face": ["very round face with full chubby cheeks", "oval face with a narrow chin", "heart-shaped face with a small pointed chin",
                 "broad square face with a wide jaw", "long face with a high forehead"],
        "skin": SKIN,
        "hair": ["short neat crop", "side-parted short hair", "short spiky top", "soft bowl cut with a full fringe", "short tight curls",
                 "very short buzz cut", "slightly long messy fringe", "short wavy hair with a side sweep"],
        "eyes": EYES,
        "brows": ["thick dark straight eyebrows", "soft thin light eyebrows", "naturally arched eyebrows"],
        "smile": ["wide toothy grin", "shy closed-mouth smile", "big dimpled smile", "playful half smile"],
        "nose": NOSE,
        "cheeks": KID_CHEEKS,
        "marks": KID_MARKS,
        "lips": KID_LIPS,
    },
    "Kids_Girls": {
        "face": ["very round face with full chubby cheeks", "oval face with a narrow chin", "heart-shaped face with a small pointed chin",
                 "broad square face with a wide jaw", "long face with a high forehead"],
        "skin": SKIN,
        "hair": ["short bob with a straight fringe", "two high pigtails", "a single high ponytail", "two neat braids",
                 "shoulder-length curly hair", "half-up hair with a small clip", "short wavy bob", "two low pigtails with ribbons"],
        "eyes": EYES,
        "brows": ["thick dark straight eyebrows", "soft thin light eyebrows", "naturally arched eyebrows"],
        "smile": ["wide toothy grin", "shy closed-mouth smile", "big dimpled smile", "playful half smile"],
        "nose": NOSE,
        "cheeks": KID_CHEEKS,
        "marks": KID_MARKS,
        "lips": KID_LIPS,
    },
    "Men": {
        "face": ["oval face", "strong square jaw", "round full face with soft jaw", "long narrow face with high cheekbones",
                 "diamond-shaped face with a pointed chin", "wide face with a broad forehead"],
        "skin": SKIN,
        "hair": ["short textured crop with tapered sides", "neat side-parted short hair", "short tight curly hair",
                 "slicked-back medium-length hair", "very short buzz cut", "wavy medium-length hair falling over the forehead",
                 "tall short quiff", "messy short fringe", "high-and-tight military cut", "longer hair tied in a small man-bun"],
        "beard": ["clean-shaven", "light stubble", "short boxed beard", "full thick beard", "moustache only", "goatee with light stubble"],
        "age": ["looks about 21", "looks about 24", "looks about 28", "looks about 32", "looks about 36"],
        "build": ["lean build", "regular build", "slim build", "broad muscular build", "slightly heavy build"],
        "eyes": EYES,
        "brows": ["thick straight eyebrows", "medium natural eyebrows", "thin eyebrows", "heavy bushy eyebrows"],
        "marks": MARKS,
        "nose": ADULT_NOSE,
        "lips": ADULT_LIPS,
    },
    "Women": {
        "face": ["oval face", "heart-shaped face with a pointed chin", "round full face", "long narrow face with high cheekbones",
                 "soft square face with a defined jaw", "wide face with full cheeks"],
        "skin": SKIN,
        "hair": ["long open straight hair behind the shoulders", "short pixie cut", "chin-length bob with a fringe",
                 "shoulder-length open wavy hair", "high sleek ponytail", "long open curly hair behind the shoulders",
                 "middle-parted low bun", "side-swept long layered hair behind the shoulders", "single long braid down the back",
                 "lob (long bob) with soft waves"],
        "colour": ["jet-black hair", "dark brown hair", "black hair with soft brown highlights", "warm chestnut-brown hair"],
        "age": ["looks about 21", "looks about 24", "looks about 27", "looks about 30", "looks about 34"],
        "build": ["petite build", "regular slim build", "athletic build", "regular build", "curvy build"],
        "eyes": EYES,
        "brows": ["thick defined eyebrows", "soft natural eyebrows", "high arched eyebrows", "thin straight eyebrows"],
        "marks": MARKS,
        "nose": ADULT_NOSE,
        "lips": ADULT_LIPS,
    },
}


def capacity(category: str) -> int:
    """How many models of this category have a different combination of features."""
    n = 1
    for options in FEATURES[category].values():
        n *= len(options)
    return n


# Features that change fastest from one model number to the next come first.
ORDER = ["hair", "face", "beard", "colour", "age", "build", "brows", "marks", "smile", "eyes", "skin"]


def look(category: str, i: int) -> Dict[str, str]:
    """Model i's features. i-1 is written in mixed radix (hair fastest), and every digit is shifted by
    the digits before it. That mapping is one-to-one, so every number up to capacity() gets its own
    combination, and model i and i+1 differ in hair, skin, face shape and more, not just one feature."""
    feats = FEATURES[category]
    names = [n for n in ORDER if n in feats] + [n for n in feats if n not in ORDER]
    k, carry, pick = i - 1, 0, {}
    for name in names:
        options = feats[name]
        k, d = divmod(k, len(options))
        pick[name] = options[(d + carry) % len(options)]
        carry += d
    return {name: pick[name] for name in feats}


DISTINCT = "a distinctive individual with their own recognisable face, not a generic model face"
DISTINCT_KID = "a child with a look of their own, not a generic face"


def describe(category: str, i: int, hair=True) -> str:
    """hair=False when a reference photo sets the hair: a written hairstyle that differs from the photo
    makes some models (Gemini Flash) switch hairstyles between shots of the same person."""
    f = look(category, i)
    if not hair:
        f["hair"] = "hairstyle exactly as in the reference photo"
    return PEOPLE[category] + ", " + ", ".join(f.values()) + ", " + (DISTINCT_KID if category.startswith("Kids") else DISTINCT)


def hairstyle(category: str, i: int) -> str:
    return look(category, i)["hair"]
