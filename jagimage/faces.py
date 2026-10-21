"""A different model for every number: studio batches ask for e.g. 300 models, and each one gets
its own combination of face shape, skin tone, hair, eyes and expression. Model i always gets the
same look, so a batch can be resumed or extended without the faces changing."""

from typing import Dict, List

# Who the model is, without the features that vary (those come from FEATURES).
PEOPLE = {
    "Men": "an Indian man aged 22-32, natural friendly expression",
    "Women": "an Indian woman aged 21-30, natural fresh makeup, nude lip, small stud earrings only",
    "Kids_Boys": ("a small 4-5 year old Indian boy, toddler proportions: baby face, large head relative "
                  "to a small short body (head about one fifth of body height), short chubby limbs"),
    "Kids_Girls": ("a small 4-5 year old Indian girl, toddler proportions: baby face, large head relative "
                   "to a small short body (head about one fifth of body height), short chubby limbs"),
}

SKIN = ["fair even skin", "light wheatish even skin", "wheatish even skin", "warm medium-brown even skin",
        "dusky even skin", "olive-toned even skin"]
NOSE = ["small button nose", "soft rounded nose", "slightly upturned nose", "small straight nose"]
KID_CHEEKS = ["rosy cheeks", "deep dimples", "a few light freckles across the nose", "smooth plain cheeks"]
KID_LIPS = ["full little lips", "thin lips", "cupid's-bow lips"]
ADULT_NOSE = ["straight nose", "slightly rounded nose", "narrow high-bridged nose", "broad soft nose"]
ADULT_LIPS = ["full lips", "medium lips", "thin lips"]
EYES = ["big round dark-brown eyes", "almond-shaped dark eyes", "large bright eyes with long lashes",
        "deep-set brown eyes", "wide-set expressive eyes"]

FEATURES: Dict[str, Dict[str, List[str]]] = {
    "Kids_Boys": {
        "face": ["round face with chubby cheeks", "oval face with soft cheeks", "heart-shaped face with a small pointed chin",
                 "soft square face with full cheeks"],
        "skin": SKIN,
        "hair": ["short neat crop", "side-parted short hair", "short spiky top", "soft bowl cut", "short curly hair",
                 "very short buzz cut", "slightly long messy fringe", "short wavy hair with a side sweep"],
        "eyes": EYES,
        "brows": ["thick dark eyebrows", "soft thin eyebrows", "naturally arched eyebrows"],
        "smile": ["wide toothy grin", "shy closed-mouth smile", "big dimpled smile", "playful half smile"],
        "nose": NOSE,
        "cheeks": KID_CHEEKS,
        "lips": KID_LIPS,
    },
    "Kids_Girls": {
        "face": ["round face with chubby cheeks", "oval face with soft cheeks", "heart-shaped face with a small pointed chin",
                 "soft square face with full cheeks"],
        "skin": SKIN,
        "hair": ["short bob with a straight fringe", "two high pigtails", "a single high ponytail", "two neat braids",
                 "shoulder-length curly hair", "half-up hair with a small clip", "short wavy bob", "two low pigtails with ribbons"],
        "eyes": EYES,
        "brows": ["thick dark eyebrows", "soft thin eyebrows", "naturally arched eyebrows"],
        "smile": ["wide toothy grin", "shy closed-mouth smile", "big dimpled smile", "playful half smile"],
        "nose": NOSE,
        "cheeks": KID_CHEEKS,
        "lips": KID_LIPS,
    },
    "Men": {
        "face": ["oval face", "square jaw", "round face", "long face with high cheekbones", "diamond-shaped face"],
        "skin": SKIN,
        "hair": ["short textured crop with tapered sides", "side-parted short hair", "short curly hair", "slicked-back medium hair",
                 "buzz cut", "wavy medium-length hair", "short quiff", "messy short fringe"],
        "beard": ["clean-shaven", "light stubble", "short trimmed beard", "full neat beard", "moustache with light stubble"],
        "eyes": EYES,
        "build": ["lean-athletic build", "regular build", "slim build", "broad athletic build"],
        "nose": ADULT_NOSE,
        "lips": ADULT_LIPS,
        "age": ["looks about 23", "looks about 26", "looks about 29", "looks about 32"],
    },
    "Women": {
        "face": ["oval face", "heart-shaped face", "round face", "long face with high cheekbones", "soft square face"],
        "skin": SKIN,
        "hair": ["soft loose waves kept off the neckline", "sleek low ponytail", "shoulder-length straight hair tucked behind the ears",
                 "short bob", "high bun", "long straight hair behind the shoulders", "curly hair tied back", "low braided bun"],
        "eyes": EYES,
        "brows": ["thick defined eyebrows", "soft natural eyebrows", "arched eyebrows"],
        "build": ["regular slim build", "petite build", "athletic build", "regular build"],
        "nose": ADULT_NOSE,
        "lips": ADULT_LIPS,
        "age": ["looks about 22", "looks about 25", "looks about 28", "looks about 30"],
    },
}


def capacity(category: str) -> int:
    """How many models of this category have a different combination of features."""
    n = 1
    for options in FEATURES[category].values():
        n *= len(options)
    return n


# Features that change fastest from one model number to the next come first.
ORDER = ["hair", "skin", "face", "eyes", "smile", "beard", "brows", "build"]


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


def describe(category: str, i: int) -> str:
    f = look(category, i)
    return PEOPLE[category] + ", " + ", ".join(f.values())


def hairstyle(category: str, i: int) -> str:
    return look(category, i)["hair"]
