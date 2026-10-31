"""Set plans from an Excel sheet: rows of (category, style, top colour, bottom colour, model image sets) become
Studio batches, one per category, style and set type, each with its exact colour pairs.

A style reads like "COORD-HOOD-TRACK PANT" or "SET-HALF SLEEVE-SHORTS": COORD is a coord set (top and bottom
in one colour), SET or COMBO a combo set (contrast colours). The same models wear every colour pair of a
style (colours are recoloured for free); a pair asked for on 45 sets goes on that style's first 45 models."""

import io
from typing import Dict, List, Tuple

from . import library as L
from . import studio

CATEGORY_NAMES = {"BOYS": "Kids_Boys", "KIDS BOYS": "Kids_Boys", "GIRLS": "Kids_Girls", "KIDS GIRLS": "Kids_Girls",
                  "MEN": "Men", "MENS": "Men", "WOMEN": "Women", "WOMENS": "Women"}
# checked in this order: "ZIP HOOD" before "HOOD", "TRACK PANT" and "PANT" are both track pants
STYLE_TOPS = [("ZIP HOOD", "ZipHoodie"), ("SWEATSHIRT", "Sweatshirt"), ("HOOD", "Hoodie"), ("SLEEVELESS", "SleevelessTee"),
              ("FULL SLEEVE", "FullSleeveTee"), ("HALF SLEEVE", "HalfSleeveTee"), ("OVERSIZED", "OversizedTee")]
STYLE_BOTTOMS = [("SHORTS", "Shorts"), ("JEANS", "Jeans"), ("PANT", "Trackpants")]
# the sheet's short names for the shop's colours (Settings > Colours)
COLOUR_ALIASES = {"GREY": "Melange", "GRAY": "Melange", "M.GREEN": "Green", "MILITARY GREEN": "Green",
                  "R.BLUE": "Blue", "ROYAL BLUE": "Blue", "D.PINK": "Dark Pink", "L.PINK": "Light Pink"}
MAX_FILE = 2_000_000
MAX_ROWS = 2000


def _norm(v) -> str:
    return " ".join(str(v or "").upper().replace("_", " ").split())


def _header(row) -> Dict[str, int]:
    """Column numbers by meaning, from a header row such as S.NO | CATEGORY | STYLE | TOP COLOR | ..."""
    cols = {}
    for i, v in enumerate(row):
        h = _norm(v)
        if "CATEGORY" in h:
            cols["category"] = i
        elif "STYLE" in h:
            cols["style"] = i
        elif "TOP" in h and "COLO" in h:
            cols["top"] = i
        elif "BOTTOM" in h and "COLO" in h:
            cols["bottom"] = i
        elif "QTY" in h or "QUANTITY" in h or "REQUIRED" in h:
            cols["qty"] = i
    return cols if {"category", "style", "top", "bottom", "qty"} <= set(cols) else {}


def colour_name(raw, fabric: str, category: str) -> str:
    """The shop's colour for a sheet colour name, or ValueError."""
    want = COLOUR_ALIASES.get(_norm(raw), _norm(raw))
    for name in L.colours_for(fabric, category):
        if _norm(name) == _norm(want):
            return name
    raise ValueError(f"colour {raw!r} isn't in the {fabric} colours for {category} (Settings > Colours)")


def style_parts(raw) -> Tuple[str, str, str]:
    """(set type, top, bottom) for a style such as "COORD-FULL SLEEVE-PANT"."""
    s = _norm(raw)
    kind = "coord" if s.startswith("COORD") else "combo" if s.startswith(("SET", "COMBO")) else ""
    top = next((t for key, t in STYLE_TOPS if key in s), None)
    bottom = next((b for key, b in STYLE_BOTTOMS if key in s), None)
    if not top or not bottom:
        raise ValueError(f"style {raw!r}: name the top (e.g. FULL SLEEVE, HOOD) and the bottom (SHORTS or PANT)")
    return kind, top, bottom


def read_plan(data: bytes, fabric: str = "CottonMix") -> Tuple[List[dict], List[str]]:
    """Batch bodies (without engine settings) from an .xlsx file, and the rows that couldn't be used."""
    import openpyxl

    if len(data) > MAX_FILE:
        raise ValueError("The sheet is too big (2 MB at most)")
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception:
        raise ValueError("That isn't an Excel .xlsx file the app can read")
    groups: Dict[Tuple[str, str, str, str], Dict[Tuple[str, str], int]] = {}
    problems: List[str] = []
    rows_seen = 0
    for ws in wb.worksheets:
        cols = {}
        for r, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if not cols:
                cols = _header(row)
                continue
            if all(v in (None, "") for v in row) or row[cols["category"]] in (None, ""):
                continue  # blank lines and the TOTAL line
            rows_seen += 1
            if rows_seen > MAX_ROWS:
                raise ValueError(f"At most {MAX_ROWS} rows")
            where = f"{ws.title} row {r}"
            try:
                category = CATEGORY_NAMES.get(_norm(row[cols["category"]]))
                if not category:
                    raise ValueError(f"category {row[cols['category']]!r}: use BOYS, GIRLS, MEN or WOMEN")
                kind, top, bottom = style_parts(row[cols["style"]])
                t = colour_name(row[cols["top"]], fabric, category)
                b = colour_name(row[cols["bottom"]], fabric, category)
                kind = kind or ("coord" if t == b else "combo")
                if studio.is_white(b, fabric):  # bottoms aren't made in white
                    problems.append(f"{where}: {t} top with a {studio.WHITE_TOP_BOTTOM} bottom (bottoms aren't made in white)")
                    b = studio.WHITE_TOP_BOTTOM
                qty = int(float(row[cols["qty"]]))
                if not 1 <= qty <= 5000:
                    raise ValueError(f"quantity {qty}: 1 to 5000")
                if top not in studio.tops_for(category):
                    raise ValueError(f"{top} isn't made for {category}")
                pairs = groups.setdefault((category, top, bottom, kind), {})
                pairs[(t, b)] = max(pairs.get((t, b), 0), qty)
            except (ValueError, TypeError, IndexError) as exc:
                problems.append(f"{where}: {exc}")
        if not cols:
            problems.append(f"{ws.title}: no header row with CATEGORY, STYLE, TOP COLOR, BOTTOM COLOR and QTY")
    bodies = []
    for (category, top, bottom, kind), pairs in groups.items():
        models = max(pairs.values())
        bodies.append({"category": category, "top": top, "bottom": bottom, "fabric": fabric, "set_type": kind,
                       "colour_pairs": [[t, b, n] for (t, b), n in pairs.items()], "models": models,
                       "shot": "outfit", "poses": ["front_main", "back"]})
    return bodies, problems
