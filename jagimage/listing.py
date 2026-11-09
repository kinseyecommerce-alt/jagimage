"""Listing creator: marketplace listing files written from the printed SKUs (jagimage.printrun manifests) and the
shop's product facts, for JioMart, Shopify, Myntra, Amazon, Flipkart and Meesho.

How it ranks well: every marketplace's search ranks on relevance (the buyer's words in the title first, then in
bullets, description and keyword fields), on filters (complete, correct attributes put a product into the
filtered results most buyers use) and on conversion (clear titles, no misleading words). So the AI, looking at
the product photo, writes each marketplace's title in that marketplace's own pattern with the words Indian buyers
type, front-loaded; bullets and description that answer a buyer's questions; keywords that add what the title
doesn't already say; and fills every attribute. The software then holds the text to each marketplace's rules
(lengths, banned words and characters, repeated words, keyword field size) so a file isn't rejected.

The limits below come from the marketplaces' 2025-26 seller guides; marketplaces change them, so a template the
shop downloads from its seller portal can be uploaded and is filled in its own columns (the AI maps them)."""

import io
import json
import re
from typing import Callable, Dict, List, Optional

MARKETS: Dict[str, dict] = {
    "amazon": {"name": "Amazon", "title_max": 150, "bullets": 5, "bullet_max": 250, "desc_max": 2000, "keywords_bytes": 249,
               "title_pattern": "Brand + Gender + Fabric + Pattern/Print theme + Product type + key features (neck, sleeve, fit) + Colour",
               "notes": "No promotional words, no ALL CAPS, no special characters ! $ ? _ { } ^ ¬ ¦, a word at most twice in the title. "
                        "Search terms: words not already in the title, no brand names, no commas, singular or plural once."},
    "flipkart": {"name": "Flipkart", "title_max": 100, "bullets": 5, "bullet_max": 150, "desc_max": 1500, "keywords_max": 10,
                 "title_pattern": "Brand + Model name (Gender Pattern Fabric Product type)",
                 "notes": "Flipkart builds the shown title from attributes: the model name and attributes must be exact. "
                          "Search keywords: up to 10 phrases buyers type."},
    "myntra": {"name": "Myntra", "title_max": 100, "bullets": 6, "bullet_max": 120, "desc_max": 1000, "keywords_max": 8,
               "title_pattern": "Brand + Gender + Colour + Print theme + Pattern + Fabric + Article type (e.g. 'SPB Boys Navy Blue Dinosaur Printed Pure Cotton T-shirt')",
               "notes": "Product display name in Myntra's style, title case. Product details as short points. Attributes drive filters."},
    "meesho": {"name": "Meesho", "title_max": 100, "bullets": 5, "bullet_max": 150, "desc_max": 1000, "keywords_max": 10,
               "title_pattern": "Gender + Pattern + Fabric + Product type + Colour (brand optional)",
               "notes": "Simple words budget buyers search in English; no brand claims."},
    "jiomart": {"name": "JioMart", "title_max": 150, "bullets": 5, "bullet_max": 200, "desc_max": 2000, "keywords_max": 10,
                "title_pattern": "Brand + Gender + Pattern + Fabric + Product type + Neck + Sleeve + Colour",
                "notes": "Key features as bullets."},
    "shopify": {"name": "Shopify", "title_max": 70, "bullets": 5, "bullet_max": 200, "desc_max": 3000, "keywords_max": 15,
                "seo_title_max": 60, "seo_desc_max": 155,
                "title_pattern": "Brand Gender Print theme Printed Fabric Product type – Colour",
                "notes": "Google SEO: SEO title under 60 characters with the main keyword first, meta description under 155 "
                         "characters that makes a buyer click, URL handle in lowercase-dashes, tags for collections."},
}
MARKET_ORDER = ["amazon", "flipkart", "myntra", "meesho", "jiomart", "shopify"]
BANNED = ["best", "cheapest", "no.1", "number one", "#1", "free shipping", "free delivery", "sale", "discount", "offer",
          "guaranteed", "100% satisfaction", "hot deal", "limited time", "buy now", "top rated", "bestseller", "best seller",
          "lowest price", "original", "certified"]
BAD_CHARS = "!$?_{}^¬¦~*<>|#@"
STYLE_TEXT = {"HalfSleeveTee": ("T-shirt", "Half Sleeve", "Round Neck", "Regular Fit"),
              "FullSleeveTee": ("T-shirt", "Full Sleeve", "Round Neck", "Regular Fit"),
              "SleevelessTee": ("Vest T-shirt", "Sleeveless", "Round Neck", "Regular Fit"),
              "OversizedTee": ("Oversized T-shirt", "Half Sleeve", "Round Neck", "Oversized Fit"),
              "Sweatshirt": ("Sweatshirt", "Full Sleeve", "Round Neck", "Regular Fit"),
              "Hoodie": ("Hoodie", "Full Sleeve", "Hooded", "Regular Fit"),
              "ZipHoodie": ("Zip Hoodie", "Full Sleeve", "Hooded", "Regular Fit")}
GENDER_TEXT = {"Men": "Men", "Women": "Women", "Kids_Boys": "Boys", "Kids_Girls": "Girls"}
FABRIC_TEXT = {"PureCotton": "Pure Cotton", "CottonMix": "Cotton Blend"}

LISTING_SYSTEM = """You are India's best e-commerce listing writer for apparel, writing for JioMart, Shopify, Myntra,
Amazon, Flipkart and Meesho. You see the product photo and get its facts. Write listings that rank and convert:
- Use the exact words Indian buyers type in each marketplace's search (e.g. "printed t shirt for boys", "cotton
  tshirt men", "round neck half sleeve", "graphic tee"), most important words first. Name the print's theme.
- Follow each marketplace's title pattern given, within its length. Never invent facts: use only the facts given
  and what the photo plainly shows (colour, print, neck, sleeve). No promotional or claim words (best, sale, offer,
  free, guaranteed, original, No.1), no ALL CAPS words, no emojis, no special characters, no competitor brands.
- Bullets: each starts with a 2-4 word lead in title case then a colon, covering print/design, fabric and feel,
  fit and sizes, occasion and styling, care. Descriptions: 3 short paragraphs, natural keywords, no keyword lists.
- Keywords: phrases a buyer types that the title does not already contain: synonyms, spellings (tshirt, tee,
  t-shirt), occasions, themes, age or gender words. No brand names, no repetition.
- Attributes: fill every one from the facts and photo with the standard value a marketplace filter uses.
Answer in JSON as the schema says."""

LISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "attributes": {"type": "object", "properties": {k: {"type": "string"} for k in (
            "colour", "colour_family", "pattern", "print_type", "print_theme", "neck", "sleeve", "fit", "occasion",
            "length", "closure", "season", "ideal_for", "wash_care", "product_type")}},
        "markets": {"type": "object", "properties": {m: {"type": "object", "properties": {
            "title": {"type": "string"}, "bullets": {"type": "array", "items": {"type": "string"}},
            "description": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}},
            **({"seo_title": {"type": "string"}, "seo_description": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}
               if m == "shopify" else {})}} for m in MARKET_ORDER}},
    },
    "required": ["attributes", "markets"],
}


def _clean_text(s: str, keep_newlines: bool = False) -> str:
    s = str(s or "")
    s = "".join(ch for ch in s if ch not in BAD_CHARS and (ord(ch) < 0x2190 or ch in "–—’‘“”•"))  # no emojis or symbols
    s = re.sub(r"[ \t]+", " ", s)
    if not keep_newlines:
        s = " ".join(s.split())
    return s.strip()


def _unclaim(s: str) -> str:
    """Promotional and claim words out (marketplaces suppress or reject listings with them)."""
    for w in sorted(BANNED, key=len, reverse=True):
        s = re.sub(r"(?i)(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", "", s)
    return re.sub(r"\s{2,}", " ", s).strip(" ,-|")


def _no_caps(s: str) -> str:
    return re.sub(r"\b([A-Z]{4,})\b", lambda m: m.group(1).title(), s)


def _cut(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut.rstrip(" ,-|:;")


def _limit_repeats(title: str, most: int = 2) -> str:
    seen: Dict[str, int] = {}
    out = []
    for w in title.split():
        k = re.sub(r"[^a-z0-9]", "", w.lower())
        seen[k] = seen.get(k, 0) + 1
        if k and seen[k] > most and len(k) > 2:
            continue
        out.append(w)
    return " ".join(out)


def amazon_search_terms(keywords: List[str], title: str, brand: str, limit: int = 249) -> str:
    """Amazon's backend search terms: words the title doesn't have, no brand, each once, space-separated, under
    the byte limit."""
    have = {re.sub(r"[^a-z0-9]", "", w.lower()) for w in title.split()} | {brand.lower()}
    words, seen = [], set()
    for k in keywords:
        for w in str(k).lower().replace(",", " ").split():
            w = re.sub(r"[^a-z0-9-]", "", w)
            if not w or w in have or w in seen or w in BANNED:
                continue
            if len((" ".join(words + [w])).encode()) > limit:
                return " ".join(words)
            words.append(w)
            seen.add(w)
    return " ".join(words)


def tidy(market: str, content: dict, brand: str) -> dict:
    """The AI's text for one marketplace held to that marketplace's rules."""
    rule = MARKETS[market]
    title = _limit_repeats(_no_caps(_unclaim(_clean_text(content.get("title")))))
    if brand and not title.lower().startswith(brand.lower()) and market in ("amazon", "myntra", "jiomart", "flipkart", "shopify"):
        title = f"{brand} {title}"
    title = _cut(title, rule["title_max"])
    bullets = [_cut(_no_caps(_unclaim(_clean_text(b))), rule["bullet_max"]) for b in content.get("bullets") or [] if str(b).strip()]
    bullets = [b for b in bullets if b][:rule["bullets"]]
    desc = _cut(_no_caps(_unclaim(_clean_text(content.get("description"), keep_newlines=True))), rule["desc_max"])
    kws = []
    for k in content.get("keywords") or []:
        k = _unclaim(_clean_text(k)).lower()
        if k and k not in kws and brand.lower() not in k:
            kws.append(k)
    out = {"title": title, "bullets": bullets, "description": desc}
    if market == "amazon":
        out["search_terms"] = amazon_search_terms(kws, title, brand, rule["keywords_bytes"])
    else:
        out["keywords"] = kws[:rule.get("keywords_max", 10)]
    if market == "shopify":
        out["seo_title"] = _cut(_unclaim(_clean_text(content.get("seo_title") or title)), rule["seo_title_max"])
        out["seo_description"] = _cut(_unclaim(_clean_text(content.get("seo_description") or desc)), rule["seo_desc_max"])
        tags = []
        for t in (content.get("tags") or []) + kws:
            t = _clean_text(t).lower()
            if t and t not in tags:
                tags.append(t)
        out["tags"] = tags[:20]
    return out


def base_attributes(sku: dict, facts: dict) -> dict:
    kind, sleeve, neck, fit = STYLE_TEXT.get(sku.get("style", ""), ("T-shirt", "Half Sleeve", "Round Neck", "Regular Fit"))
    return {"brand": facts.get("brand") or "SPB", "gender": GENDER_TEXT.get(sku.get("category", ""), "Unisex"),
            "fabric": facts.get("fabric") or FABRIC_TEXT.get(sku.get("fabric", ""), "Cotton"),
            "product_type": kind, "sleeve": sleeve, "neck": neck, "fit": fit, "colour": sku.get("colour", ""),
            "pattern": "Printed", "print_theme": sku.get("theme", "")}


def write(ask: Callable, sku: dict, facts: dict, photo: Optional[tuple], markets: List[str]) -> dict:
    """One SKU's listings for the given marketplaces: {attributes, markets: {m: tidy content}}."""
    attrs = base_attributes(sku, facts)
    lines = [f"SKU: {sku['sku']}", f"Facts: {json.dumps({**attrs, **{k: v for k, v in facts.items() if v}}, ensure_ascii=False)}",
             f"Print: {sku.get('print', '')}; theme {sku.get('theme', '')}; shows: {sku.get('print_description', '')}"
             + (f"; text on it: \"{sku['print_text']}\"" if sku.get("print_text") else ""),
             "Write for these marketplaces:"]
    for m in markets:
        r = MARKETS[m]
        lines.append(f"- {m}: title pattern '{r['title_pattern']}', title at most {r['title_max']} characters, "
                     f"{r['bullets']} bullets of at most {r['bullet_max']} characters, description at most {r['desc_max']} characters. {r['notes']}")
    ans = ask("\n".join(lines), LISTING_SYSTEM, LISTING_SCHEMA, [photo] if photo else [])
    ai_attrs = {k: _clean_text(v)[:60] for k, v in (ans.get("attributes") or {}).items() if str(v or "").strip()}
    got = ans.get("markets") or {}
    brand = attrs["brand"]
    return {"attributes": {**ai_attrs, **{k: v for k, v in attrs.items() if v}},
            "markets": {m: tidy(m, got.get(m) or {}, brand) for m in markets}}


def sizes_of(facts: dict) -> List[str]:
    raw = facts.get("sizes") or ""
    sizes = [s.strip() for s in (raw if isinstance(raw, list) else str(raw).split(",")) if s.strip()]
    return sizes or ["Free Size"]


def _handle(title: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")[:80]


def rows_for(market: str, sku: dict, listing: dict, facts: dict) -> List[dict]:
    """The marketplace's sheet rows for one SKU: one per size (a variant), images on every row."""
    c, a = listing["markets"][market], listing["attributes"]
    urls = [im.get("url", "") for im in sku.get("images") or [] if im.get("url")]
    rows = []
    for i, size in enumerate(sizes_of(facts)):
        r = {"Seller SKU": f"{sku['sku']}-{re.sub(r'[^A-Za-z0-9]', '', size).upper()}", "Parent SKU": sku["sku"],
             "Title": c["title"], "Brand": a.get("brand", ""), "Gender": a.get("gender", ""), "Product Type": a.get("product_type", ""),
             "Colour": a.get("colour", ""), "Colour Family": a.get("colour_family", ""), "Size": size,
             "Fabric": a.get("fabric", ""), "Pattern": a.get("pattern", ""), "Print Theme": a.get("print_theme", ""),
             "Neck": a.get("neck", ""), "Sleeve": a.get("sleeve", ""), "Fit": a.get("fit", ""), "Occasion": a.get("occasion", ""),
             "Wash Care": a.get("wash_care", "") or facts.get("wash_care", ""), "Ideal For": a.get("ideal_for", "") or a.get("gender", ""),
             "MRP": facts.get("mrp", ""), "Selling Price": facts.get("price", ""), "HSN": facts.get("hsn", "6109"),
             "GST %": facts.get("gst", "5"), "Country of Origin": facts.get("origin", "India"),
             "Manufacturer": facts.get("manufacturer", ""), "Packer": facts.get("packer", "") or facts.get("manufacturer", ""),
             "Net Quantity": facts.get("net_quantity", "1 N"), "Weight (g)": facts.get("weight", ""), "GSM": facts.get("gsm", ""),
             "Description": c["description"]}
        for j in range(MARKETS[market]["bullets"]):
            r[f"Bullet {j + 1}"] = c["bullets"][j] if j < len(c["bullets"]) else ""
        if market == "amazon":
            r["Search Terms"] = c.get("search_terms", "")
        else:
            r["Keywords"] = ", ".join(c.get("keywords") or [])
        for j in range(8):
            r[f"Image {j + 1}"] = urls[j] if j < len(urls) else ""
        rows.append(r)
    return rows


SHOPIFY_COLUMNS = ["Handle", "Title", "Body (HTML)", "Vendor", "Product Category", "Type", "Tags", "Published",
                   "Option1 Name", "Option1 Value", "Option2 Name", "Option2 Value", "Variant SKU", "Variant Grams",
                   "Variant Inventory Tracker", "Variant Inventory Qty", "Variant Inventory Policy", "Variant Fulfillment Service",
                   "Variant Price", "Variant Compare At Price", "Variant Requires Shipping", "Variant Taxable", "Image Src",
                   "Image Position", "Image Alt Text", "SEO Title", "SEO Description", "Status"]


def shopify_rows(sku: dict, listing: dict, facts: dict) -> List[dict]:
    """Shopify's product import CSV: the first row has the product, one row per size, then extra image rows."""
    c, a = listing["markets"]["shopify"], listing["attributes"]
    handle = _handle(f"{c['title']} {sku['sku']}")
    body = "".join(f"<p>{p.strip()}</p>" for p in c["description"].split("\n") if p.strip())
    if c["bullets"]:
        body += "<ul>" + "".join(f"<li>{b}</li>" for b in c["bullets"]) + "</ul>"
    urls = [im.get("url", "") for im in sku.get("images") or [] if im.get("url")]
    rows = []
    for i, size in enumerate(sizes_of(facts)):
        r = {k: "" for k in SHOPIFY_COLUMNS}
        r.update({"Handle": handle, "Option1 Name": "Size", "Option1 Value": size, "Option2 Name": "Colour" if i == 0 else "",
                  "Option2 Value": a.get("colour", ""), "Variant SKU": f"{sku['sku']}-{re.sub(r'[^A-Za-z0-9]', '', size).upper()}",
                  "Variant Grams": facts.get("weight", ""), "Variant Inventory Tracker": "shopify",
                  "Variant Inventory Qty": facts.get("stock", "0"), "Variant Inventory Policy": "deny",
                  "Variant Fulfillment Service": "manual", "Variant Price": facts.get("price", ""),
                  "Variant Compare At Price": facts.get("mrp", ""), "Variant Requires Shipping": "TRUE", "Variant Taxable": "TRUE"})
        if i == 0:
            r.update({"Title": c["title"], "Body (HTML)": body, "Vendor": a.get("brand", ""),
                      "Product Category": "Apparel & Accessories > Clothing > Shirts & Tops", "Type": a.get("product_type", ""),
                      "Tags": ", ".join(c.get("tags") or []), "Published": "TRUE", "SEO Title": c.get("seo_title", ""),
                      "SEO Description": c.get("seo_description", ""), "Status": "active"})
        if i < len(urls):
            r.update({"Image Src": urls[i], "Image Position": str(i + 1), "Image Alt Text": f"{c['title']} - view {i + 1}"})
        rows.append(r)
    for j in range(len(rows), len(urls)):
        r = {k: "" for k in SHOPIFY_COLUMNS}
        r.update({"Handle": handle, "Image Src": urls[j], "Image Position": str(j + 1), "Image Alt Text": f"{c['title']} - view {j + 1}"})
        rows.append(r)
    return rows


def workbook(market: str, rows: List[dict]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = MARKETS[market]["name"][:30]
    cols = SHOPIFY_COLUMNS if market == "shopify" else (list(rows[0].keys()) if rows else [])
    ws.append(cols)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in rows:
        ws.append([r.get(k, "") for k in cols])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def shopify_csv(rows: List[dict]) -> bytes:
    import csv

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SHOPIFY_COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


# ---- the marketplace's own template, filled in its own columns
MAP_SYSTEM = """You fill a marketplace's bulk listing template (Excel) for a clothing seller. You get the template's
first rows and the list of field names the seller's software has. Say which row holds the column names the data
must follow (header_row, 1-based), on which row the first product goes (first_data_row), and for each template
column (by its letter) which software field fills it, or a fixed value for a column every product shares (e.g.
"Country of Origin": "India"), or nothing. Use only the given field names. Map image columns in order (main image =
Image 1). Leave a column empty rather than guess."""

MAP_SCHEMA = {
    "type": "object",
    "properties": {"header_row": {"type": "integer"}, "first_data_row": {"type": "integer"},
                   "columns": {"type": "array", "items": {"type": "object", "properties": {
                       "column": {"type": "string"}, "field": {"type": "string"}, "fixed": {"type": "string"}}}}},
    "required": ["header_row", "first_data_row", "columns"],
}


def template_preview(data: bytes, sheet: Optional[str] = None, rows: int = 8) -> tuple:
    """(workbook, sheet, text of its first rows by column letter) of an uploaded template."""
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    wb = load_workbook(io.BytesIO(data))
    ws = wb[sheet] if sheet and sheet in wb.sheetnames else max(wb.worksheets, key=lambda w: w.max_column * min(w.max_row, 10))
    lines = []
    for r in range(1, min(rows, ws.max_row) + 1):
        cells = []
        for c in range(1, min(ws.max_column, 250) + 1):
            v = ws.cell(r, c).value
            if v not in (None, ""):
                cells.append(f"{get_column_letter(c)}={str(v)[:60]}")
        lines.append(f"Row {r}: " + "; ".join(cells))
    return wb, ws, "\n".join(lines)


def fill_template(ask: Callable, data: bytes, market: str, rows: List[dict]) -> bytes:
    """The uploaded template with the rows written into its own columns, below its header."""
    from openpyxl.utils import column_index_from_string

    if not rows:
        raise ValueError("No listings to put in the template")
    wb, ws, preview = template_preview(data)
    fields = list(rows[0].keys())
    ans = ask(f"Marketplace: {MARKETS[market]['name']}\nTemplate sheet '{ws.title}':\n{preview}\n\nSoftware fields: {json.dumps(fields)}",
              MAP_SYSTEM, MAP_SCHEMA)
    start = max(2, int(ans.get("first_data_row") or 2))
    cols = []
    for m in ans.get("columns") or []:
        try:
            idx = column_index_from_string(str(m.get("column", "")).strip().upper())
        except ValueError:
            continue
        f, fixed = m.get("field") or "", m.get("fixed") or ""
        if f in fields or fixed:
            cols.append((idx, f if f in fields else "", str(fixed)[:200]))
    if not cols:
        raise ValueError("The template's columns couldn't be matched: check it is the marketplace's listing template")
    for i, r in enumerate(rows):
        for idx, f, fixed in cols:
            ws.cell(start + i, idx).value = r.get(f, "") if f else fixed
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class ListingJob:
    """Listings for a set of printed SKUs, in a background thread: one AI call per SKU, then one file per
    marketplace (the shop's own template filled where one was uploaded)."""

    def __init__(self, ask: Callable, data, skus: List[dict], markets: List[str], facts: dict,
                 templates: Optional[Dict[str, bytes]] = None):
        import threading
        import time
        import uuid
        from pathlib import Path

        markets = [m for m in MARKET_ORDER if m in (markets or [])]
        if not markets:
            raise ValueError("Pick at least one marketplace")
        if not skus:
            raise ValueError("Pick the SKUs to list")
        if len(skus) > 2000:
            raise ValueError("At most 2000 SKUs in one go")
        self.id = uuid.uuid4().hex[:8]
        self.ask, self.skus, self.markets = ask, skus, markets
        self.facts = {k: " ".join(str(v).split())[:300] for k, v in (facts or {}).items() if v not in (None, "")}
        self.templates = templates or {}
        self.dir = Path(data) / "listings" / self.id
        self.done, self.errors, self.files, self.error = 0, [], {}, None
        self.preview: List[dict] = []
        self.started, self.finished = time.time(), None
        self._time = time
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"listing-{self.id}")

    def status(self) -> dict:
        return {"id": self.id, "markets": self.markets, "total": len(self.skus), "done": self.done,
                "errors": self.errors[-20:], "error": self.error, "running": self.finished is None,
                "files": {m: f"/api/listings/{self.id}/file/{m}" for m in self.files}, "preview": self.preview[:3]}

    def _photo(self, sku: dict):
        f = sku.get("local")
        if f:
            from pathlib import Path

            p = Path(f)
            if p.exists():
                from PIL import Image

                im = Image.open(p).convert("RGB")
                im.thumbnail((1024, 1024))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
                return buf.getvalue(), "image/jpeg"
        return None

    def _run(self):
        try:
            rows: Dict[str, List[dict]] = {m: [] for m in self.markets}
            listings = []
            for sku in self.skus:
                try:
                    lst = write(self.ask, sku, self.facts, self._photo(sku), self.markets)
                except Exception as exc:
                    self.errors.append(f"{sku.get('sku')}: {type(exc).__name__}: {exc}"[:300])
                    continue
                listings.append({"sku": sku["sku"], **lst})
                if len(self.preview) < 3:
                    self.preview.append({"sku": sku["sku"], **lst})
                for m in self.markets:
                    rows[m] += shopify_rows(sku, lst, self.facts) if m == "shopify" else rows_for(m, sku, lst, self.facts)
                self.done += 1
            if not listings:
                raise RuntimeError("No listing could be written" + (f": {self.errors[0]}" if self.errors else ""))
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "listings.json").write_text(json.dumps(listings, indent=1, ensure_ascii=False))
            for m in self.markets:
                if m in self.templates:
                    try:
                        data, name = fill_template(self.ask, self.templates[m], m, rows[m]), f"{m}_template_filled.xlsx"
                    except Exception as exc:
                        self.errors.append(f"{MARKETS[m]['name']} template: {exc}"[:300])
                        data, name = workbook(m, rows[m]), f"{m}_listing.xlsx"
                elif m == "shopify":
                    data, name = shopify_csv(rows[m]), "shopify_products.csv"
                else:
                    data, name = workbook(m, rows[m]), f"{m}_listing.xlsx"
                (self.dir / name).write_bytes(data)
                self.files[m] = name
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            self.finished = self._time.time()

    def start(self):
        self.thread.start()
        return self
