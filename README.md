# jagimage

An image agent for an e-commerce store, built on OpenAI's **`gpt-image-2.5-flare`** model. Describe a product, or hand it a real product photo, and it makes images for each sales channel: marketplace hero shots, transparent cutouts, lifestyle scenes, flat lays, detail close-ups, social ads and website banners. Your brand style goes into every prompt.

## Setup

```bash
pip install -e .
export OPENAI_API_KEY=sk-...        # see .env.example
```

## Usage

```bash
# List the shot types
jagimage shots

# Generate from a text description
jagimage create "matte black insulated steel water bottle, 750ml" --shots hero,lifestyle,social

# Start from your real product photo (keeps the product exact and restyles the scene)
jagimage create "cognac leather weekender bag" --photo bag.jpg --shots lifestyle,banner -n 3

# Apply a brand profile to every image
jagimage create "ceramic pour-over set" --brand examples/brand.json --shots flatlay

# Run a whole catalog from a CSV (name, description, shots, photo)
jagimage batch examples/products.csv --brand examples/brand.json
```

Images are saved to `output/<product-slug>/<shot>-<n>.png`. Existing files are never overwritten.

### Shot types

| Key | Size | Use |
|---|---|---|
| `hero` | 1024x1024 | Main image for Amazon/Shopify, on a pure white background |
| `cutout` | 1024x1024 | Transparent PNG for compositing |
| `lifestyle` | 1536x1024 | Product in use in a real-world setting |
| `flatlay` | 1024x1024 | Top-down styled arrangement |
| `detail` | 1024x1024 | Close-up of materials and craftsmanship |
| `social` | 1024x1536 | Vertical story or ad with room for captions |
| `banner` | 1536x1024 | Website hero with empty space for a headline |

Add or change shots in `jagimage/presets.py`.

### Options

`--quality low|medium|high|xhigh|max|auto` (default `high`), `--format png|jpeg|webp`, `--size`, `--extra "art direction"`, `--out DIR`, `--model` (or set `JAGIMAGE_MODEL`).

## Kids' tee catalogue

`jagimage catalog` makes a full Myntra-style catalogue for each style and colour: Boys Half-, Full- and Sleeveless tees in 7 colours, and Girls Half- and Full-sleeve tees in 9 colours. That's 39 SKUs. Each SKU gets a front shot plus 4 poses, 195 images in total.

```bash
# Look at the exact prompts first (no API calls, no cost)
jagimage catalog --kids examples/kids.json --gender Boys --colour Navy --dry-run

# Try one SKU, then run everything
jagimage catalog --kids examples/kids.json --gender Girls --style HalfSleeveTee --colour Pink
jagimage catalog --kids examples/kids.json
```

- **Same child in every shot:** the front shot (`01-front.png`) is made first. Each pose (`02-threequarter`, `03-side`, `04-back`, `05-handpocket`) is then made from that front shot, so the child, garment and studio stay the same.
- **Face reference:** set `face_ref` in `examples/kids.json` to a portrait photo (the path is relative to the JSON file). It is used only for the child's face, on every shot.
- **Resumable:** images that already exist are skipped, so you can re-run after an interruption. Use `--force` to regenerate them.
- **Output:** `output/<Gender>/<Style>/<Colour>/<shot>.png`, size 1024x1536.

### Fabric texture

- **Fabric close-up:** `--texture` adds a `06-fabric` shot to each SKU. It's an extreme macro of the chest and ribbed collar, with raking light that shows the knit loops and stitching, and no child in frame. It is made from that SKU's front shot, so it's the same garment.
- **Real fabric photos:** `--swatches DIR` uses your own fabric photos as the reference for knit texture and exact colour on every shot. Name them `<Colour>.jpg` (e.g. `Navy.jpg`), or `<Style>-<Colour>.jpg` for a different fabric per style (that name wins). `.jpeg`, `.png` and `.webp` also work. Colours with no photo fall back to the text description, and the run tells you which ones those are. A flat, evenly lit, colour-accurate close-up of the actual fabric works best.
- **Fabric wording:** the knit description used in every prompt is `FABRIC` in `jagimage/catalog.py`. Change it there, e.g. to give a GSM or say combed or bio-washed cotton.

```bash
jagimage catalog --kids examples/kids.json --swatches swatches/ --texture
```
- Other options: `--poses 02-threequarter,04-back`, `--quality`, `--size`, `--fidelity high` (sends `input_fidelity` on reference edits, if your model supports it).

Colours, poses, categories and prompts live in `jagimage/catalog.py`. By default, full-sleeve tees are styled with joggers (prompts swap "shorts" for "joggers") and everything else with shorts. Bottoms are navy, or charcoal under navy and black tops. Change this in `pants()` and `legs()`.

## SPB Textile prompt library

`jagimage library` makes images from the full SPB Textile prompt library. It covers **Men, Women, Kids Boys and Kids Girls**, each in **7 products** (HalfSleeveTee, FullSleeveTee, SleevelessTee, Shorts, Trackpants, Hoodie, Sweatshirt). Each product comes in **Pure Cotton** (9 colours) or **Cotton Mix** (17 colours), and each variant gets **6 poses** (front_main, front_main_white, threequarter, side, back, handpocket). The prompts are built from the library's parts in `jagimage/library.py`, and tests check them word for word against the library document.

Identity is kept consistent in two stages, as the library describes:

```bash
# 1. Make fronts only (front_main + front_main_white), using each category's face reference
jagimage library --models examples/models.json --category Men --product HalfSleeveTee --fabric PureCotton --stage fronts
# 2. Review them, delete any you reject, and re-run step 1 to remake them
# 3. Make the other poses: image 1 = face reference, image 2 = the approved front_main
jagimage library --models examples/models.json --category Men --product HalfSleeveTee --fabric PureCotton --stage poses
```

- **A different model for each variant:** each product/fabric/colour gets its own model from the category's list in `examples/models.json`. The list rotates, and the model stays the same across that variant's 6 poses. Each entry can have:
  - a `face` photo,
  - a `hairstyle` (put the exact hairstyle for every Kids_Girls entry),
  - a `look` (extra face details).

  An entry without a photo creates a brand-new face. Run with no `--models` file at all and every variant gets a new face. Use `--face-by product` to keep one model across all colours of a product. Whichever model was used for a front is saved in `model.json` next to the images, so the poses always match it.
- Filters take comma-separated values: `--category`, `--product`, `--fabric`, `--colour "Navy Blue,Light Pink"`, `--poses`.
- `--dry-run` prints the prompts without calling the API. `--export prompts.json` writes the whole library (category → fabric → pose → prompt/negative).
- The OpenAI image API has no negative-prompt field, so each pose's negative list is added to the prompt as an `Avoid: …` line. `--no-negative` turns that off.
- Images that already exist are skipped. Output goes to `output/<Category>/<Product>/<Fabric>/<Colour>/<pose>.png`.
- The full library is 728 variants × 6 poses = 4,368 images. Use filters to run it in batches.

## Use from Python

```python
from jagimage import ImageAgent, Brand

agent = ImageAgent(brand=Brand.load("examples/brand.json"))
paths = agent.create("linen throw pillow, sage green", shot="lifestyle", n=2)
```

## Tests

```bash
pip install -e ".[dev]" && pytest
```

### Gemini, and two base colours for the whole palette

`--provider gemini` uses Google's Gemini image model (`gemini-3-pro-image` by default) at `--gemini-size 2K` (HD; `4K` also works), 3:4 portrait. Set `GEMINI_API_KEY` and install with `pip install -e ".[gemini]"`.

`--bases` shoots each style in just **two base colours** with AI, then makes every palette colour from those photos with exact-hex recolouring:

```bash
# 1. AI: fronts of both bases (light = mint green, dark = bottle green), review them
jagimage library --bases --provider gemini --category Kids_Boys --product HalfSleeveTee --fabric PureCotton --stage fronts
# 2. AI: the other poses from the approved fronts, then recolour all 9 colours
jagimage library --bases --provider gemini --category Kids_Boys --product HalfSleeveTee --fabric PureCotton
```

- Light colours (OKLab lightness 0.55 and up, e.g. White, Yellow, Lavender) come from the light base; darker ones (Black, Navy, Red, Maroon) from the dark base.
- The bases are greens because no skin, hair, grey studio, denim or black trousers is green, so the garment can be found reliably. Change them with `--base-light` / `--base-dark`.
- Every recoloured photo gets a `qc.json` entry with its ΔE to the target hex (under 3 is on shade). The garment mask is saved as `_base_*/<pose>_mask.png`; edit it (white = garment) and re-run with `--stage recolor --force` to fix a bad edge.
- Pure black and pure white are shown the way they photograph: black at about `#1A1A1A`, white at about `#F2F2F2` with highlights up to `#FFFFFF`. At the exact hex there's no room left for the knit, rib and stitches, so the shirt would be a flat shape. Their `qc.json` notes this, and the ΔE is measured against that shade.
- Edges are unmixed from the photo itself, so they stay as smooth as the AI drew them. Backdrop showing between the legs or between arm and body is never recoloured.
- Base photos live in `_base_light/` and `_base_dark/` next to the colour folders.
- Full library: 336 styles×poses ×2 = 672 AI images instead of 4,368.

## Recolor web app

`web/recolor/index.html` is a browser tool for recolouring existing garment photos. Open the file in a browser; nothing is uploaded anywhere. `app.html` is the same page without the HTML wrapper (the published version).

- **Pick the garment:** it detects colour regions; hover to see them, click to select. "Pick only the part I click" keeps black hair separate from a black tee. "Include similar shades" also pulls in folds and shadows.
- **Exact colour:** set a hex code or click a Pure Cotton / Cotton Mix swatch. Each region shows ΔE to the target (under 3 = on shade).
- **Three modes:**
  - **Exact match:** lands on the hex on average.
  - **Keep lightness:** keeps the photo's own light and shadow.
  - **Scale shading:** best for big jumps like black to white.
- **Colorways:** one photo in every palette colour, saved as a ZIP.
- **Batch and history:** apply one photo's edits to all photos, export as a ZIP, undo/redo, and a shade ramp you can export as CSS or JSON.
