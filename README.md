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

## Studio app: hundreds of models, any top and bottom colours

`jagimage serve` opens the Studio on your computer (http://127.0.0.1:8765). It makes catalogue photos of real-looking models in batch:

```bash
pip install -e ".[gemini]"
jagimage serve            # then open http://127.0.0.1:8765
```

1. **Category:** Men, Women, Kids Boys or Kids Girls.
2. **Outfit:**
   - Top: half sleeve, full sleeve or sleeveless tee, hoodie, or sweatshirt.
   - Bottom: shorts, track pants, or jeans (jeans stay as the AI makes them).
   - Fabric: Pure Cotton or Cotton Mix.
3. **Shot:**
   - **Top and bottom:** full body, head to shoes, and both garments are recoloured.
   - **Top only:** framed from the head to just above the knee. Only the top is recoloured, and the bottom is made by the AI in one colour you pick. That's 1–2 AI images per model.
4. **Top colours and bottom colours:** pick from either palette, or add any `#RRGGBB`. Choose "Same colour as the top" for matching sets. Or pick bottom colours, and every top colour is made with every bottom colour.
5. **Models:** e.g. 300. Every model is a different real-looking person, with a different face shape, skin tone, hair, eyes and smile.
   - **Faces never repeat in the app:** each model number has its own combination of features (over 11,000 per category), and neighbouring numbers differ in hair, skin and face shape.
   - **Ledger:** the Studio keeps a list of used numbers in `output/studio/_models.json`. "Start at model no." is automatic: a new batch begins after the last number used in that category.
   - **Overlap check:** typing a range another batch already used is refused, so you can't make the same faces twice. Re-running the same batch, with "Load settings", continues it.
6. **Engine:** Gemini (1K/2K/4K) or OpenAI. Paste the API key in the app, or set `GEMINI_API_KEY` / `OPENAI_API_KEY`. The key stays in the app's memory on your computer.

**How it works:**
- **Base photos:** the AI shoots each model with the top in a green base colour and the bottom in a purple one. Each model needs at most 4 base photos (light/dark top × light/dark bottom), and only the ones your colours use.
- **Same person in every photo:** the model's first photo is the face reference for their other base photos.
- **Colour check:** a base photo where the AI got the colours wrong is retried.
- **Recolouring:** every top/bottom pair is recoloured to the exact hex on your computer, at no API cost. You also get a 1500×1500 marketplace JPG and a `qc.json` with the ΔE of the top and the bottom.

**Hands free** (on by default): the prompt asks for hands that never touch the clothes, nothing in the pockets or hands, and arms a little away from the body, so the whole garment shows. It applies to every pose except the hand-in-pocket pose.

**Your own poses and a test photo first:**
- Under **Write a pose**, pick "+ New pose…". Give it a name and describe what the model does, e.g. "sitting cross-legged on the studio floor, one hand waving at the camera". You can also change the wording of any built-in pose, and reset it later.
- **Extra instructions for every photo** adds a line to every prompt, e.g. "big laughing expression".
- **Preview prompt** shows the exact text sent to Gemini or OpenAI.
- **Generate test photo** makes one model in that pose and your first colour pair, then recolours it. That's 1 AI image for front main, or 2 for any other pose, so you can check a pose before running the batch.

**Save to Google Drive:**
- Install [Google Drive for desktop](https://www.google.com/drive/download/). The Studio finds its "My Drive" folder, and you can pick it under **Save to → Also save finished photos to**. You can also choose "Another folder…" and type any path.
- Finished photos are copied there as `<batch>/<Top>__<Bottom>/m0001_front_main.jpg`, and Drive uploads them automatically.
- You choose the files: 1500×1500 JPG, full-size PNG, or both.
- Base photos, masks and everything else stay in the working folder, which you can change under **Save to → Working folder**.

The Run box shows the count and the **estimated AI cost** before you start, in $ and ₹: AI images × list price (Gemini 3 Pro 2K about $0.137 each including the prompt and reference photos), plus 10% for retries, and the cost per finished photo. Prices live in `jagimage/pricing.py` (checked 2026-10-21); set `JAGIMAGE_USD_INR` to change the rupee rate.

**Batch mode** (Gemini, Engine section): half price. The app sends the base photos as Google batch jobs in up to 3 rounds (each model's first front; then the other fronts and the first base's poses; then the remaining poses, because each round uses photos from the one before as references). Google promises each job within 24 hours and is usually much faster. The job IDs are saved in the batch folder (`_batch_jobs.json`), so if the app is closed while Google is working, the next start collects that job instead of paying for it again. Photos the AI gets wrong are sent again in the next round.

**Same models in every style:** pick extra styles under "Same models in more styles", e.g. Men's half sleeve, full sleeve and sleeveless tees, hoodies and sweatshirts.
- **Order:** the styles run one after another with the same model numbers.
- **Same face:** each model's first front photo is saved in `output/studio/_faces/<Category>/m0001.png` and used as the face reference in every other style.
- **AI cost:** each style needs its own AI base photos, because a hoodie isn't a tee, so the cost adds up per style. Colours are still recoloured for free.
- **Adding a style later:** tick "Reuse models made earlier in this category" and type the first model number.

**Large daily volumes** (e.g. 1,000 new models a day):
- **Faces:** there are over 550,000 different faces per category.
- **Batch job size:** batch jobs are split into chunks of 400 photos (`JAGIMAGE_BATCH_CHUNK`) to fit a new Gemini account's batch quota. Raise the chunk size once the project reaches Tier 2.
- **Disk space:** turn off "Also keep a full-size PNG" to save about 4 MB per photo. The Run box shows the disk space a batch needs.

The Run box shows the count before you start. For example, 300 models × 9 colours (matching sets) needs 600 AI images and gives 2,700 photos.

**Other features:**
- **Resuming:** stopping and starting again skips everything already made.
- **Recolour only:** add colours later without new AI calls.
- **Downloads:** the gallery downloads all photos as a ZIP of JPGs or full-size PNGs.
- **Output layout:**
  - `output/studio/<batch>/m0001/<Top>__<Bottom>/` for each outfit.
  - `output/studio/<batch>/m0001/_base/` for the AI base photos and their editable masks.

## Recolor web app

`web/recolor/index.html` is a browser tool for recolouring existing garment photos. Open the file in a browser; nothing is uploaded anywhere. `app.html` is the same page without the HTML wrapper (the published version).

- **Pick the garment:** it detects colour regions; hover to see them, click to select. "Pick only the part I click" keeps black hair separate from a black tee. "Include similar shades" also pulls in folds and shadows. A coloured garment only grows into its own hue, so a pale shirt never takes the grey studio or skin with it.
- **Clean edges and creases:** the garment's edge is unmixed from the photo, so it stays as smooth as the original, and deep creases the camera crushed to pure black are filled with soft shading instead of turning into blotches.
- **Exact colour:** set a hex code or click a Pure Cotton / Cotton Mix swatch. Each region shows ΔE to the target (under 3 = on shade).
- **Three modes:**
  - **Exact match:** lands on the hex on average.
  - **Keep lightness:** keeps the photo's own light and shadow.
  - **Scale shading:** best for big jumps like black to white.
- **Colorways:** one photo in every palette colour, saved as a ZIP.
- **Batch and history:** apply one photo's edits to all photos, export as a ZIP, undo/redo, and a shade ramp you can export as CSS or JSON.
