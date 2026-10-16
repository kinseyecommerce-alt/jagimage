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
