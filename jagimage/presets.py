"""Shot presets for common e-commerce image needs.

Each preset knows how to frame a product for a specific sales channel, and the
image size and background that channel expects.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Shot:
    key: str
    label: str
    template: str
    size: str = "1024x1024"
    background: str = "opaque"


SHOTS = {
    s.key: s
    for s in [
        Shot(
            "hero",
            "Marketplace hero (white background)",
            "Professional studio product photo of {product}, centered, filling about 85% "
            "of the frame, on a pure white (#FFFFFF) seamless background, soft even "
            "lighting, subtle natural contact shadow, sharp focus, true-to-life color. "
            "Meets Amazon/Shopify main-image guidelines.",
        ),
        Shot(
            "cutout",
            "Transparent cutout",
            "Isolated product photo of {product} with no background, crisp clean edges, "
            "studio lighting, true-to-life color, no shadow.",
            background="transparent",
        ),
        Shot(
            "lifestyle",
            "Lifestyle scene",
            "Photorealistic lifestyle photo of {product} being used in a natural, "
            "aspirational real-world setting that suits the product, shallow depth of "
            "field, warm natural light, the product is the clear focal point.",
            size="1536x1024",
        ),
        Shot(
            "flatlay",
            "Flat lay",
            "Top-down flat lay photo of {product} arranged with a few complementary "
            "props on a textured surface, balanced composition, soft diffused light, "
            "editorial magazine style.",
        ),
        Shot(
            "detail",
            "Close-up detail",
            "Macro close-up photo highlighting the materials, texture and craftsmanship "
            "of {product}, dramatic side lighting, very sharp detail.",
        ),
        Shot(
            "social",
            "Social / story ad (portrait)",
            "Scroll-stopping vertical social media ad image of {product}, bold "
            "composition, vibrant but on-brand colors, generous empty space at the top "
            "and bottom for captions, photorealistic.",
            size="1024x1536",
        ),
        Shot(
            "banner",
            "Website banner (landscape)",
            "Wide website hero banner featuring {product} on the right third, clean "
            "uncluttered background extending to the left for headline text, premium "
            "commercial photography look.",
            size="1536x1024",
        ),
    ]
}
