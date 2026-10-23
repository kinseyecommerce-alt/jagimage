import json

from jagimage import pricing, studio
from jagimage.providers import GeminiProvider, usage_tokens

# usageMetadata of a real Gemini 3.1 Flash 2K batch response (half sleeve tee, back view)
REAL = {"promptTokenCount": 1136, "candidatesTokenCount": 2061, "totalTokenCount": 3197,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 878}, {"modality": "IMAGE", "tokenCount": 258}],
        "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1680}]}


def test_googles_counts_give_the_real_price_of_one_image(tmp_path):
    assert usage_tokens(REAL) == {"prompt": 1136, "image_out": 1680, "other_out": 381}
    p = GeminiProvider(client=object(), model="gemini-3.1-flash-image")
    p.usage_log = tmp_path / "_usage.jsonl"
    p._log_usage("m0002|light-light|back", REAL, batch=True)
    c = pricing.actual_cost(p.usage_log, photos=9)
    # batch: 1136 x $0.25/M + 1680 x $30/M + 381 x $1.50/M = $0.0512 = Rs 4.95
    assert c["requests"] == 1 and c["per_request_inr"] == 4.95 and c["extra_images_inr"] == 0
    assert c["per_photo_inr"] == 0.55


def test_extra_images_in_one_response_are_counted_as_waste(tmp_path):
    f = tmp_path / "_usage.jsonl"
    f.write_text(json.dumps({"key": "a", "model": "gemini-3.1-flash-image", "batch": True,
                             "prompt": 1136, "image_out": 4 * 1680, "other_out": 400}) + "\n")
    c = pricing.actual_cost(f)
    assert c["extra_images_inr"] == round(3 * 1680 * 30e-6 * pricing.USD_INR)  # 3 unused images at $30/M (batch)
    assert pricing.actual_cost(tmp_path / "missing.jsonl") is None


def test_prompt_asks_for_exactly_one_image():
    s = studio.Spec(top_colours=["Red"])
    assert studio.ONE_IMAGE in studio.prompt(s, 1, ("light", "light"), "front_main", "none")
