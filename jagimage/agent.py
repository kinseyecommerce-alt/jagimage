"""ImageAgent: turns product info + brand profile into finished e-commerce images."""

import base64
import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .presets import SHOTS, Shot

DEFAULT_MODEL = "gpt-image-2.5-flare"


@dataclass
class Brand:
    name: str = ""
    style: str = ""
    colors: List[str] = field(default_factory=list)
    audience: str = ""
    avoid: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "Brand":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def guidance(self) -> str:
        parts = []
        if self.style:
            parts.append(f"Visual style: {self.style}.")
        if self.colors:
            parts.append(f"Brand palette to echo in props/background: {', '.join(self.colors)}.")
        if self.audience:
            parts.append(f"Target audience: {self.audience}.")
        if self.avoid:
            parts.append(f"Avoid: {', '.join(self.avoid)}.")
        return " ".join(parts)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "product"


class ImageAgent:
    def __init__(
        self,
        client=None,
        model: Optional[str] = None,
        brand: Optional[Brand] = None,
        output_dir: str = "output",
        quality: str = "high",
        output_format: str = "png",
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model or os.environ.get("JAGIMAGE_MODEL", DEFAULT_MODEL)
        self.brand = brand or Brand()
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.output_format = output_format

    def build_prompt(self, product: str, shot: Shot, extra: str = "", from_photo: bool = False) -> str:
        subject = "the exact product shown in the reference image" if from_photo else product
        prompt = shot.template.format(product=subject)
        if from_photo and product:
            prompt += f" The product is: {product}. Keep its shape, colors, materials, labels and logo exactly as in the reference."
        guidance = self.brand.guidance()
        if guidance:
            prompt += " " + guidance
        if extra:
            prompt += " " + extra
        return prompt

    def create(
        self,
        product: str,
        shot: str = "hero",
        photo: Optional[str] = None,
        extra: str = "",
        n: int = 1,
        size: Optional[str] = None,
    ) -> List[Path]:
        """Generate `n` images of one product in one shot style; returns saved paths."""
        if shot not in SHOTS:
            raise ValueError(f"Unknown shot '{shot}'. Choose from: {', '.join(SHOTS)}")
        preset = SHOTS[shot]
        background = preset.background
        output_format = self.output_format
        if background == "transparent" and output_format == "jpeg":
            output_format = "png"  # JPEG cannot hold transparency

        params = dict(
            model=self.model,
            prompt=self.build_prompt(product, preset, extra, from_photo=bool(photo)),
            n=n,
            size=size or preset.size,
            quality=self.quality,
            background=background,
            output_format=output_format,
        )
        if photo:
            with open(photo, "rb") as f:
                result = self.client.images.edit(image=f, **params)
        else:
            result = self.client.images.generate(**params)
        return self._save(result, product, shot, output_format)

    def _save(self, result, product: str, shot: str, output_format: str) -> List[Path]:
        folder = self.output_dir / slugify(product)
        folder.mkdir(parents=True, exist_ok=True)
        ext = "jpg" if output_format == "jpeg" else output_format
        paths = []
        index = 1
        for item in result.data:
            while (folder / f"{shot}-{index}.{ext}").exists():  # never overwrite earlier runs
                index += 1
            path = folder / f"{shot}-{index}.{ext}"
            path.write_bytes(base64.b64decode(item.b64_json))
            paths.append(path)
        return paths

    def run_batch(self, csv_path: str, default_shots: Iterable[str] = ("hero",), n: int = 1):
        """Process a CSV with columns: name, description, shots, photo (only `name` required).

        Yields (product name, shot, saved paths or Exception) so one failure
        doesn't stop the whole catalog.
        """
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                desc = (row.get("description") or "").strip()
                product = f"{name} — {desc}" if desc else name
                shots = [s.strip() for s in (row.get("shots") or "").split(",") if s.strip()]
                photo = (row.get("photo") or "").strip() or None
                for shot in shots or list(default_shots):
                    try:
                        paths = self.create(product, shot=shot, photo=photo, n=n)
                        yield name, shot, paths
                    except Exception as exc:  # report and continue with the next item
                        yield name, shot, exc
