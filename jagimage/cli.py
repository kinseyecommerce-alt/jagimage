"""Command-line interface: `jagimage create ...`, `jagimage batch ...`, `jagimage shots`."""

import argparse
import sys

from .agent import DEFAULT_MODEL, Brand, ImageAgent
from .presets import SHOTS


def _agent(args) -> ImageAgent:
    brand = Brand.load(args.brand) if args.brand else None
    return ImageAgent(
        model=args.model,
        brand=brand,
        output_dir=args.out,
        quality=args.quality,
        output_format=args.format,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="jagimage", description="E-commerce image agent")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help=f"Image model (default: $JAGIMAGE_MODEL or {DEFAULT_MODEL})")
    common.add_argument("--brand", help="Path to a brand profile JSON (style, colors, audience, avoid)")
    common.add_argument("--out", default="output", help="Output folder (default: output)")
    common.add_argument("--quality", default="high", choices=["low", "medium", "high", "xhigh", "max", "auto"])
    common.add_argument("--format", default="png", choices=["png", "jpeg", "webp"])
    common.add_argument("-n", type=int, default=1, help="Variations per shot")

    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", parents=[common], help="Generate images for one product")
    create.add_argument("product", help='Product description, e.g. "matte black steel water bottle"')
    create.add_argument("--shots", default="hero", help="Comma-separated shot types (see `jagimage shots`)")
    create.add_argument("--photo", help="Your real product photo; keeps the product exact and restyles the scene")
    create.add_argument("--extra", default="", help="Extra art direction appended to the prompt")
    create.add_argument("--size", help="Override size, e.g. 1024x1024, 1536x1024, 1024x1536")

    batch = sub.add_parser("batch", parents=[common], help="Generate images for every product in a CSV")
    batch.add_argument("csv", help="CSV with columns: name, description, shots, photo")
    batch.add_argument("--shots", default="hero", help="Default shots when a row leaves `shots` empty")

    sub.add_parser("shots", help="List available shot types")

    args = parser.parse_args(argv)

    if args.command == "shots":
        for s in SHOTS.values():
            print(f"{s.key:10} {s.size:10} {s.label}")
        return 0

    agent = _agent(args)
    failures = 0
    if args.command == "create":
        for shot in [s.strip() for s in args.shots.split(",") if s.strip()]:
            try:
                for path in agent.create(args.product, shot, photo=args.photo, extra=args.extra, n=args.n, size=args.size):
                    print(f"[{shot}] saved {path}")
            except Exception as exc:
                failures += 1
                print(f"[{shot}] failed: {exc}", file=sys.stderr)
    else:
        defaults = [s.strip() for s in args.shots.split(",") if s.strip()]
        for name, shot, result in agent.run_batch(args.csv, default_shots=defaults, n=args.n):
            if isinstance(result, Exception):
                failures += 1
                print(f"{name} [{shot}] failed: {result}", file=sys.stderr)
            else:
                for path in result:
                    print(f"{name} [{shot}] saved {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
