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


def _catalog(args) -> int:
    from . import catalog

    kids = catalog.load_kids(args.kids)
    poses = [p.strip() for p in args.poses.split(",")] if args.poses else list(catalog.POSES)
    unknown = [p for p in poses if p not in catalog.POSES]
    if unknown:
        print(f"Unknown pose(s): {', '.join(unknown)}. Choose from: {', '.join(catalog.POSES)}", file=sys.stderr)
        return 2
    items = list(catalog.skus(kids, args.gender, args.style, args.colour, swatches=args.swatches))
    if not items:
        print("No SKUs match those filters.", file=sys.stderr)
        return 2
    shots = 1 + len(poses) + (1 if args.texture else 0)
    print(f"{len(items)} SKUs x {shots} shots = up to {len(items) * shots} images")
    if args.swatches:
        missing = sorted({s["colour"] for s in items if not s["swatch"]})
        if missing:
            print(f"No swatch photo found for: {', '.join(missing)} (those use the text description only)")
    if args.dry_run:
        for s in items:
            print(f"\n=== {s['gender']}/{s['style']}/{s['colour']} ===")
            print(f"[{catalog.FRONT}] {catalog.front_prompt(s)}")
            for p in poses:
                print(f"[{p}] {catalog.pose_prompt(s, p)}")
            if args.texture:
                print(f"[{catalog.TEXTURE}] {catalog.texture_prompt(s)}")
        return 0
    agent = catalog.CatalogAgent(model=args.model, output_dir=args.out, quality=args.quality,
                                 size=args.size, fidelity=args.fidelity)
    failures = 0
    for s in items:
        sku = f"{s['gender']}/{s['style']}/{s['colour']}"
        try:
            for shot, path, status in agent.run_sku(s, poses=poses, force=args.force, texture=args.texture):
                print(f"{sku} [{shot}] {status} {path}")
        except Exception as exc:  # poses depend on the front shot, so move on to the next SKU
            failures += 1
            print(f"{sku} failed: {exc}", file=sys.stderr)
    return 1 if failures else 0


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

    cat = sub.add_parser("catalog", help="Kids' tee catalogue: front + pose shots for every style x colour")
    cat.add_argument("--kids", required=True, help="JSON with one child model per gender (see examples/kids.json)")
    cat.add_argument("--gender", choices=["Boys", "Girls"])
    cat.add_argument("--style", help="e.g. HalfSleeveTee, FullSleeveTee, SleevelessTee")
    cat.add_argument("--colour", help="e.g. Navy")
    cat.add_argument("--poses", help="Comma-separated poses (default: all), e.g. 02-threequarter,04-back")
    cat.add_argument("--texture", action="store_true", help="Also make a fabric close-up (06-fabric) per SKU")
    cat.add_argument("--swatches", help="Folder of real fabric photos named <Colour>.jpg or <Style>-<Colour>.jpg, used as texture/colour reference")
    cat.add_argument("--model", help=f"Image model (default: $JAGIMAGE_MODEL or {DEFAULT_MODEL})")
    cat.add_argument("--out", default="output")
    cat.add_argument("--quality", default="high", choices=["low", "medium", "high", "xhigh", "max", "auto"])
    cat.add_argument("--size", default="1024x1536")
    cat.add_argument("--fidelity", choices=["high", "low"], help="input_fidelity for reference edits, if your model supports it")
    cat.add_argument("--force", action="store_true", help="Regenerate images that already exist")
    cat.add_argument("--dry-run", action="store_true", help="Print the prompts without calling the API")

    args = parser.parse_args(argv)

    if args.command == "shots":
        for s in SHOTS.values():
            print(f"{s.key:10} {s.size:10} {s.label}")
        return 0

    if args.command == "catalog":
        return _catalog(args)

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
