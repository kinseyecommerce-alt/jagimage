"""Command-line interface: `jagimage create ...`, `jagimage batch ...`, `jagimage shots`."""

import argparse
from pathlib import Path
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


def _csv(value):
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _library(args) -> int:
    from . import library as L

    if args.export:
        L.export(args.export)
        print(f"Wrote {args.export}")
        return 0
    checks = [(_csv(args.category), L.CATEGORIES, "category"), (_csv(args.product), L.PRODUCTS, "product"),
              (_csv(args.fabric), list(L.FABRICS), "fabric"), (_csv(args.poses), list(L.POSES), "pose")]
    for given, allowed, name in checks:
        bad = [g for g in given or [] if g not in allowed]
        if bad:
            print(f"Unknown {name}: {', '.join(bad)}. Choose from: {', '.join(allowed)}", file=sys.stderr)
            return 2
    poses = _csv(args.poses) or list(L.POSES)
    items = list(L.items(_csv(args.category), _csv(args.product), _csv(args.fabric), _csv(args.colour)))
    if not items:
        print("No SKUs match those filters (colour names must match the fabric's palette).", file=sys.stderr)
        return 2
    models = L.load_models(args.models)
    if args.bases:
        return _library_bases(args, L, items, poses, models)
    print(f"{len(items)} SKUs x {len(poses)} poses = up to {len(items) * len(poses)} images")
    if args.dry_run:
        runner = L.LibraryRunner(None, models, out=args.out, negative=not args.no_negative, face_by=args.face_by)
        for it in items:
            ident = runner.identity(it)
            who = Path(ident["face"]).name if ident.get("face") else "new face"
            extra = "; ".join(f"{k}: {ident[k]}" for k in ("hairstyle", "look") if ident.get(k))
            print(f"\n=== {it['category']}/{it['product']}/{it['fabric']}/{it['colour']}  (model: {who}{'; ' + extra if extra else ''}) ===")
            for pose in poses:
                print(f"[{pose}] {runner.prompt(it, pose)}")
        return 0
    runner = L.LibraryRunner(_engine(args), models, out=args.out, negative=not args.no_negative, face_by=args.face_by)
    failures = 0
    for it in items:
        sku = f"{it['category']}/{it['product']}/{it['fabric']}/{it['colour']}"
        try:
            for pose, path, status in runner.run(it, poses, stage=args.stage, force=args.force):
                print(f"{sku} [{pose}] {status} {path}")
        except Exception as exc:
            failures += 1
            print(f"{sku} failed: {exc}", file=sys.stderr)
    return 1 if failures else 0


def _engine(args):
    from .catalog import CatalogAgent
    from .providers import make_provider

    provider = make_provider(args.provider, model=args.model, quality=args.quality, size=args.size,
                             fidelity=args.fidelity, image_size=args.gemini_size)
    return CatalogAgent(provider=provider, output_dir=args.out)


def _library_bases(args, L, items, poses, models) -> int:
    """AI makes each style in two base colours; every palette colour is recoloured from them."""
    named = {"#8FD3A8": "mint green", "#1E4636": "bottle green"}
    light, dark = args.base_light.upper(), args.base_dark.upper()
    bases = {"light": (named.get(light, "colour " + light), light), "dark": (named.get(dark, "colour " + dark), dark)}
    styles = list({(i["category"], i["product"], i["fabric"]): i for i in items}.values())
    ai = len(styles) * 2 * len(poses)
    print(f"{len(styles)} styles x 2 base colours x {len(poses)} poses = {ai} AI images; "
          f"{len(items)} colours x {len(poses)} poses = {len(items) * len(poses)} recoloured images")
    runner = L.LibraryRunner(None, models, out=args.out, negative=not args.no_negative, face_by=args.face_by)
    if args.dry_run:
        for st in styles:
            for which in ("light", "dark"):
                b = L.base_item(st, which, bases)
                print(f"\n=== {st['category']}/{st['product']}/{st['fabric']} {which} base {b['hex']} ===")
                for pose in poses:
                    print(f"[{pose}] {runner.prompt(b, pose)}")
        print("\nColours and the base each one is made from:")
        for it in items:
            print(f"  {it['category']}/{it['product']}/{it['fabric']}/{it['colour']} <- {L.base_for(it)} base")
        return 0
    failures = 0
    if args.stage != "recolor":
        runner.agent = _engine(args)
        for st in styles:
            for which in ("light", "dark"):
                b = L.base_item(st, which, bases)
                tag = f"{st['category']}/{st['product']}/{st['fabric']} {which} base"
                try:
                    for pose, path, status in runner.run(b, poses, stage=args.stage, force=args.force):
                        print(f"{tag} [{pose}] {status} {path}")
                except Exception as exc:
                    failures += 1
                    print(f"{tag} failed: {exc}", file=sys.stderr)
    if args.stage in ("all", "poses", "recolor"):
        for it in items:
            sku = f"{it['category']}/{it['product']}/{it['fabric']}/{it['colour']}"
            for pose, path, status, de in L.recolor_item(it, poses, out=args.out, bases=bases, mode=args.recolor_mode, force=args.force):
                note = f" ΔE {de:.2f}" + ("" if de < 3 else " CHECK") if de is not None else ""
                print(f"{sku} [{pose}] {status}{note} {path}")
                if status.startswith("needs a mask") or (de is not None and de >= 3):
                    failures += 1
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

    lib = sub.add_parser("library", help="SPB Textile prompt library: Men/Women/Kids x 7 products x 2 fabrics x 6 poses")
    lib.add_argument("--models", help="JSON with a pool of models per category (see examples/models.json); omit for a new face per SKU")
    lib.add_argument("--face-by", default="sku", choices=["sku", "product"],
                     help="sku = a different model for every product/fabric/colour; product = one model per product+fabric across its colours")
    lib.add_argument("--category", help="Comma-separated: Men, Women, Kids_Boys, Kids_Girls")
    lib.add_argument("--product", help="Comma-separated: HalfSleeveTee, FullSleeveTee, SleevelessTee, Shorts, Trackpants, Hoodie, Sweatshirt")
    lib.add_argument("--fabric", help="Comma-separated: PureCotton, CottonMix")
    lib.add_argument("--colour", help='Comma-separated colour names, e.g. "Navy Blue,Light Pink"')
    lib.add_argument("--poses", help="Comma-separated: front_main, front_main_white, threequarter, side, back, handpocket")
    lib.add_argument("--stage", default="all", choices=["fronts", "poses", "all", "recolor"],
                     help="fronts = front_main(+white) only for review; poses = other poses from approved fronts; "
                          "recolor = (with --bases) only recolour from existing base photos")
    lib.add_argument("--provider", default="openai", choices=["openai", "gemini"], help="Image engine (default: openai)")
    lib.add_argument("--gemini-size", default="2K", choices=["1K", "2K", "4K"], help="Gemini output resolution (default: 2K HD)")
    lib.add_argument("--bases", action="store_true",
                     help="AI makes each style in 2 base colours only; every palette colour is recoloured from them to its exact hex")
    lib.add_argument("--base-light", default="#8FD3A8", help="Light base colour (default: mint #8FD3A8)")
    lib.add_argument("--base-dark", default="#1E4636", help="Dark base colour (default: bottle green #1E4636)")
    lib.add_argument("--recolor-mode", default="match", choices=["match", "hue", "scale"],
                     help="match = exact hex on average; hue = keep lightness; scale = scale shading")
    lib.add_argument("--no-negative", action="store_true", help="Don't append the negative list as an 'Avoid:' line")
    lib.add_argument("--export", metavar="FILE", help="Write the whole library as prompts.json and exit")
    lib.add_argument("--model", help=f"Image model (default: $JAGIMAGE_MODEL or {DEFAULT_MODEL})")
    lib.add_argument("--out", default="output")
    lib.add_argument("--quality", default="high", choices=["low", "medium", "high", "xhigh", "max", "auto"])
    lib.add_argument("--size", default="1024x1536")
    lib.add_argument("--fidelity", choices=["high", "low"], help="input_fidelity for reference edits, if your model supports it")
    lib.add_argument("--force", action="store_true", help="Regenerate images that already exist")
    lib.add_argument("--dry-run", action="store_true", help="Print the prompts without calling the API")

    args = parser.parse_args(argv)

    if args.command == "shots":
        for s in SHOTS.values():
            print(f"{s.key:10} {s.size:10} {s.label}")
        return 0

    if args.command == "catalog":
        return _catalog(args)
    if args.command == "library":
        return _library(args)

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
