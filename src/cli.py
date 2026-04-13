"""Command-line interface for Seedcraft."""

from __future__ import annotations

import argparse
import json
import sys

from .engine import SeedEngine
from .generator import DataGenerator
from .presets import PRESETS
from .analyzer import describe_dataset, correlation_matrix, format_stats, format_correlation_matrix
from .exporters import to_csv, to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seedcraft",
        description="Seedcraft — Deterministic, correlation-aware synthetic data generation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- generate --
    gen_parser = subparsers.add_parser("generate", help="Generate data from a preset schema")
    gen_parser.add_argument("preset", choices=list(PRESETS.keys()),
                            help="Which preset schema to use")
    gen_parser.add_argument("-n", "--rows", type=int, default=None,
                            help="Number of rows (overrides preset default)")
    gen_parser.add_argument("-s", "--seeds", type=int, nargs="+",
                            default=[3813, 7889, 6140, 439],
                            help="Seed numbers for deterministic generation")
    gen_parser.add_argument("-f", "--format", choices=["csv", "json", "json-columns"],
                            default="csv", help="Output format")
    gen_parser.add_argument("-o", "--output", type=str, default=None,
                            help="Output file path (stdout if omitted)")

    # -- analyze --
    analyze_parser = subparsers.add_parser("analyze", help="Generate and analyze a preset dataset")
    analyze_parser.add_argument("preset", choices=list(PRESETS.keys()))
    analyze_parser.add_argument("-n", "--rows", type=int, default=None)
    analyze_parser.add_argument("-s", "--seeds", type=int, nargs="+",
                                default=[3813, 7889, 6140, 439])

    # -- list --
    subparsers.add_parser("list", help="List available presets")

    # -- info --
    info_parser = subparsers.add_parser("info", help="Show details about a preset schema")
    info_parser.add_argument("preset", choices=list(PRESETS.keys()))

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "list":
        return _cmd_list()
    elif args.command == "info":
        return _cmd_info(args.preset)
    elif args.command == "generate":
        return _cmd_generate(args)
    elif args.command == "analyze":
        return _cmd_analyze(args)
    else:
        parser.print_help()
        return 1


def _cmd_list() -> int:
    print("Available presets:\n")
    for name, factory in PRESETS.items():
        schema = factory()
        print(f"  {name}")
        print(f"    Fields: {', '.join(schema.field_names())}")
        print(f"    Default rows: {schema.row_count}")
        print(f"    Correlations: {len(schema.correlations)}")
        print()
    return 0


def _cmd_info(preset_name: str) -> int:
    schema = PRESETS[preset_name]()
    info = schema.to_dict()
    print(json.dumps(info, indent=2))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    schema = PRESETS[args.preset]()
    if args.rows is not None:
        schema.row_count = args.rows

    engine = SeedEngine(args.seeds)
    gen = DataGenerator(engine)
    data = gen.generate(schema)

    if args.format == "csv":
        output = to_csv(data, path=args.output)
    elif args.format == "json":
        output = to_json(data, path=args.output, orient="records")
    elif args.format == "json-columns":
        output = to_json(data, path=args.output, orient="columns")
    else:
        output = to_csv(data)

    if args.output:
        print(f"Written to {output}")
    else:
        print(output)

    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    schema = PRESETS[args.preset]()
    if args.rows is not None:
        schema.row_count = args.rows

    engine = SeedEngine(args.seeds)
    gen = DataGenerator(engine)
    data = gen.generate(schema)

    stats = describe_dataset(data)
    corr = correlation_matrix(data)

    print(f"=== {schema.name} ({schema.row_count} rows) ===")
    print(f"\nSeeds: {args.seeds}")
    print(f"\n--- Column Statistics ---")
    print(format_stats(stats))
    print(f"\n--- Correlation Matrix (numeric columns) ---")
    print(format_correlation_matrix(corr))

    return 0


if __name__ == "__main__":
    sys.exit(main())
