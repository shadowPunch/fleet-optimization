"""Command-line entry point: `dispatch-eval <command> --config configs/nyc.yaml`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {path}")


def _validate(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.validation import run_validation

    report = run_validation(_load_config(args.config))
    summary = {k: v for k, v in report.items() if not isinstance(v, list)}
    print(json.dumps(summary, indent=2, default=float))
    _write_json(report, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="dispatch-eval", description=__doc__)
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser("validate", help="calibrate the NYC twin and validate on held-out days")
    validate.add_argument("--config", type=Path, default=Path("configs/nyc.yaml"))
    validate.add_argument("--output", type=Path, default=Path("results/nyc_validation.json"))
    validate.set_defaults(func=_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
