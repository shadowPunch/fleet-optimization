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


def _study(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.policy_study import run_policy_study

    validation = json.loads(args.validation.read_text())
    summary = run_policy_study(
        _load_config(args.config), args.regime, validation, args.output_dir, args.workers,
        args.n_bootstrap,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "policies"}, indent=2, default=float))


def _eda(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.eda import run_eda
    from dispatch_eval.tracking import tracked_run

    cfg = _load_config(args.config)
    with tracked_run("nyc-eda", "eda", cfg, tags=["nyc", "eda"]) as run:
        summary = run_eda(cfg, args.output_dir)
        run.summary.update(summary)
    print(json.dumps(summary, indent=2, default=float))


def _report(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.dashboard import build_dashboard
    from dispatch_eval.studies.report import build_report

    for path in build_report(args.results_dir, args.output_dir):
        print(f"wrote {path}")
    print(f"wrote {build_dashboard(args.results_dir, args.dashboard)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="dispatch-eval", description=__doc__)
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser("validate", help="calibrate the NYC twin and validate on held-out days")
    validate.add_argument("--config", type=Path, default=Path("configs/nyc.yaml"))
    validate.add_argument("--output", type=Path, default=Path("results/nyc_validation.json"))
    validate.set_defaults(func=_validate)

    study = sub.add_parser("study", help="tune and compare the policy ladder on the NYC twin")
    study.add_argument("--config", type=Path, default=Path("configs/nyc.yaml"))
    study.add_argument("--regime", default="calibrated", help="a key under study.regimes")
    study.add_argument("--validation", type=Path, default=Path("results/nyc_validation.json"))
    study.add_argument("--output-dir", type=Path, default=Path("results"))
    study.add_argument("--workers", type=int, default=1)
    study.add_argument("--n-bootstrap", type=int, default=None, help="override the config")
    study.set_defaults(func=_study)

    eda = sub.add_parser("eda", help="exploratory tables for the NYC study window")
    eda.add_argument("--config", type=Path, default=Path("configs/nyc.yaml"))
    eda.add_argument("--output-dir", type=Path, default=Path("results/eda"))
    eda.set_defaults(func=_eda)

    report = sub.add_parser("report", help="rebuild all figures from saved results")
    report.add_argument("--results-dir", type=Path, default=Path("results"))
    report.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    report.add_argument("--dashboard", type=Path, default=Path("docs/dashboard.html"))
    report.set_defaults(func=_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
