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
    draws = None
    if args.draws:
        start, stop = (int(x) for x in args.draws.split(":"))
        draws = range(start, stop)
    summary = run_policy_study(
        _load_config(args.config), args.regime, validation, args.output_dir, args.workers,
        args.n_bootstrap, draws,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "policies"}, indent=2, default=float))


def _merge(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.policy_study import merge_policy_study
    from dispatch_eval.tracking import tracked_run

    with tracked_run(f"nyc-study-{args.regime}-merged", "policy-study",
                     {"regime": args.regime}, tags=["nyc", args.regime, "merge"]) as run:
        summary = merge_policy_study(args.regime, args.output_dir)
        run.summary.update({"best_policy": summary["best_policy"],
                            "n_draws": summary["n_draws_completed"]})
    print(f"merged {summary['n_draws_completed']} draws; best policy: {summary['best_policy']}")


def _eda(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.eda import run_eda
    from dispatch_eval.tracking import tracked_run

    cfg = _load_config(args.config)
    with tracked_run("nyc-eda", "eda", cfg, tags=["nyc", "eda"]) as run:
        summary = run_eda(cfg, args.output_dir)
        run.summary.update(summary)
    print(json.dumps(summary, indent=2, default=float))


def _report(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.dashboard import build_dashboard, export_pdf
    from dispatch_eval.studies.report import build_report

    for path in build_report(args.results_dir, args.output_dir):
        print(f"wrote {path}")
    print(f"wrote {build_dashboard(args.results_dir, args.dashboard)}")
    if args.pdf:
        print(f"wrote {export_pdf(args.dashboard, args.dashboard.with_suffix('.pdf'))}")


def _refresh(args: argparse.Namespace) -> None:
    from dispatch_eval.studies.policy_study import refresh_policy_study

    cfg = _load_config(args.config)
    hours = cfg["data"]["window_end_hour"] - cfg["data"]["window_start_hour"]
    for regime in args.regimes:
        summary = refresh_policy_study(regime, args.output_dir, float(hours))
        print(f"refreshed {regime}: best policy {summary['best_policy']}")


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
    study.add_argument("--draws", default=None, help="run one shard of draws, e.g. 0:50")
    study.set_defaults(func=_study)

    merge = sub.add_parser("merge", help="combine a regime's shards into the final study files")
    merge.add_argument("--regime", required=True)
    merge.add_argument("--output-dir", type=Path, default=Path("results"))
    merge.set_defaults(func=_merge)

    refresh = sub.add_parser("refresh", help="recompute study summaries from saved per-cell results")
    refresh.add_argument("regimes", nargs="+")
    refresh.add_argument("--config", type=Path, default=Path("configs/nyc.yaml"))
    refresh.add_argument("--output-dir", type=Path, default=Path("results"))
    refresh.set_defaults(func=_refresh)

    eda = sub.add_parser("eda", help="exploratory tables for the NYC study window")
    eda.add_argument("--config", type=Path, default=Path("configs/nyc.yaml"))
    eda.add_argument("--output-dir", type=Path, default=Path("results/eda"))
    eda.set_defaults(func=_eda)

    report = sub.add_parser("report", help="rebuild all figures from saved results")
    report.add_argument("--results-dir", type=Path, default=Path("results"))
    report.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    report.add_argument("--dashboard", type=Path, default=Path("docs/dashboard.html"))
    report.add_argument("--pdf", action="store_true", help="also print the dashboard to PDF (needs Chrome)")
    report.set_defaults(func=_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
