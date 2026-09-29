"""Build the self-contained HTML dashboard from saved results.

The page (`dashboard_template.html`) is static HTML + inline SVG charts; this
module only gathers the numbers from `results/` and embeds them as JSON, so
the dashboard shows exactly what the result files contain.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from dispatch_eval.policies.registry import POLICY_LABELS
from dispatch_eval.studies.report import MIN_CURVE_RESPONSE, REGIME_ORDER, greedy_curve_response

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
PLACEHOLDER = "/*__DATA__*/null"


def _eda(eda_dir: Path) -> dict:
    demand = pl.read_parquet(eda_dir / "demand_by_slot.parquet")
    components = pl.read_parquet(eda_dir / "wait_components_by_hour.parquet")
    return {
        "summary": json.loads((eda_dir / "summary.json").read_text()),
        "demand": [
            {"slot": r["slot"].strftime("%H:%M"), "mean": r["mean_requests"],
             "min": r["min_requests"], "max": r["max_requests"]}
            for r in demand.iter_rows(named=True)
        ],
        "components": [
            {"hour": r["hour"], "trips": r["trips"], "wait": r["mean_wait_s"],
             "approach": r["mean_approach_s"], "boarding": r["mean_boarding_s"]}
            for r in components.iter_rows(named=True)
        ],
    }


def dashboard_payload(results_dir: Path) -> dict:
    validation = json.loads((results_dir / "nyc_validation.json").read_text())
    studies = {
        regime: json.loads(path.read_text())
        for regime in REGIME_ORDER
        if (path := results_dir / f"nyc_study_{regime}.json").exists()
    }
    for study in studies.values():
        study["cars_priceable"] = (
            "fleet_curve" in study and greedy_curve_response(study) >= MIN_CURVE_RESPONSE
        )
    return {
        "generated": date.today().isoformat(),
        "labels": POLICY_LABELS,
        "policy_order": list(POLICY_LABELS),
        "regime_order": REGIME_ORDER,
        "eda": _eda(results_dir / "eda"),
        "validation": {
            "wait_ks_distance": validation["wait_ks_distance"],
            "hour_of_day_cosine": validation["hour_of_day_cosine"],
            "quantiles": validation["quantiles"],
            "grid": validation["calibration_grid"],
        },
        "studies": studies,
    }


def build_dashboard(results_dir: Path, output: Path) -> Path:
    template = TEMPLATE.read_text()
    if PLACEHOLDER not in template:
        raise ValueError(f"{TEMPLATE} has no {PLACEHOLDER} placeholder")
    payload = json.dumps(dashboard_payload(results_dir), default=float).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(PLACEHOLDER, payload))
    return output


CHROME_NAMES = ("google-chrome", "chromium", "chromium-browser", "chrome")


def export_pdf(dashboard_html: Path, pdf: Path) -> Path:
    """Print the dashboard to an A4 PDF (light theme) with headless Chrome.

    The page is authored as a body fragment for the artifact host, so it is
    wrapped in a full document pinned to the light theme before printing.
    """
    chrome = next((path for name in CHROME_NAMES if (path := shutil.which(name))), None)
    if chrome is None:
        raise RuntimeError(f"PDF export needs one of {CHROME_NAMES} on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "dashboard.html"
        page.write_text(
            '<!doctype html><html data-theme="light"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
            f'<body style="margin:0">{dashboard_html.read_text()}</body></html>'
        )
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=10000", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf.resolve()}", page.as_uri()],
            check=True, capture_output=True,
        )
    return pdf
