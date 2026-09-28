"""Static figures for the README and report, drawn from saved results only.

Every figure is rebuilt from `results/` (validation JSON, EDA tables, study
summaries) — never by re-running a simulation — so a figure can't drift
from the numbers it claims to show.

Colors: a fixed categorical order per policy (color follows the policy, not
its rank), one-hue sequential blue for magnitude, neutral gray for the
greedy baseline. Values are direct-labelled because two of the categorical
hues sit below 3:1 contrast on the light surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from dispatch_eval.policies.registry import POLICY_LABELS

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BASELINE = "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

POLICY_COLORS = {"greedy": BASELINE} | dict(
    zip(["batched", "value_aware", "fluid_rebalance", "lookahead_rebalance"], SERIES, strict=True)
)
REGIME_ORDER = ["shortage", "tight", "mid", "abundant"]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "font.size": 10,
        "legend.frameon": False,
    })


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def validation_quantiles(validation: dict, out: Path) -> Path:
    q = validation["quantiles"]
    labels = [f"p{int(r['quantile'] * 100)}" for r in q]
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, name, color, dx in [
        ("held_out_wait", "Real (held-out days)", SERIES[0], -0.12),
        ("simulated_wait", "Simulated twin", SERIES[1], 0.12),
    ]:
        values = [r[key] for r in q]
        ax.scatter(x + dx, values, s=46, color=color, edgecolor=SURFACE, linewidth=2, label=name,
                   zorder=3)
    for i, r in enumerate(q):
        ax.annotate(f"{r['held_out_wait']:.0f}s / {r['simulated_wait']:.0f}s", (i, r["simulated_wait"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8,
                    color=TEXT_MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rider wait (seconds)")
    ax.set_title(f"Held-out validation: KS distance {validation['wait_ks_distance']:.3f} "
                 "(threshold 0.10)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(r["simulated_wait"] for r in q) * 1.2)
    return _save(fig, out / "validation_quantiles.png")


def calibration_grid(validation: dict, out: Path) -> Path:
    grid = pl.DataFrame(validation["calibration_grid"])
    fleets = sorted(grid["fleet_size"].unique())
    medians = sorted(grid["intra_zone_median_seconds"].unique())
    ks = np.full((len(medians), len(fleets)), np.nan)
    for row in grid.iter_rows(named=True):
        ks[medians.index(row["intra_zone_median_seconds"]), fleets.index(row["fleet_size"])] = (
            row["ks_distance"]
        )
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    # Dark = close fit (low KS), so the eye lands on the good region.
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(1 - ks, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(medians)):
        for j in range(len(fleets)):
            r, g, b = to_rgb(cmap(norm(1 - ks[i, j])))
            dark_cell = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45
            ax.text(j, i, f"{ks[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark_cell else TEXT,
                    fontweight="bold" if ks[i, j] <= 0.10 else "normal")
    ax.set_xticks(range(len(fleets)), [f"{f:,}" for f in fleets])
    ax.set_yticks(range(len(medians)), [f"{m}s" for m in medians])
    ax.set_xlabel("Fleet size (vehicles)")
    ax.set_ylabel("Same-zone pickup median")
    ax.grid(False)
    ax.set_title("Supply calibration: KS distance to calibration-day waits (bold = within 0.10)")
    return _save(fig, out / "calibration_grid.png")


def demand_by_slot(eda_dir: Path, out: Path) -> Path:
    t = pl.read_parquet(eda_dir / "demand_by_slot.parquet")
    x = np.arange(t.height)
    labels = [s.strftime("%H:%M") for s in t["slot"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.fill_between(x, t["min_requests"], t["max_requests"], color=BLUES[0], linewidth=0,
                    label="Range across 14 weekdays")
    ax.plot(x, t["mean_requests"], color=SERIES[0], linewidth=2, label="Mean")
    peak = int(np.argmax(t["mean_requests"]))
    ax.annotate(f"{t['mean_requests'][peak]:,.0f} requests / 15 min", (x[peak], t["mean_requests"][peak]),
                textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8, color=TEXT)
    ax.set_xticks(x[::4], labels[::4])
    ax.set_ylabel("Requests per 15 min")
    ax.set_ylim(0, None)
    ax.set_title("Manhattan Uber demand, weekdays 12:00-18:00 (Jan 2024)")
    ax.legend(loc="upper left")
    return _save(fig, out / "demand_by_slot.png")


def wait_components(eda_dir: Path, out: Path) -> Path:
    t = pl.read_parquet(eda_dir / "wait_components_by_hour.parquet")
    labels = [f"{h}:00" for h in t["hour"]]
    x = np.arange(t.height)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x, t["mean_approach_s"], width=0.62, color=SERIES[0], label="Driver approach")
    ax.bar(x, t["mean_boarding_s"], width=0.62, bottom=t["mean_approach_s"] + 2, color=SERIES[1],
           label="Rider boarding")
    for i, total in enumerate(t["mean_wait_s"]):
        ax.text(i, total + 6, f"{total:.0f}s", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean seconds")
    ax.set_title("Where the wait goes: approach vs boarding, by hour")
    ax.legend(loc="upper left", ncols=2)
    ax.set_ylim(0, float(t["mean_wait_s"].max()) * 1.18)
    return _save(fig, out / "wait_components_by_hour.png")


def zone_waits(eda_dir: Path, out: Path, k: int = 12) -> Path:
    t = pl.read_parquet(eda_dir / "zone_service.parquet").head(k).reverse()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(t.height)
    ax.barh(y, t["median_wait_s"], height=0.62, color=SERIES[0])
    for i, (w, d) in enumerate(zip(t["median_wait_s"], t["requests_per_day"], strict=True)):
        ax.text(w + 3, i, f"{w:.0f}s  ·  {d:,.0f}/day", va="center", fontsize=8, color=TEXT_MUTED)
    ax.set_yticks(y, t["name"])
    ax.set_xlabel("Median rider wait (seconds)")
    ax.set_xlim(0, float(t["median_wait_s"].max()) * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Median wait in the {k} busiest pickup zones")
    return _save(fig, out / "zone_waits.png")


def _load_studies(results_dir: Path) -> dict[str, dict]:
    studies = {}
    for regime in REGIME_ORDER:
        path = results_dir / f"nyc_study_{regime}.json"
        if path.exists():
            studies[regime] = json.loads(path.read_text())
    return studies


def _per_regime_panels(studies: dict[str, dict], value, interval, xlabel: str, title: str,
                       fmt: str, path: Path) -> Path:
    regimes = list(studies)
    policies = [p for p in POLICY_LABELS if p != "greedy"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.1 * len(regimes), 3.4), sharey=True,
                             squeeze=False)
    for ax, regime in zip(axes[0], regimes, strict=True):
        s = studies[regime]
        for i, name in enumerate(policies):
            v, (lo, hi) = value(s, name), interval(s, name)
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=POLICY_COLORS[name],
                        markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=2,
                        capsize=0)
            ax.annotate(fmt.format(v), (hi, i), textcoords="offset points", xytext=(5, 0),
                        va="center", fontsize=8, color=TEXT_MUTED)
        ax.axvline(0, color=BASELINE, linewidth=1)
        ax.set_title(f"{regime}  ·  {s['fleet_size']:,} cars", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.35)
    axes[0][0].set_yticks(range(len(policies)), [POLICY_LABELS[p] for p in policies])
    axes[0][0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12, color=TEXT)
    fig.supxlabel(xlabel, fontsize=10, color=TEXT_MUTED)
    fig.tight_layout()
    return _save(fig, path)


def policy_gains(studies: dict[str, dict], out: Path) -> Path:
    def by(s, name):
        return next(p for p in s["policies"] if p["policy"] == name)

    def pct(s, name):
        ref = by(s, "greedy")["mean_wait_seconds"]
        return 100 * by(s, name)["gain_vs_reference_seconds"] / ref

    def pct_ci(s, name):
        ref = by(s, "greedy")["mean_wait_seconds"]
        lo, hi = by(s, name)["gain_vs_reference_ci"]
        return 100 * lo / ref, 100 * hi / ref

    return _per_regime_panels(
        studies, pct, pct_ci, "Wait reduction vs greedy (%)",
        "Wait-time reduction vs greedy dispatch, 95% bootstrap intervals", "{:.1f}%",
        out / "policy_gains.png",
    )


def vehicles_worth(studies: dict[str, dict], out: Path) -> Path:
    return _per_regime_panels(
        studies,
        lambda s, n: s["vehicles_worth"][n]["vehicles_worth_mean"],
        lambda s, n: tuple(s["vehicles_worth"][n]["vehicles_worth_ci"]),
        "Extra cars greedy would need",
        "Each algorithm's gain priced in vehicles (fleet-equivalent vs greedy)", "{:,.0f}",
        out / "vehicles_worth.png",
    )


def build_report(results_dir: Path, out: Path) -> list[Path]:
    plt.switch_backend("Agg")  # files only, no display needed
    _style()
    out.mkdir(parents=True, exist_ok=True)
    validation = json.loads((results_dir / "nyc_validation.json").read_text())
    eda_dir = results_dir / "eda"
    figures = [
        validation_quantiles(validation, out),
        calibration_grid(validation, out),
        demand_by_slot(eda_dir, out),
        wait_components(eda_dir, out),
        zone_waits(eda_dir, out),
    ]
    studies = _load_studies(results_dir)
    if studies:
        figures += [policy_gains(studies, out), vehicles_worth(studies, out)]
    return figures
