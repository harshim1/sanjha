"""
The two demo artifacts named in the 24-hour build plan:
  - hours 0-3: "the decomposition chart"
  - hours 3-8: "reliability diagram; coverage through the 2024-25 spike"

Colors are the validated default categorical palette from the dataviz skill
(references/palette.md) — not eyeballed. One axis per panel throughout (no
dual-axis charts); different units (USD/lb futures vs. a dimensionless gap)
get separate stacked panels, never a second y-axis on the same plot.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.run_backtest import run
from models.decomposition import decomposition_error, recover_local_gap

# Validated categorical palette (light mode) — slots 1/2/3, in fixed order.
BLUE = "#2a78d6"  # slot 1 — model / Sanjha
ORANGE = "#eb6834"  # slot 2 — naive baseline
AQUA = "#1baf7a"  # slot 3 — ground truth (ok to use since it's not a 4th categorical series here)
GRAY_MID = "#52514e"  # text-secondary
GRAY_LIGHT = "#cfcdc6"
SPIKE_TINT = "#f6d9d6"  # light wash for the injected spike-regime band, not a status color on data
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRAY_LIGHT,
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": GRAY_MID,
        "ytick.color": GRAY_MID,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

OUT_DIR = os.path.dirname(__file__)


def decomposition_chart(results: dict) -> str:
    from data import fetch

    data = fetch.load_all()
    futures = data["futures"]
    recovered = recover_local_gap(data["farmgate_anchors"]).sort_values("date")
    err = decomposition_error(recovered)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    # Top: futures price, spike regime shaded
    ax_top.plot(futures["date"], futures["close"], color=BLUE, linewidth=1.4, label="Arabica futures (USD/lb)")
    if "in_spike_regime" in futures.columns and futures["in_spike_regime"].any():
        spike_dates = futures.loc[futures["in_spike_regime"], "date"]
        ax_top.axvspan(spike_dates.min(), spike_dates.max(), color=SPIKE_TINT, zorder=0, label="Spike regime")
    ax_top.set_title("Global futures price — the forecast Sanjha borrows, not predicts", loc="left", fontsize=11)
    ax_top.set_ylabel("USD / lb")
    ax_top.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.92, loc="upper left")

    # Bottom: recovered vs true local gap at farmgate anchors
    ax_bot.plot(recovered["date"], recovered["true_local_gap"], color=AQUA, linewidth=1.8, marker="o", markersize=3, label="True local gap (synthetic ground truth)")
    ax_bot.plot(recovered["date"], recovered["recovered_local_gap"], color=BLUE, linewidth=1.8, marker="o", markersize=3, label="Recovered local gap (decomposition)")
    ax_bot.set_title(
        f"Local gap b$_{{v,t}}$ recovered from farmgate anchors — RMSE {err.get('rmse', float('nan')):.3f}",
        loc="left",
        fontsize=11,
    )
    ax_bot.set_ylabel("log-gap (dimensionless)")
    ax_bot.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.92, loc="upper left")

    fig.suptitle("Sanjha — price decomposition", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT_DIR, "decomposition_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def reliability_diagram(results: dict) -> str:
    rel = results["reliability"]
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0.45, 1.0], [0.45, 1.0], color=GRAY_LIGHT, linewidth=1.5, linestyle="--", label="Perfect calibration")
    ax.plot(
        rel["nominal_coverage"], rel["achieved_coverage"], color=BLUE, marker="o", linewidth=2, markersize=6,
        label="Sanjha (adaptive conformal)",
    )
    for _, row in rel.iterrows():
        ax.annotate(
            f"{row['achieved_coverage']:.2f}",
            (row["nominal_coverage"], row["achieved_coverage"]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8,
            color=GRAY_MID,
        )

    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Nominal coverage (target)")
    ax.set_ylabel("Achieved coverage (backtest)")
    ax.set_title("Reliability diagram — synthetic backtest, 149 weeks", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "reliability_diagram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def coverage_spike_chart(results: dict) -> str:
    model = results["model"].copy()
    baseline = results["baseline"]
    model["date"] = pd.to_datetime(model["date"])
    baseline_date = pd.to_datetime(baseline["date"])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Top: price with model's adaptive band + actual
    if model["in_spike_regime"].any():
        spike_dates = model.loc[model["in_spike_regime"].astype(bool), "date"]
        ax_top.axvspan(spike_dates.min(), spike_dates.max(), color=SPIKE_TINT, zorder=0, label="Spike regime")
    ax_top.fill_between(model["date"], model["lower_price"], model["upper_price"], color=BLUE, alpha=0.18, linewidth=0, label="Sanjha 80% band (adaptive)")
    ax_top.plot(model["date"], model["median_price"], color=BLUE, linewidth=1.2)
    ax_top.plot(model["date"], np.exp(model["log_actual"]), color="#0b0b0b", linewidth=1.0, label="Realized price")
    ax_top.set_ylabel("USD / lb")
    ax_top.set_title("Adaptive-conformal band vs. realized price, through the spike regime", loc="left", fontsize=11)
    ax_top.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.92, loc="upper left", ncol=1)

    # Bottom: rolling coverage, model vs baseline
    window = 20
    model_roll_cov = 1 - model["err"].rolling(window, min_periods=8).mean()
    baseline_roll_cov = 1 - baseline["err"].rolling(window, min_periods=8).mean()
    ax_bot.axhline(0.8, color=GRAY_LIGHT, linewidth=1.3, linestyle="--", label="80% target")
    ax_bot.plot(model["date"], model_roll_cov, color=BLUE, linewidth=1.6, label="Sanjha (adaptive conformal)")
    ax_bot.plot(baseline_date, baseline_roll_cov, color=ORANGE, linewidth=1.6, label="Naive baseline (fixed historical spread)")
    ax_bot.set_ylim(0.3, 1.05)
    ax_bot.set_ylabel(f"Rolling {window}-week coverage")
    ax_bot.legend(frameon=False, loc="lower left", ncol=1, fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "coverage_spike.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    results = run()
    paths = [decomposition_chart(results), reliability_diagram(results), coverage_spike_chart(results)]
    for p in paths:
        print("wrote", p)
