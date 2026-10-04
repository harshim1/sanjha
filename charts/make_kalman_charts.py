"""
The two demo artifacts for build-plan hours 8-11 (local-gap Kalman filter +
village pooling, section 3.3 / "Vand Chhako"):
  - the gap-tracking chart: recovered b_hat vs. true local gap per village,
    sparse village included, so a judge can see pooling close the gap on a
    village with almost no farmer reports.
  - the fake-report-flood test: RMSE vs. adversarial-report share, robust
    (Tukey biweight) vs. naive (plain mean), showing the honest breakdown
    point rather than a cherry-picked "robustness always wins" claim.

Reuses the validated categorical palette from charts/make_charts.py (same
dataviz-skill palette, not re-derived) and the same one-axis-per-panel rule.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from charts.make_charts import AQUA, BLUE, GRAY_LIGHT, GRAY_MID, ORANGE, SURFACE
from data.synthetic import simulate_farmer_reports, simulate_region_villages
from models.kalman import run_village_pooling, score_against_truth

OUT_DIR = os.path.dirname(__file__)

N_WEEKS = 150
SPARSE_VILLAGES = ["Serem"]
DEMO_SEED = 100
ADVERSARIAL_FRAC_DEMO = 0.15


def village_gap_tracking_chart() -> str:
    true_gaps = simulate_region_villages(n_weeks=N_WEEKS, seed=DEMO_SEED)
    reports = simulate_farmer_reports(
        true_gaps, sparse_villages=SPARSE_VILLAGES, adversarial_frac=ADVERSARIAL_FRAC_DEMO, seed=200
    )
    pooled = run_village_pooling(reports, robust=True)
    unpooled = run_village_pooling(reports, robust=True, tau2=1e6)

    villages = sorted(true_gaps["village"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, v in zip(axes.flat, villages):
        truth_v = true_gaps[true_gaps.village == v].sort_values("date")
        pooled_v = pooled[pooled.village == v].sort_values("date")
        unpooled_v = unpooled[unpooled.village == v].sort_values("date")
        reports_v = reports[reports.village == v]

        ax.scatter(
            reports_v["date"], reports_v["reported_gap"],
            s=10, color=GRAY_LIGHT, alpha=0.6, zorder=1,
            label="Farmer-reported offer (raw)" if v == villages[0] else None,
        )
        ax.plot(truth_v["date"], truth_v["true_local_gap"], color=AQUA, linewidth=2.0, zorder=3,
                label="True local gap (synthetic ground truth)" if v == villages[0] else None)
        if v in SPARSE_VILLAGES:
            ax.plot(unpooled_v["date"], unpooled_v["b_hat"], color=ORANGE, linewidth=1.3, linestyle="--", zorder=2,
                     label="Kalman, no pooling" if v == villages[0] or v == SPARSE_VILLAGES[0] else None)
        ax.plot(pooled_v["date"], pooled_v["b_hat"], color=BLUE, linewidth=1.8, zorder=4,
                label="Kalman + partial pooling (reported)" if v == villages[0] else None)

        err = score_against_truth(pooled[pooled.village == v], true_gaps[true_gaps.village == v])
        tag = "  (sparse: 0-1 reports/wk)" if v in SPARSE_VILLAGES else f"  ({reports_v.shape[0]} reports)"
        ax.set_title(f"{v}{tag} — pooled RMSE {err['rmse_overall']:.3f}", loc="left", fontsize=10)
        ax.set_ylabel("local gap b$_{v,t}$ (log, dimensionless)")
        ax.set_ylim(-0.05, 0.55)
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.suptitle(
        "Sanjha — tracking the local buyer-margin gap from sparse, noisy farmer reports",
        fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, facecolor=SURFACE,
               edgecolor="none", framealpha=0.95, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    path = os.path.join(OUT_DIR, "village_gap_tracking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def flood_test_chart() -> str:
    true_gaps = simulate_region_villages(n_weeks=N_WEEKS, seed=DEMO_SEED)
    fracs = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    robust_rmse, naive_rmse = [], []
    for frac in fracs:
        reports = simulate_farmer_reports(true_gaps, sparse_villages=SPARSE_VILLAGES, adversarial_frac=float(frac), seed=200)
        robust_rmse.append(score_against_truth(run_village_pooling(reports, robust=True), true_gaps)["rmse_overall"])
        naive_rmse.append(score_against_truth(run_village_pooling(reports, robust=False), true_gaps)["rmse_overall"])
    robust_rmse, naive_rmse = np.array(robust_rmse), np.array(naive_rmse)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvspan(0.5, fracs.max(), color="#f6d9d6", zorder=0,
               label="Past the ~50% breakdown point\n(a village's own reports are\nmajority-fake that period)")
    ax.plot(fracs, naive_rmse, color=ORANGE, linewidth=1.8, marker="o", markersize=4,
            label="Naive (plain mean, no robust weighting)")
    ax.plot(fracs, robust_rmse, color=BLUE, linewidth=2.2, marker="o", markersize=4,
            label="Robust (Tukey biweight, median-seeded)")

    ax.set_xlabel("Share of a village's own farmer reports that are fake/adversarial that period")
    ax.set_ylabel("Region-wide RMSE of recovered local gap vs. ground truth")
    ax.set_title("Fake-report flood test — robust weighting helps until the breakdown point, honestly", loc="left", fontsize=11)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.92, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "flood_test.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    paths = [village_gap_tracking_chart(), flood_test_chart()]
    for p in paths:
        print("wrote", p)
