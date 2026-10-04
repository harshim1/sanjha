"""
The hours 11-13 demo artifacts named in the 24-hour build plan:
  - "sell now, half, or wait" with probabilities (charts/decision_choices.png)
  - the abstention (risk-coverage) curve (charts/risk_coverage.png)

Palette and rcParams come from charts/make_charts.py so all demo charts read
as one set.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from backtest.run_decision_backtest import MAX_MEAN_SURPRISE, latest_week_inputs, run
from charts.make_charts import BLUE, GRAY_LIGHT, GRAY_MID, ORANGE, OUT_DIR, SURFACE
from models.decision import evaluate_choices

CHOICE_LABEL = {"sell_now": "Sell all now", "sell_half": "Sell half, wait {h} wk", "wait": "Wait {h} wk"}


def decision_choices_chart(results: dict) -> str | None:
    inputs = latest_week_inputs(results)
    out = evaluate_choices(
        inputs["fair_price_now"],
        inputs["sigma_week"],
        z_pool=inputs["z_pool"],
        relative_width=inputs["relative_width"],
        w_star=inputs["w_star"],
    )
    if out["abstain"]:
        print("latest week is an abstention week — no choices chart to draw")
        return None
    choices = out["choices"].iloc[::-1].reset_index(drop=True)  # sell_now at the top
    offer = float(out["choices"].loc[out["choices"]["choice"] == "sell_now", "median_value"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axvline(offer, color=GRAY_LIGHT, linewidth=1.3, linestyle="--", zorder=1)
    for y, row in choices.iterrows():
        color = GRAY_MID if row["choice"] == "sell_now" else (ORANGE if row["choice"] == "sell_half" else BLUE)
        ax.plot([row["bad_case_p10"], row["good_case_p90"]], [y, y], color=color, linewidth=6, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot(row["median_value"], y, "o", color=color, markersize=7, zorder=3)
        if row["choice"] != "sell_now":
            ax.annotate(
                f"bad case {row['bad_case_p10']:.0f}",
                (row["bad_case_p10"], y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=GRAY_MID,
            )
            ax.annotate(
                f"beats selling now: {row['p_beats_selling_now']:.0%}",
                (row["good_case_p90"], y), xytext=(10, -3), textcoords="offset points", fontsize=9,
            )
    ax.set_yticks(range(len(choices)))
    ax.set_yticklabels([CHOICE_LABEL[c].format(h=h) for c, h in zip(choices["choice"], choices["horizon_weeks"])])
    ax.set_xlabel("Value per kg in today's money (KES) — bar = bad case (10th pct) to good case (90th pct), dot = median")
    ax.set_xlim(right=ax.get_xlim()[1] + 0.22 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    ax.grid(axis="y", visible=False)
    ax.set_title(
        f"Sell now, half, or wait? Week of {inputs['date']:%d %b %Y}, fair price {offer:.0f} KES/kg",
        loc="left", fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "decision_choices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def risk_coverage_chart(results: dict) -> str:
    curve, chosen = results["risk_coverage"], results["w_star"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(MAX_MEAN_SURPRISE, color=GRAY_LIGHT, linewidth=1.3, linestyle="--", label=f"Tolerance ({MAX_MEAN_SURPRISE:.0%})")
    ax.plot(curve["share_answered"], curve["mean_abs_surprise"], color=BLUE, linewidth=1.8, marker="o", markersize=4, label="Mean surprise on answered weeks")
    ax.plot(curve["share_answered"], curve["p90_abs_surprise"], color=ORANGE, linewidth=1.4, marker="o", markersize=3, label="90th-percentile surprise")
    ax.plot(chosen["share_answered"], chosen["mean_abs_surprise"], "o", markersize=11, markerfacecolor="none", markeredgecolor="#0b0b0b", markeredgewidth=1.4,
            label=f"Novelty guard: abstain only above the widest backtested band ({chosen['w_star']:.1%})")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Share of weeks the card answers (narrowest bands first)")
    ax.set_ylabel("|realized − card price| / card price")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Abstention: narrow bands are only slightly safer, so the card abstains on novelty, not risk", loc="left", fontsize=11)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor="none", loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "risk_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    results = run()
    for p in (decision_choices_chart(results), risk_coverage_chart(results)):
        if p:
            print("wrote", p)
