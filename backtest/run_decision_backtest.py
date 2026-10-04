"""
Backtest for hours 11-13 of the build plan: the decision layer and the
abstention curve.

Two questions, both answered with no look-ahead:

1. Is P(waiting pays) any good? At every weekly forecast origin, for each
   horizon h, models/decision.py states a probability that waiting h weeks
   beats selling today (after storage loss, cost and Noor's cost of cash).
   h weeks later we know whether it did. Scored with the Brier score against
   a climatology baseline (the share of past, already-resolved weeks where
   waiting paid) and with a reliability table.

2. Where should the model abstain? The risk-coverage curve from
   run_backtest.py, with w* picked by models/decision.py::choose_w_star().

Scope note: the price series here is the farmgate REFERENCE price
F_t * X_t * k * exp(-b) with the local gap b held constant at the median gap
recovered from the anchors, so this backtests the global part of the
forecast (futures + FX). Gap drift (the Kalman filter's q) is therefore left
out of sigma_week here and included in the live card via
decision.weekly_log_sigma(). Horizons overlap across weekly origins, so the
per-origin outcomes are autocorrelated — treat n as an upper bound on the
effective sample size.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from backtest.run_backtest import run as run_forecast_backtest
from backtest.scoring import brier_score, brier_skill_score, probability_reliability
from data import fetch
from data.synthetic import DEFAULT_K
from models.conformal import relative_width
from models.decision import (
    MIN_POOL,
    TRADING_DAYS_PER_WEEK,
    DecisionParams,
    breakeven_price,
    choose_w_star,
    p_waiting_pays,
    weekly_log_sigma,
)
from models.decomposition import recover_local_gap

HORIZONS = (2, 4, 8)
# Abstention policy (decided after the session-3 backtest): a NOVELTY GUARD.
# The risk-coverage curve is too shallow to justify a risk-based cut — width
# finds a calm fifth of weeks but not the dangerous ones — so the card stays
# silent only when this week's band is wider than any band the model has been
# scored on. The tolerance below is set so that every backtested week is
# answered; choose_w_star() then returns the widest backtested band as w*.
# What the card may claim: "wider than anything we have been tested on", not
# "too risky".
MAX_MEAN_SURPRISE = 0.04
NOVELTY_WARMUP_WEEKS = 52


def novelty_guard_history(width: pd.Series, warmup: int = NOVELTY_WARMUP_WEEKS) -> pd.Series:
    """Which past weeks the guard WOULD have abstained on, with no look-ahead:
    the band was wider than every band seen before it (after a warm-up)."""
    prior_max = width.shift(1).expanding(min_periods=warmup).max()
    return (width > prior_max).fillna(False)


def build_decision_frame(results: dict, data: dict) -> pd.DataFrame:
    """Weekly origins with everything the decision layer needs at each one:
    farmgate reference price, FX, weekly sigma, and the signed standardized
    1-week futures shock realized AFTER that origin (for the bootstrap pool —
    only shocks from strictly earlier origins are ever used)."""
    frame = results["frame"].copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"]).astype("datetime64[ns]")

    fx = data["fx"][["date", "fx_rate"]].copy()
    fx["date"] = pd.to_datetime(fx["date"]).astype("datetime64[ns]")
    fx = fx.sort_values("date")
    fx["sigma_x"] = np.log(fx["fx_rate"]).diff().expanding(min_periods=30).std()
    frame = pd.merge_asof(frame.sort_values("origin_date"), fx, left_on="origin_date", right_on="date", direction="backward")
    frame["sigma_x"] = frame["sigma_x"].fillna(0.0)

    gap = float(recover_local_gap(data["farmgate_anchors"])["recovered_local_gap"].median())
    frame["farmgate_ref"] = frame["close_t"] * frame["fx_rate"] * DEFAULT_K * np.exp(-gap)
    frame["sigma_week"] = [weekly_log_sigma(sf, sx, gap_q=0.0) for sf, sx in zip(frame["sigma_f"], frame["sigma_x"])]
    frame["z_next"] = (np.log(frame["close_t_h"]) - np.log(frame["close_t"])) / (
        frame["sigma_f"] * np.sqrt(TRADING_DAYS_PER_WEEK)
    )
    return frame.drop(columns=["date"]).reset_index(drop=True)


def backtest_waiting_pays(frame: pd.DataFrame, params: DecisionParams = DecisionParams(), horizons=HORIZONS) -> pd.DataFrame:
    price = frame["farmgate_ref"].values
    z = frame["z_next"].values
    rows = []
    for i in range(MIN_POOL, len(frame)):
        pool = z[:i]  # shocks realized at or before origin i
        for h in horizons:
            if i + h >= len(frame):
                continue
            prob = p_waiting_pays(price[i], frame["sigma_week"].values[i], h, params, z_pool=pool, seed=i)
            rows.append(
                {
                    "origin_date": frame["origin_date"].values[i],
                    "origin_idx": i,
                    "horizon_weeks": h,
                    "p_waiting_pays": prob,
                    "waiting_paid": float(price[i + h] > breakeven_price(price[i], h, params)),
                }
            )
    out = pd.DataFrame(rows)

    # climatology baseline: share of outcomes at this horizon already known at origin i
    out["p_climatology"] = np.nan
    for h, grp in out.groupby("horizon_weeks"):
        idx = grp["origin_idx"].values
        paid = grp["waiting_paid"].values
        base = [paid[idx + h <= i].mean() if (idx + h <= i).any() else 0.5 for i in idx]
        out.loc[grp.index, "p_climatology"] = base
    return out


def score_waiting_pays(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, grp in list(bt.groupby("horizon_weeks")) + [("all", bt)]:
        rows.append(
            {
                "horizon_weeks": h,
                "n": len(grp),
                "base_rate_waiting_paid": float(grp["waiting_paid"].mean()),
                "mean_stated_prob": float(grp["p_waiting_pays"].mean()),
                "brier_model": brier_score(grp["p_waiting_pays"], grp["waiting_paid"]),
                "brier_climatology": brier_score(grp["p_climatology"], grp["waiting_paid"]),
                "brier_coin_flip": 0.25,
                "skill_vs_climatology": brier_skill_score(grp["p_waiting_pays"], grp["waiting_paid"], grp["p_climatology"]),
            }
        )
    return pd.DataFrame(rows)


def run(cache_dir: str = "backtest/cache", params: DecisionParams = DecisionParams()) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    data = fetch.load_all()
    results = run_forecast_backtest(cache_dir)
    frame = build_decision_frame(results, data)

    bt = backtest_waiting_pays(frame, params)
    scores = score_waiting_pays(bt)
    reliability = probability_reliability(bt["p_waiting_pays"], bt["waiting_paid"])
    chosen = choose_w_star(results["risk_coverage"], MAX_MEAN_SURPRISE)
    model = results["model"]
    guard = novelty_guard_history(relative_width(model))
    surprise = (np.exp(model["log_actual"]) - model["median_price"]).abs() / model["median_price"]
    chosen.update(
        {
            "policy": "novelty_guard",
            "past_abstentions": int(guard.sum()),
            "past_abstention_dates": [f"{d:%Y-%m-%d}" for d in pd.to_datetime(model.loc[guard, "date"])],
            "mean_surprise_on_abstained": float(surprise[guard].mean()) if guard.any() else float("nan"),
        }
    )

    bt.to_csv(os.path.join(cache_dir, "waiting_pays_backtest.csv"), index=False)
    scores.to_csv(os.path.join(cache_dir, "waiting_pays_scores.csv"), index=False)
    reliability.to_csv(os.path.join(cache_dir, "waiting_pays_reliability.csv"), index=False)
    return {
        "frame": frame,
        "model": results["model"],
        "risk_coverage": results["risk_coverage"],
        "w_star": chosen,
        "waiting_pays": bt,
        "scores": scores,
        "reliability": reliability,
        "sources": {name: sorted(df["source"].unique()) for name, df in data.items() if "source" in df.columns},
    }


def latest_week_inputs(results: dict | None = None) -> dict:
    """Everything evaluate_choices() needs for the most recent forecast
    origin — what the card builder (hours 13-17) will call. Here sigma_week
    DOES include the local gap's weekly drift."""
    results = run() if results is None else results
    frame, model = results["frame"], results["model"]
    last = frame.iloc[-1]
    return {
        "date": pd.Timestamp(last["origin_date"]),
        "fair_price_now": float(last["farmgate_ref"]),
        "sigma_week": weekly_log_sigma(last["sigma_f"], last["sigma_x"]),
        "z_pool": frame["z_next"].values,
        "relative_width": float(relative_width(model).iloc[-1]),
        "w_star": float(results["w_star"]["w_star"]),
    }


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 200)
    print("data sources:", out["sources"])
    print("\nP(waiting pays) — Brier scores")
    print(out["scores"].round(4).to_string(index=False))
    print("\nP(waiting pays) — reliability (stated vs. observed)")
    print(out["reliability"].round(3).to_string(index=False))
    print("\nrisk-coverage curve")
    print(out["risk_coverage"].round(4).to_string(index=False))
    print("\nchosen abstention threshold:", {k: round(v, 4) if isinstance(v, float) else v for k, v in out["w_star"].items()})
