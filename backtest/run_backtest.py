"""
Full backtest orchestration for hours 3-8 of the build plan: GARCH model,
adaptive conformal, backtest and scoring. Produces the two demo artifacts
named there — a reliability diagram, and coverage through the injected
2024-25-style spike regime — plus the pinball/CRPS/Diebold-Mariano
comparison against a naive baseline.

Scope note: this backtests the GLOBAL price forecast (futures, "borrow the
market's answer", calibrated with GARCH + adaptive conformal) at a weekly
cadence. It deliberately does NOT yet fold in the local gap b_v,t — that's
tracked by the Kalman filter + village pooling, which is hours 8-11 of the
build plan and hasn't been built in this session. Combining the two (global
forecast x local-gap estimate -> a farmgate-price band) is a short, mechanical
extra step once that filter exists; see combine_with_local_gap() below for
the one-shot version used by decomposition.py today (constant/latest gap).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from backtest.scoring import coverage, crps_gaussian, diebold_mariano, pinball_loss, reliability_curve, risk_coverage_curve
from data import fetch
from models.conformal import relative_width, run_adaptive_conformal
from models.volatility import fx_sigma, horizon_variance, rolling_futures_sigma

HORIZON_DAYS = 5  # ~1 week ahead, matching the SMS card's weekly cadence
STEP_DAYS = 5  # forecast origins spaced a week apart (non-overlapping)
ALPHA_TARGET = 0.2  # 80% coverage target
# Adaptive-conformal learning rate. 0.05 made the quantile swing between
# ~0.9 and ~2.1 sigma in reaction to each miss, which drowned the volatility
# signal in the band width (narrowest-fifth bands missed 37% of the time);
# 0.01 keeps overall coverage at target with a steadier quantile.
GAMMA = 0.01
# Real data carries no ground-truth regime flag, so the "spike" is the
# 2024-25 arabica run-up by date (synthetic data flags its own injected one).
REAL_SPIKE_START, REAL_SPIKE_END = pd.Timestamp("2024-11-01"), pd.Timestamp("2025-04-30")


def build_weekly_forecast_frame(futures: pd.DataFrame, min_history: int = 250) -> pd.DataFrame:
    """Daily GARCH sigma, resampled to weekly forecast origins with the
    h-day-ahead realized price attached (no look-ahead: sigma at t only used
    data up to t; the realized price is strictly in the future of t).
    """
    roll = rolling_futures_sigma(futures["close"], futures["date"], min_history=min_history, horizon_days=HORIZON_DAYS)
    roll = roll.dropna(subset=["sigma_f"]).reset_index(drop=True)

    origins = roll.iloc[::STEP_DAYS].copy()
    origins = origins[origins.index + HORIZON_DAYS < len(roll)]

    # keyed by date, not position: roll's index was reset after dropping the
    # first min_history rows, so it no longer lines up with futures' rows
    spike_by_date = (
        dict(zip(futures["date"].values, futures["in_spike_regime"].values))
        if "in_spike_regime" in futures.columns
        else None
    )

    rows = []
    close_vals = roll["close"].values
    date_vals = roll["date"].values
    for idx in origins.index:
        if idx + HORIZON_DAYS >= len(roll):
            continue
        rows.append(
            {
                "origin_date": date_vals[idx],
                "target_date": date_vals[idx + HORIZON_DAYS],
                "close_t": close_vals[idx],
                "close_t_h": close_vals[idx + HORIZON_DAYS],
                "sigma_f": roll["sigma_f"].values[idx],
                "in_spike_regime": spike_by_date[date_vals[idx]]
                if spike_by_date is not None
                else REAL_SPIKE_START <= pd.Timestamp(date_vals[idx]) <= REAL_SPIKE_END,
            }
        )
    return pd.DataFrame(rows)


def naive_baseline_frame(frame: pd.DataFrame, window_weeks: int = 52) -> pd.DataFrame:
    """'Last price +/- historical spread': a trailing, non-adaptive rolling
    std of weekly log returns (no GARCH clustering, no online conformal
    recalibration), with a fixed Gaussian quantile at the same target alpha.
    """
    from scipy.stats import norm

    log_ret = np.log(frame["close_t"]).diff()
    hist_sigma = log_ret.rolling(window_weeks, min_periods=10).std()
    hist_sigma = hist_sigma.bfill()
    z = norm.ppf(1 - ALPHA_TARGET / 2)

    median_log = np.log(frame["close_t"])
    sigma_h = hist_sigma * np.sqrt(HORIZON_DAYS)  # scaling weekly-step sigma up to the h-day horizon
    lower_log = median_log - z * sigma_h
    upper_log = median_log + z * sigma_h
    actual_log = np.log(frame["close_t_h"])
    err = ((actual_log < lower_log) | (actual_log > upper_log)).astype(float)

    return pd.DataFrame(
        {
            "date": frame["target_date"],
            "log_median": median_log,
            "sigma_h": sigma_h,
            "log_lower": lower_log,
            "log_upper": upper_log,
            "lower_price": np.exp(lower_log),
            "upper_price": np.exp(upper_log),
            "median_price": frame["close_t"],
            "log_actual": actual_log,
            "err": err,
        }
    )


def model_frame(frame: pd.DataFrame, alpha_target: float = ALPHA_TARGET, gamma: float = GAMMA) -> pd.DataFrame:
    median_log = np.log(frame["close_t"]).values
    sigma_h = np.sqrt(horizon_variance(frame["sigma_f"].values, HORIZON_DAYS))
    actual_log = np.log(frame["close_t_h"]).values
    out = run_adaptive_conformal(
        frame["target_date"], median_log, sigma_h, actual_log, alpha_target=alpha_target, gamma=gamma
    )
    out["sigma_h"] = sigma_h
    out["in_spike_regime"] = frame["in_spike_regime"].values
    return out


def combine_with_local_gap(futures_price: float, fx_rate: float, local_gap_hat: float, k: float) -> float:
    """One-shot combination of a futures-price forecast with FX and a
    (currently constant / latest-recovered, not yet Kalman-tracked) local-gap
    estimate, giving a farmgate price per kg — the forward direction of
    decomposition.py's implied_farmgate_price(). This is the full layer-2
    pipeline from the architecture doc; wiring the Kalman filter's live
    b_v,t estimate in here (instead of a constant) is the next build
    session's job (hours 8-11)."""
    log_p = np.log(futures_price) + np.log(fx_rate) + np.log(k) - local_gap_hat
    return float(np.exp(log_p))


def run(cache_dir: str = "backtest/cache") -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    data = fetch.load_all()
    futures = data["futures"]

    frame = build_weekly_forecast_frame(futures)
    model = model_frame(frame)
    baseline = naive_baseline_frame(frame)

    model.to_csv(os.path.join(cache_dir, "model_forecast.csv"), index=False)
    baseline.to_csv(os.path.join(cache_dir, "baseline_forecast.csv"), index=False)

    # --- Pinball loss at the two tail quantiles implied by the 80% interval ---
    q_lo, q_hi = ALPHA_TARGET / 2, 1 - ALPHA_TARGET / 2
    model_pinball = {
        "lower": pinball_loss(np.exp(model["log_actual"]), model["lower_price"], q_lo),
        "upper": pinball_loss(np.exp(model["log_actual"]), model["upper_price"], q_hi),
    }
    baseline_pinball = {
        "lower": pinball_loss(np.exp(baseline["log_actual"]), baseline["lower_price"], q_lo),
        "upper": pinball_loss(np.exp(baseline["log_actual"]), baseline["upper_price"], q_hi),
    }

    # --- CRPS (Gaussian form, using each method's own sigma_h) ---
    model_crps = crps_gaussian(np.exp(model["log_actual"]), model["median_price"], model["sigma_h"] * model["median_price"])
    baseline_crps = crps_gaussian(
        np.exp(baseline["log_actual"]), baseline["median_price"], baseline["sigma_h"] * baseline["median_price"]
    )
    dm = diebold_mariano(model_crps, baseline_crps, h=HORIZON_DAYS)

    # --- Coverage, overall and during the injected spike ---
    overall_coverage_model = coverage(model["err"])
    overall_coverage_baseline = coverage(baseline["err"])
    spike_mask = model["in_spike_regime"].astype(bool)
    spike_coverage_model = coverage(model.loc[spike_mask, "err"]) if spike_mask.any() else float("nan")
    spike_coverage_baseline = coverage(baseline.loc[spike_mask, "err"]) if spike_mask.any() else float("nan")

    # --- Reliability diagram (model only — the adaptive method's whole point) ---
    def _run_at(alpha_target: float) -> pd.DataFrame:
        return model_frame(frame, alpha_target=alpha_target)

    reliability = reliability_curve(_run_at)
    reliability.to_csv(os.path.join(cache_dir, "reliability.csv"), index=False)

    # --- Risk-coverage curve for the abstention threshold (section 3.5) ---
    width = relative_width(model)
    surprise = (np.exp(model["log_actual"]) - model["median_price"]).abs() / model["median_price"]
    rc_curve = risk_coverage_curve(width, model["err"], surprise=surprise)
    rc_curve.to_csv(os.path.join(cache_dir, "risk_coverage.csv"), index=False)

    summary = {
        "n_weeks": len(model),
        "coverage_overall": {"model": overall_coverage_model, "baseline": overall_coverage_baseline, "target": 1 - ALPHA_TARGET},
        "coverage_during_spike": {"model": spike_coverage_model, "baseline": spike_coverage_baseline},
        "avg_relative_width": {"model": float(width.mean()), "baseline": float(((baseline["upper_price"] - baseline["lower_price"]) / baseline["median_price"]).mean())},
        "pinball_loss": {"model": model_pinball, "baseline": baseline_pinball},
        "crps_mean": {"model": float(np.nanmean(model_crps)), "baseline": float(np.nanmean(baseline_crps))},
        "diebold_mariano": dm,
    }
    return {
        "summary": summary,
        "model": model,
        "baseline": baseline,
        "reliability": reliability,
        "risk_coverage": rc_curve,
        "frame": frame,
    }


if __name__ == "__main__":
    import json

    results = run()
    print(json.dumps(results["summary"], indent=2))
