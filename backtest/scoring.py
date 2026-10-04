"""
Proper scoring rules — Sanjha's math section 3.7 (Gneiting & Raftery, JASA
2007): pinball loss and CRPS for range accuracy, a Diebold-Mariano test for
whether the improvement over a naive baseline is real, coverage and a
reliability diagram for calibration, and the Brier score for the decision
layer's "waiting pays" probability (models/decision.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def pinball_loss(y_true: np.ndarray, y_pred_quantile: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss at quantile level q in (0, 1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_quantile = np.asarray(y_pred_quantile, dtype=float)
    diff = y_true - y_pred_quantile
    loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
    return float(np.nanmean(loss))


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution N(mu, sigma^2).
    Returns the per-observation CRPS (lower is better); average for a summary.
    """
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where(sigma <= 0, np.nan, sigma)
    z = (y_true - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano test on the loss differential d_t = loss_a - loss_b.
    H0: the two forecasts have equal expected loss. A negative, significant
    statistic means model A (e.g. Sanjha's GARCH+conformal engine) beats B
    (the naive baseline). Uses a Newey-West long-run variance with h-1 lags
    (h = forecast horizon in steps) so the test is valid even if d_t is
    autocorrelated, which one-step-ahead rolling forecasts usually are a
    little.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n, "mean_diff": np.nan}

    d_bar = d.mean()
    # Newey-West variance of the mean, lag truncation = h - 1
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for lag in range(1, max(1, h)):
        if lag >= n:
            break
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var += 2 * (1 - lag / h) * cov
    var = max(var, 1e-12)
    se = np.sqrt(var / n)
    dm_stat = d_bar / se if se > 0 else np.nan
    p_value = 2 * (1 - norm.cdf(abs(dm_stat))) if np.isfinite(dm_stat) else np.nan
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "n": int(n), "mean_diff": float(d_bar)}


def coverage(err: pd.Series) -> float:
    valid = err.dropna()
    return float(1 - valid.mean()) if len(valid) else float("nan")


def reliability_curve(
    run_fn,
    nominal_levels: list[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Build a reliability (calibration) diagram: for each nominal coverage
    level, re-run the forecasting method (via `run_fn(alpha_target)`, which
    should return a DataFrame with an `err` column) and record the empirical
    coverage actually achieved. A well-calibrated method sits on the
    diagonal (nominal == achieved).
    """
    rows = []
    for nominal in nominal_levels:
        alpha_target = 1 - nominal
        df = run_fn(alpha_target)
        achieved = coverage(df["err"])
        rows.append({"nominal_coverage": nominal, "achieved_coverage": achieved})
    return pd.DataFrame(rows)


def risk_coverage_curve(
    width: pd.Series,
    err: pd.Series,
    thresholds: np.ndarray | None = None,
    surprise: pd.Series | None = None,
) -> pd.DataFrame:
    """Section 3.5's abstention rule: for a range of relative-width
    thresholds w*, what share of weeks would the model ANSWER (relative
    width <= w*), and what's the risk on those answered weeks? Pick w*
    where that risk drops to what a farmer could tolerate.

    Two risk measures are reported. `error_rate_on_answered` is the band's
    miss rate — but adaptive conformal holds that near the target at EVERY
    width by construction (and narrow-band weeks, if anything, miss more), so
    abstaining on width cannot buy a lower miss rate. Pass `surprise`
    (|realized - median| / median per week) to also get
    `mean_abs_surprise` / `p90_abs_surprise`: how far, in share of the price,
    the realized price landed from the number on the card. That is the cost
    of a miss a farmer actually bears, it does rise with width, and it is
    the risk models/decision.py::choose_w_star() selects on.
    """
    if thresholds is None:
        thresholds = np.quantile(width.dropna(), np.linspace(0.05, 1.0, 20))
    rows = []
    for w_star in sorted(set(thresholds)):
        answered = width <= w_star
        share_answered = float(answered.mean())
        err_on_answered = coverage(err[answered]) if answered.any() else np.nan
        err_rate_on_answered = 1 - err_on_answered if np.isfinite(err_on_answered) else np.nan
        row = {
            "w_star": float(w_star),
            "share_answered": share_answered,
            "error_rate_on_answered": err_rate_on_answered,
        }
        if surprise is not None:
            s = surprise[answered].dropna()
            row["mean_abs_surprise"] = float(s.mean()) if len(s) else np.nan
            row["p90_abs_surprise"] = float(s.quantile(0.9)) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Brier score for a binary event (section 3.7: the decision layer's
    "waiting pays" probability). Mean squared gap between the stated
    probability and what happened (0/1); lower is better, 0.25 is what a
    constant 50% scores."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    ok = np.isfinite(prob) & np.isfinite(outcome)
    return float(np.mean((prob[ok] - outcome[ok]) ** 2)) if ok.any() else float("nan")


def brier_skill_score(prob: np.ndarray, outcome: np.ndarray, reference_prob: np.ndarray) -> float:
    """1 - Brier(model) / Brier(reference). Positive = beats the reference."""
    ref = brier_score(reference_prob, outcome)
    return float(1 - brier_score(prob, outcome) / ref) if ref > 0 else float("nan")


def probability_reliability(prob: np.ndarray, outcome: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Reliability table for a stated probability: within each equal-count
    bin of stated probabilities, how often did the event actually happen?"""
    df = pd.DataFrame({"prob": prob, "outcome": outcome}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["mean_stated", "observed_rate", "n"])
    df["bin"] = pd.qcut(df["prob"], q=min(bins, df["prob"].nunique()), duplicates="drop")
    out = df.groupby("bin", observed=True).agg(mean_stated=("prob", "mean"), observed_rate=("outcome", "mean"), n=("outcome", "size"))
    return out.reset_index(drop=True)
