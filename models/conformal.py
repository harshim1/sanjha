"""
Adaptive conformal inference — Sanjha's math section 3.4.

A fixed-width uncertainty band drifts out of calibration as volatility
regimes shift (exactly what the injected spike in data/synthetic.py is for).
Gibbs & Candes (NeurIPS 2021) fix this by updating the target miss-rate
alpha online:

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where err_t = 1 if last period's realized price fell outside the band the
model gave at the time, and alpha is the long-run target miss rate (e.g.
0.2 for 80% coverage). When the model has been missing too often, alpha_t
shrinks (bands widen); when it's been too conservative, alpha_t grows
(bands tighten) — all without ever assuming the forecast errors are
Gaussian: the band itself is set from the EMPIRICAL quantile of past
studentized residuals, not a z-score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def studentized_residual(log_actual: float, log_median_forecast: float, sigma_h: float) -> float:
    """Scale-free nonconformity score: how many sigma_h's away the realized
    log-price landed from the central forecast. sigma_h must be > 0.
    """
    if sigma_h <= 0 or not np.isfinite(sigma_h):
        return np.nan
    return abs(log_actual - log_median_forecast) / sigma_h


def run_adaptive_conformal(
    dates: pd.Series,
    log_median_forecast: np.ndarray,
    sigma_h: np.ndarray,
    log_actual: np.ndarray,
    alpha_target: float = 0.2,
    gamma: float = 0.05,
    min_calib: int = 15,
    alpha_bounds: tuple[float, float] = (0.02, 0.6),
) -> pd.DataFrame:
    """Run the adaptive-conformal loop step by step through time (no
    look-ahead: the band at time t only uses nonconformity scores observed
    strictly before t).

    Returns a DataFrame with the band (lower/upper, in log-price and price
    space), the online alpha_t, whether that period missed (err_t), and the
    empirical quantile used.
    """
    n = len(dates)
    alpha_t = alpha_target
    scores: list[float] = []  # past nonconformity scores, grows online

    rows = []
    for i in range(n):
        if len(scores) >= min_calib:
            q_level = min(0.995, 1 - alpha_t)
            q = float(np.quantile(scores, q_level))
        else:
            # not enough history yet — fall back to a Gaussian-ish z-score
            # so the card can still show a (wider, honest) range on day one
            from scipy.stats import norm  # local import: only needed for cold start

            q = float(norm.ppf(1 - alpha_t / 2))

        median = log_median_forecast[i]
        sig = sigma_h[i]
        lower_log = median - q * sig
        upper_log = median + q * sig

        actual = log_actual[i]
        err = np.nan
        if np.isfinite(actual):
            err = 1.0 if (actual < lower_log or actual > upper_log) else 0.0
            s = studentized_residual(actual, median, sig)
            if np.isfinite(s):
                scores.append(s)

        rows.append(
            {
                "date": dates.iloc[i] if hasattr(dates, "iloc") else dates[i],
                "alpha_t": alpha_t,
                "q": q,
                "log_median": median,
                "log_lower": lower_log,
                "log_upper": upper_log,
                "median_price": np.exp(median),
                "lower_price": np.exp(lower_log),
                "upper_price": np.exp(upper_log),
                "log_actual": actual,
                "err": err,
            }
        )

        if np.isfinite(err):
            alpha_t = alpha_t + gamma * (alpha_target - err)
            alpha_t = float(np.clip(alpha_t, alpha_bounds[0], alpha_bounds[1]))

    return pd.DataFrame(rows)


def relative_width(df: pd.DataFrame) -> pd.Series:
    """(upper - lower) / median — the abstention rule's input (section 3.5)."""
    return (df["upper_price"] - df["lower_price"]) / df["median_price"]


def coverage_rate(df: pd.DataFrame) -> float:
    valid = df["err"].dropna()
    if len(valid) == 0:
        return float("nan")
    return float(1 - valid.mean())


if __name__ == "__main__":
    # Minimal smoke test: Gaussian-generated data, check coverage lands near target.
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="W"))
    sigma = np.full(n, 0.05)
    sigma[150:200] = 0.15  # inject a regime shift
    median = np.zeros(n)
    actual = median + rng.normal(0, sigma)

    out = run_adaptive_conformal(dates, median, sigma, actual, alpha_target=0.2, gamma=0.05)
    print("overall coverage:", coverage_rate(out), "(target 0.8)")
    print("coverage during regime shift:", coverage_rate(out.iloc[150:200]))
    print("coverage after shift settles:", coverage_rate(out.iloc[200:260]))
