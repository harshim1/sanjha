"""
Global price forecast — Sanjha's math section 3.2: "borrow the market's
answer." The central forecast for the futures price at any horizon is just
the current futures price (we don't try to beat the market); the spread
comes from a GARCH(1,1) volatility model on daily log returns, with an
exponentially-weighted (EWMA) fallback if GARCH fails to fit. Horizon-h
variance is sigma_F^2 * h, and the FX rate is treated as a random walk with
its own (much smaller) volatility — as specified in the architecture doc,
with one refinement: for the weekly card, sigma_F is GARCH's forecast
averaged over the coming week (see rolling_futures_sigma), so volatility
mean-reversion within the horizon is respected. Beyond one week the decision
layer still scales flat (sigma_week^2 * h).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def fit_garch_sigma(log_returns: np.ndarray, horizon: int = 1) -> float | None:
    """Fit a GARCH(1,1) on daily log returns and return the forecast daily
    conditional std AVERAGED over the next `horizon` days (sqrt of the mean
    of the 1..h-step variance forecasts), in the same decimal units as the
    input returns. horizon=1 is the plain one-step sigma. Returns None if
    the fit fails (caller should fall back to EWMA).
    """
    try:
        from arch import arch_model

        # arch_model is numerically happier with returns scaled to ~O(1-10);
        # we scale by 100 (percent returns) and rescale sigma back down after.
        scaled = pd.Series(log_returns).dropna() * 100
        if len(scaled) < 50:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            fit = model.fit(disp="off")
        fcast = fit.forecast(horizon=horizon, reindex=False)
        sigma_pct = float(np.sqrt(fcast.variance.values[-1].mean()))
        return sigma_pct / 100.0
    except Exception as exc:  # noqa: BLE001 — any convergence/library issue -> fallback
        warnings.warn(f"[Sanjha volatility] GARCH fit failed ({exc}); falling back to EWMA.")
        return None


def ewma_sigma(log_returns: np.ndarray, lam: float = 0.94) -> float:
    """Exponentially-weighted volatility estimate — the GARCH fallback named
    in the brief. lam=0.94 is the standard RiskMetrics daily decay factor.
    """
    r = pd.Series(log_returns).dropna().values
    if len(r) == 0:
        return 0.0
    var = r[0] ** 2
    for x in r[1:]:
        var = lam * var + (1 - lam) * x**2
    return float(np.sqrt(var))


def futures_sigma(log_returns: np.ndarray, lam: float = 0.94, horizon: int = 1) -> float:
    """GARCH(1,1) sigma, falling back to EWMA on fit failure — section 3.2's
    'volatility ... with an exponentially weighted estimate as a fallback.'
    """
    sigma = fit_garch_sigma(log_returns, horizon=horizon)
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        sigma = ewma_sigma(log_returns, lam=lam)
    return sigma


def fx_sigma(fx_log_returns: np.ndarray) -> float:
    """FX is modeled as a plain random walk, so its volatility is just the
    sample std of log returns (no GARCH — the brief doesn't ask for one here).
    """
    r = pd.Series(fx_log_returns).dropna().values
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1))


def horizon_variance(sigma_one_step, horizon_steps: int):
    """sigma_F^2 * h, as specified. Works on a scalar or an array/Series."""
    result = np.asarray(sigma_one_step, dtype=float) ** 2 * horizon_steps
    return float(result) if result.ndim == 0 else result


def rolling_futures_sigma(
    close: pd.Series,
    dates: pd.Series,
    min_history: int = 250,
    refit_every: int = 5,
    horizon_days: int = 5,
) -> pd.DataFrame:
    """Rolling-origin sigma_F estimate through time, refitting GARCH every
    `refit_every` trading days on an EXPANDING window (only data available up
    to that point — no look-ahead), forward-filled between refits. This is
    what the backtest and the adaptive-conformal calibration run against.

    sigma_f is the daily sigma averaged over the next `horizon_days`, using
    GARCH's own multi-step forecast, so horizon_variance(sigma_f,
    horizon_days) is the model's h-day variance WITH volatility
    mean-reversion: a spike in today's sigma is not assumed to last the
    whole week.
    """
    log_ret = np.log(close).diff()
    n = len(close)
    sigma = np.full(n, np.nan)

    last_fit_idx = -1
    last_sigma = np.nan
    for t in range(min_history, n):
        if t - last_fit_idx >= refit_every or np.isnan(last_sigma):
            last_sigma = futures_sigma(log_ret.values[: t + 1], horizon=horizon_days)
            last_fit_idx = t
        sigma[t] = last_sigma

    return pd.DataFrame({"date": dates.values, "close": close.values, "log_return": log_ret.values, "sigma_f": sigma})


if __name__ == "__main__":
    from data import fetch

    data = fetch.load_all()
    futures = data["futures"]
    roll = rolling_futures_sigma(futures["close"], futures["date"])
    print(roll.dropna().tail(10))
    print("sigma_f summary:\n", roll["sigma_f"].describe())

    fx = data["fx"]
    fx_log_ret = np.log(fx["fx_rate"]).diff()
    print("fx sigma (full sample):", fx_sigma(fx_log_ret))
