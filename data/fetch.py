"""
Data pull layer for Sanjha. Tries the REAL sources first (yfinance for KC=F
arabica futures, FRED for the IMF arabica series and FX, as named in the
architecture doc). If a source is unreachable — as it currently is from this
sandboxed build environment, whose network egress allowlist blocks Yahoo
Finance and FRED — it falls back to data/synthetic.py and prints a loud
warning so synthetic data is never mistaken for real data downstream.

Run this file directly on a machine with open network access (your laptop,
the hackathon venue's wifi, a CI runner) to pull the real series; nothing in
models/ or backtest/ needs to change either way, since both paths return the
same schema.

Set a FRED_API_KEY environment variable to use the FRED fallback for the
monthly IMF arabica series (PCOFFOTMUSDM) and FX. Get a free key at
https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

from . import synthetic

# Twelve years, not three: the last two alone are one long high-volatility
# regime, too little variation to fit or test a volatility model on.
FUTURES_HISTORY = "12y"
SYNTHETIC_FX_LATEST = 129.0  # KES per USD, illustrative
FRED_SERIES_COFFEE = "PCOFFOTMUSDM"  # IMF arabica price, monthly, USD/kg (check units before use)
FRED_SERIES_FX = "DEXKEUS"  # Kenyan shilling per USD, if available; verify series id before relying on it


def _warn_synthetic(what: str) -> None:
    warnings.warn(
        f"[Sanjha data] Could not reach live source for {what} — "
        f"falling back to data/synthetic.py. Results below are SYNTHETIC, "
        f"not real market data. Re-run this script with network access to "
        f"pull real data.",
        stacklevel=2,
    )


def fetch_futures_yfinance(period: str = FUTURES_HISTORY, ticker: str = "KC=F") -> pd.DataFrame | None:
    """Daily arabica futures via yfinance. Returns None (never raises) if the
    network is unreachable, so callers can fall back cleanly."""
    try:
        import yfinance as yf

        raw = yf.download(ticker, period=period, interval="1d", progress=False)
        if raw is None or raw.empty:
            return None
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw.rename(columns={"Date": "date", "Close": "close"})[["date", "close"]].copy()
        # KC=F is quoted in US cents per pound; everything downstream (the
        # decomposition's F_t, DEFAULT_K) expects USD per pound.
        df["close"] = df["close"] / 100.0
        df["source"] = "yfinance"
        return df
    except Exception as exc:  # noqa: BLE001 — any network/library failure should just trigger fallback
        print(f"yfinance fetch failed: {exc}", file=sys.stderr)
        return None


def fetch_fred_series(series_id: str) -> pd.DataFrame | None:
    """A FRED series (monthly IMF arabica price, or FX). Needs FRED_API_KEY."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None
    try:
        import requests

        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            return None
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).rename(columns={"value": series_id})
        df["source"] = "fred"
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"FRED fetch failed for {series_id}: {exc}", file=sys.stderr)
        return None


def get_futures(n_days_fallback: int = 1000) -> pd.DataFrame:
    df = fetch_futures_yfinance()
    if df is not None and len(df) > 30:
        return df
    _warn_synthetic("coffee futures (KC=F)")
    return synthetic.simulate_futures_garch(n_days=n_days_fallback)[["date", "close", "source"]]


def get_futures_with_regime_info(n_days_fallback: int = 1000) -> pd.DataFrame:
    """Like get_futures, but keeps the extra synthetic columns (true_sigma,
    in_spike_regime) when falling back — useful for backtest validation.
    Real data obviously has no such ground-truth columns.
    """
    df = fetch_futures_yfinance()
    if df is not None and len(df) > 30:
        return df
    _warn_synthetic("coffee futures (KC=F)")
    return synthetic.simulate_futures_garch(n_days=n_days_fallback)


def get_fx(n_days_fallback: int = 1000) -> pd.DataFrame:
    fx = fetch_fred_series(FRED_SERIES_FX)
    if fx is not None and len(fx) > 30:
        return fx.rename(columns={FRED_SERIES_FX: "fx_rate"})[["date", "fx_rate", "source"]]
    _warn_synthetic("FX rate (KES/USD)")
    fx = synthetic.simulate_fx(n_days=n_days_fallback)
    # pin the LATEST synthetic rate to a realistic level, so a long simulated
    # history can't drift today's card price somewhere implausible
    fx["fx_rate"] *= SYNTHETIC_FX_LATEST / fx["fx_rate"].iloc[-1]
    return fx


def get_farmgate_anchors_synthetic_only(futures: pd.DataFrame, fx: pd.DataFrame, village: str = "Ondera") -> dict:
    """There is no public API for farmgate/cooperative payout prices — per the
    architecture doc, this is the gap Sanjha fills. For now we always generate
    these from the known decomposition so decomposition.py has ground truth
    to validate against. Swap in archived UCDA / Nairobi Coffee Exchange
    reports here once you've transcribed them (see the brief's data sources).
    """
    n_weeks = int(len(futures) / 5) + 2
    true_gap = synthetic.simulate_village_local_gap(n_weeks=n_weeks, village=village)
    anchors = synthetic.simulate_farmgate_anchors(futures, fx, true_gap)
    return {"anchors": anchors, "true_local_gap": true_gap}


def load_all(n_days: int = 1000, village: str = "Ondera", cache_dir: str = "data/cache") -> dict[str, pd.DataFrame]:
    """One call that returns everything the rest of the pipeline needs, and
    caches each piece to CSV so repeated runs don't refetch."""
    os.makedirs(cache_dir, exist_ok=True)

    futures = get_futures_with_regime_info(n_days_fallback=n_days)
    fx = get_fx(n_days_fallback=max(n_days, len(futures) + 60))  # synthetic FX must span the futures history
    gate = get_farmgate_anchors_synthetic_only(futures, fx, village=village)

    data = {
        "futures": futures,
        "fx": fx,
        "farmgate_anchors": gate["anchors"],
        "true_local_gap": gate["true_local_gap"],
    }
    for name, df in data.items():
        df.to_csv(os.path.join(cache_dir, f"{name}.csv"), index=False)
    return data


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        synth = (df.get("source") == "synthetic").any() if "source" in df.columns else "?"
        print(f"{name}: {df.shape}, synthetic={synth}")
