"""
Price decomposition — Sanjha's math section 3.1:

    log P_farm_{v,t} = log F_t + log X_t + log k - b_{v,t}

Given an observed farmgate anchor price, the futures price, the FX rate and
the unit/yield constant k, we solve for the local gap:

    b_{v,t} = log F_t + log X_t + log k - log P_farm_{v,t}

This is the one quantity Sanjha actually models (the rest — F_t, X_t — are
taken from the market, per the "don't try to beat the coffee market" design
principle). This module only RECOVERS b_v,t from observed data at the anchor
dates; models/kalman.py (next build session) is what tracks it continuously
between anchors using sparse farmer-reported offers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.synthetic import DEFAULT_K


def recover_local_gap(
    anchors: pd.DataFrame,
    k: float = DEFAULT_K,
    price_col: str = "farmgate_price_per_kg",
    futures_col: str = "close",
    fx_col: str = "fx_rate",
) -> pd.DataFrame:
    """Recover b_{v,t} at each anchor date from log P = log F + log X + log k - b.

    `anchors` must already carry the futures price and FX rate for each
    anchor date (data/fetch.py's farmgate_anchors frame does, since it's
    built by resampling the merged futures+FX+price series).
    """
    df = anchors.copy()
    missing = {price_col, futures_col, fx_col} - set(df.columns)
    if missing:
        raise ValueError(f"anchors is missing required columns: {missing}")

    log_p_farm = np.log(df[price_col])
    log_f = np.log(df[futures_col])
    log_x = np.log(df[fx_col])
    df["recovered_local_gap"] = log_f + log_x + np.log(k) - log_p_farm
    return df


def decomposition_error(df: pd.DataFrame) -> dict:
    """If ground truth is available (synthetic runs carry `true_local_gap`),
    report how well the recovery matches it. Real-data runs won't have a
    `true_local_gap` column and this returns an empty dict — recovery from
    real anchors has no ground truth to check against, by definition.
    """
    if "true_local_gap" not in df.columns or "recovered_local_gap" not in df.columns:
        return {}
    err = df["recovered_local_gap"] - df["true_local_gap"]
    return {
        "mean_abs_error": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "n": int(len(df)),
    }


def implied_farmgate_price(
    futures_price: float,
    fx_rate: float,
    local_gap: float,
    k: float = DEFAULT_K,
) -> float:
    """The forward direction: given a futures price, FX rate and a local-gap
    estimate (e.g. from the Kalman filter), what farmgate price in KES/kg
    does that imply? This is what ultimately goes on the SMS card.
    """
    log_p = np.log(futures_price) + np.log(fx_rate) + np.log(k) - local_gap
    return float(np.exp(log_p))


if __name__ == "__main__":
    from data import fetch

    data = fetch.load_all()
    # data["farmgate_anchors"] already carries a monthly-averaged true_local_gap
    # column end to end from synthetic.py (real anchors would carry none).
    recovered = recover_local_gap(data["farmgate_anchors"]).sort_values("date")
    print(recovered[["date", "recovered_local_gap", "true_local_gap"]].head(10))
    print(decomposition_error(recovered))
