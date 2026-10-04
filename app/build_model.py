"""
Compile everything the live system needs into one small JSON file,
app/model.json: the forecast state for the coming week, the track record,
the decision-layer inputs and the intent classifier. The server and the
offline web app read only this file — no pandas, no GARCH, no network at
run time. Re-run weekly (or whenever data is refreshed):

    python -m app.build_model
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from backtest.run_backtest import ALPHA_TARGET, GAMMA, HORIZON_DAYS
from backtest.run_decision_backtest import run as run_decision_backtest
from data import fetch
from data.synthetic import DEFAULT_K
from models.decision import TRADING_DAYS_PER_WEEK, DecisionParams
from models.decomposition import recover_local_gap
from models.kalman import DEFAULT_Q, DEFAULT_R, DEFAULT_TAU2
from models.volatility import futures_sigma, fx_sigma
from sms import classifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")
TRACK_WEEKS = 12
HISTORY_WEEKS = 26


def build() -> dict:
    results = run_decision_backtest()
    data = fetch.load_all()
    futures, fx, model, frame = data["futures"], data["fx"], results["model"], results["frame"]

    # --- the coming week, from the latest data (the backtest frame stops a week short) ---
    sigma_f = futures_sigma(np.log(futures["close"]).diff().dropna().values, horizon=HORIZON_DAYS)
    sigma_x = fx_sigma(np.log(fx["fx_rate"]).diff())
    sigma_week_global = float(np.sqrt(TRADING_DAYS_PER_WEEK * (sigma_f**2 + sigma_x**2)))
    last = model.iloc[-1]
    alpha_next = float(np.clip(last["alpha_t"] + GAMMA * (ALPHA_TARGET - last["err"]), 0.02, 0.6))
    scores = (model["log_actual"] - model["log_median"]).abs() / model["sigma_h"]
    conformal_q = float(np.quantile(scores.dropna(), min(0.995, 1 - alpha_next)))
    sigma_fut_week = sigma_f * np.sqrt(TRADING_DAYS_PER_WEEK)
    relative_width = float(np.exp(conformal_q * sigma_fut_week) - np.exp(-conformal_q * sigma_fut_week))

    track = model["err"].dropna().tail(TRACK_WEEKS)
    region_gap = float(recover_local_gap(data["farmgate_anchors"])["recovered_local_gap"].tail(6).median())
    hist = frame.tail(HISTORY_WEEKS)

    intents = classifier.train()
    intents.pop("_held_out_errors")

    return {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "week_of": f"{pd.Timestamp(futures['date'].iloc[-1]):%Y-%m-%d}",
        "sources": results["sources"],
        "futures_usd_lb": round(float(futures["close"].iloc[-1]), 4),
        "fx": round(float(fx["fx_rate"].iloc[-1]), 3),
        "k": DEFAULT_K,
        "sigma_week_global": round(sigma_week_global, 5),
        "conformal_q": round(conformal_q, 4),
        "relative_width": round(relative_width, 4),
        "w_star": round(float(results["w_star"]["w_star"]), 4),
        "abstention_policy": results["w_star"]["policy"],
        "track": {"hits": int(len(track) - track.sum()), "n": int(len(track))},
        "kalman": {"q": DEFAULT_Q, "r": DEFAULT_R, "tau2": DEFAULT_TAU2, "region_gap0": round(region_gap, 4), "region_var0": DEFAULT_R},
        "decision_params": asdict(DecisionParams()),
        "z_pool": [round(float(z), 2) for z in frame["z_next"].dropna()],
        "history": [
            {"date": f"{pd.Timestamp(d):%Y-%m-%d}", "ref": round(float(c * x * DEFAULT_K), 1)}
            for d, c, x in zip(hist["origin_date"], hist["close_t"], hist["fx_rate"])
        ],
        "classifier": intents,
    }


def load() -> dict:
    with open(MODEL_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    m = build()
    with open(MODEL_PATH, "w") as f:
        json.dump(m, f, separators=(",", ":"))
    size = os.path.getsize(MODEL_PATH) / 1024
    print(f"wrote {MODEL_PATH}: {size:.1f} KB")
    print({k: m[k] for k in ("week_of", "sources", "futures_usd_lb", "fx", "sigma_week_global", "conformal_q", "relative_width", "w_star", "track")})
    print("classifier:", m["classifier"]["scores"])
