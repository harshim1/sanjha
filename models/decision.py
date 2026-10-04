"""
Decision layer — Sanjha's math section 3.6: "sell now, wait h weeks, or sell
half?" Waiting h weeks pays off if

    P_{t+h} * (1 - s_h) - c_h  >  P_t * (1 + r)^h

    s_h  storage loss on parchment over h weeks (small, not zero)
    c_h  cost of storing / transporting, KES per kg over h weeks
    r    Noor's real cost of cash per week (an informal borrowing rate she
         enters once, or a cooperative default)

For each of three choices — sell all now, sell half, wait — this module
reports P(waiting pays) and the 10th-percentile "bad case" outcome, all in
today's money per kg so the three are directly comparable. It never issues a
command: there is deliberately no `recommend()` function here, and nothing
ranks the choices. Noor decides.

Where the distribution of P_{t+h} comes from: the median is today's fair
price (the futures price is the market's own forecast — we don't try to beat
it), and the spread is sigma_week * sqrt(h), with the SHAPE of the shocks
bootstrapped from the model's own past standardized 1-week forecast errors
(symmetrized, so past drift is never extrapolated) rather than assumed Gaussian — the same "no normality assumption" stance as
models/conformal.py. Falls back to Gaussian shocks until enough history
exists.

Abstention (section 3.5) also lives here: `choose_w_star()` picks the
relative-width threshold from a risk-coverage curve, and `evaluate_choices()`
returns no probabilities at all in a week the model abstains.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.kalman import DEFAULT_Q

TRADING_DAYS_PER_WEEK = 5
MIN_POOL = 15  # same cold-start threshold as conformal.py's min_calib
BAD_CASE_QUANTILE = 0.10


@dataclass(frozen=True)
class DecisionParams:
    """Noor's side of the inequality. Defaults are cooperative-level
    placeholders, not measured values — each should be set per cooperative
    (and r ideally per farmer, entered once).
    """

    storage_loss_per_week: float = 0.00125  # ~0.5% of parchment value per month
    cost_per_kg_per_week: float = 0.5  # KES per kg per week of storage/transport
    cash_rate_per_week: float = 0.0113  # ~5% per month informal borrowing rate

    def storage_loss(self, h_weeks: int) -> float:
        return 1 - (1 - self.storage_loss_per_week) ** h_weeks

    def cost(self, h_weeks: int) -> float:
        return self.cost_per_kg_per_week * h_weeks

    def cash_factor(self, h_weeks: int) -> float:
        return (1 + self.cash_rate_per_week) ** h_weeks


def weekly_log_sigma(sigma_f_daily: float, sigma_x_daily: float = 0.0, gap_q: float = DEFAULT_Q) -> float:
    """One-week log-sd of the FARMGATE price: futures and FX daily variances
    scaled to a week (sigma^2 * h, as in volatility.py), plus the local gap's
    weekly random-walk variance q from the Kalman filter. The three are
    treated as independent.
    """
    return float(np.sqrt(TRADING_DAYS_PER_WEEK * (sigma_f_daily**2 + sigma_x_daily**2) + gap_q))


def breakeven_price(price_now: float, h_weeks: int, params: DecisionParams = DecisionParams()) -> float:
    """The price in h weeks at which waiting exactly breaks even with selling
    at `price_now` today — the inequality above solved for P_{t+h}."""
    return (price_now * params.cash_factor(h_weeks) + params.cost(h_weeks)) / (1 - params.storage_loss(h_weeks))


def value_of_waiting(future_price, h_weeks: int, params: DecisionParams = DecisionParams()):
    """What a kg sold in h weeks at `future_price` is worth in today's money:
    net of storage loss and cost, discounted at Noor's cost of cash."""
    net = np.asarray(future_price, dtype=float) * (1 - params.storage_loss(h_weeks)) - params.cost(h_weeks)
    return net / params.cash_factor(h_weeks)


def draw_future_prices(
    fair_price_now: float,
    sigma_week: float,
    h_weeks: int,
    z_pool: np.ndarray | None = None,
    n_draws: int = 4000,
    seed: int = 0,
) -> np.ndarray:
    """Draws of P_{t+h}. Median = today's fair price; log-sd = sigma_week *
    sqrt(h). Each h-week shock is the sum of h resampled 1-week standardized
    shocks from `z_pool` (past forecast errors / their sigma), so the size
    and fat tails of the model's real errors carry through.
    """
    rng = np.random.default_rng(seed)
    pool = None if z_pool is None else np.asarray(z_pool, dtype=float)
    pool = None if pool is None else pool[np.isfinite(pool)]
    if pool is None or len(pool) < MIN_POOL:
        z = rng.standard_normal((n_draws, h_weeks))
    else:
        # symmetrized: keeps the pool's spread and fat tails, drops its drift.
        # A run of past up-weeks must not become a forecast of more up-weeks.
        z = rng.choice(np.concatenate([pool, -pool]), size=(n_draws, h_weeks), replace=True)
    return fair_price_now * np.exp(sigma_week * z.sum(axis=1))


def p_waiting_pays(
    fair_price_now: float,
    sigma_week: float,
    h_weeks: int,
    params: DecisionParams = DecisionParams(),
    offer: float | None = None,
    z_pool: np.ndarray | None = None,
    n_draws: int = 4000,
    seed: int = 0,
) -> float:
    """P(P_{t+h}(1-s_h) - c_h > offer * (1+r)^h). `offer` is the price on the
    table today; it defaults to the fair price."""
    offer = fair_price_now if offer is None else offer
    future = draw_future_prices(fair_price_now, sigma_week, h_weeks, z_pool, n_draws, seed)
    return float(np.mean(future > breakeven_price(offer, h_weeks, params)))


def evaluate_choices(
    fair_price_now: float,
    sigma_week: float,
    horizons: tuple[int, ...] = (2, 4, 8),
    params: DecisionParams = DecisionParams(),
    offer: float | None = None,
    z_pool: np.ndarray | None = None,
    relative_width: float | None = None,
    w_star: float | None = None,
    n_draws: int = 4000,
    seed: int = 0,
) -> dict:
    """The three choices side by side, per horizon, in today's money per kg.

    Returns {"abstain": bool, "choices": DataFrame}. In an abstention week
    (this week's relative band width above w_star) `choices` is empty — the
    card says "I'm not sure, ask the cooperative" instead of showing numbers
    the model doesn't stand behind.
    """
    columns = ["choice", "horizon_weeks", "median_value", "bad_case_p10", "good_case_p90", "p_beats_selling_now"]
    if relative_width is not None and w_star is not None and relative_width > w_star:
        return {"abstain": True, "choices": pd.DataFrame(columns=columns)}

    offer = fair_price_now if offer is None else offer
    rows = [
        {
            "choice": "sell_now",
            "horizon_weeks": 0,
            "median_value": offer,
            "bad_case_p10": offer,
            "good_case_p90": offer,
            "p_beats_selling_now": np.nan,
        }
    ]
    for h in horizons:
        future = draw_future_prices(fair_price_now, sigma_week, h, z_pool, n_draws, seed)
        wait_value = value_of_waiting(future, h, params)
        # same event as "waiting pays": value in today's money beats the offer
        p_pays = float(np.mean(wait_value > offer))
        for choice, value in (("sell_half", 0.5 * offer + 0.5 * wait_value), ("wait", wait_value)):
            rows.append(
                {
                    "choice": choice,
                    "horizon_weeks": h,
                    "median_value": float(np.median(value)),
                    "bad_case_p10": float(np.quantile(value, BAD_CASE_QUANTILE)),
                    "good_case_p90": float(np.quantile(value, 1 - BAD_CASE_QUANTILE)),
                    "p_beats_selling_now": p_pays,
                }
            )
    return {"abstain": False, "choices": pd.DataFrame(rows, columns=columns)}


def choose_w_star(curve: pd.DataFrame, max_risk: float, risk_col: str = "mean_abs_surprise") -> dict:
    """Pick the abstention threshold from a risk-coverage curve
    (backtest/scoring.py): the LARGEST w* — i.e. answer as often as possible —
    whose risk on the answered weeks stays at or below what a farmer could
    tolerate.

    If no threshold meets the tolerance, width is not separating risky weeks
    from safe ones and the curve gives no basis for a cut. The result then
    carries `feasible=False` and w* falls back to the widest band in the
    backtest, so the model abstains only on a week wider than anything it
    has been scored on — a novelty guard, not a risk guarantee.
    """
    ok = curve[curve[risk_col] <= max_risk]
    feasible = not ok.empty
    row = (ok if feasible else curve).loc[lambda d: d["w_star"].idxmax()]
    return {**row.to_dict(), "feasible": feasible, "max_risk": max_risk}


if __name__ == "__main__":
    from backtest.run_decision_backtest import latest_week_inputs

    inputs = latest_week_inputs()
    out = evaluate_choices(
        inputs["fair_price_now"],
        inputs["sigma_week"],
        z_pool=inputs["z_pool"],
        relative_width=inputs["relative_width"],
        w_star=inputs["w_star"],
    )
    print(f"week of {inputs['date']:%Y-%m-%d}: fair price {inputs['fair_price_now']:.0f} KES/kg, "
          f"band width {inputs['relative_width']:.1%} vs w* {inputs['w_star']:.1%}")
    if out["abstain"]:
        print("ABSTAIN — range too wide this week; the card points to the cooperative.")
    else:
        print(out["choices"].round(3).to_string(index=False))
