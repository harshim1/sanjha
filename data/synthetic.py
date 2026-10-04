"""
Synthetic data generator for Sanjha — used ONLY because this sandbox's network
egress allowlist blocks Yahoo Finance / FRED / exchangerate APIs (see fetch.py,
which hits the real endpoints and falls back to this module when they are
unreachable). Swap in real data by running fetch.py somewhere with open network
access (your laptop, the hackathon venue, a CI runner) — nothing downstream
needs to change.

Every series produced here carries a `source` column set to "synthetic" so it
can never silently be mistaken for real data in a chart or a claim to judges.

Design notes:
- Futures price is simulated with a genuine GARCH(1,1) volatility process (not
  just i.i.d. noise), so the GARCH fit in models/volatility.py has real
  clustering to recover. A deliberate "spike" regime is injected over a window
  to stand in for the 2024-25 arabica price spike mentioned in the brief, so
  the adaptive-conformal backtest has a real regime shift to prove itself on.
- Farmgate anchors are generated FROM the known decomposition (futures + FX +
  a true, slowly-drifting local gap b_v,t) plus observation noise. Because the
  true b_v,t is known here, you can score how well the pipeline recovers it —
  a stronger validation than live data alone would give you at this stage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Unit conversion: 1 lb = 0.453592 kg; green coffee -> parchment yield ratio.
# This is the "k" in log P_farm = log F + log X + log k - b. We fold a plausible
# green-to-parchment conversion factor into k; replace with your own sourced
# ratio before presenting (see "Claims to verify" in the Pitch & Judging doc).
DEFAULT_K = 1.35  # placeholder unit/yield conversion constant


def simulate_futures_garch(
    n_days: int = 1000,
    start_price: float = 1.80,  # USD per lb, a plausible arabica level
    omega: float = 1e-6,
    alpha: float = 0.08,
    beta: float = 0.88,
    mu: float = 0.0002,
    spike_start: int | None = None,
    spike_len: int = 90,
    spike_mu_mult: float = 4.0,
    spike_vol_mult: float = 3.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate a daily arabica futures price series via GARCH(1,1) log-returns.

    omega/alpha/beta are the GARCH(1,1) parameters (alpha+beta < 1 for a
    stationary base regime). A spike window multiplies drift and the variance
    innovation to mimic a sharp price run-up like 2024-25, so the backtest has
    a real regime change to detect.
    """
    rng = np.random.default_rng(seed)
    if spike_start is None:
        spike_start = int(n_days * 0.65)

    returns = np.zeros(n_days)  # observed returns (what moves price) — may be regime-amplified
    base_innov = np.zeros(n_days)  # UNAMPLIFIED innovation that drives the GARCH recursion itself,
    # so a spike's amplification affects the price path without feeding back into sigma2 and
    # exploding (alpha * (amplified_return)^2 would otherwise push persistence past 1).
    sigma2 = np.zeros(n_days)
    sigma2[0] = omega / max(1e-9, (1 - alpha - beta))

    for t in range(1, n_days):
        in_spike = spike_start <= t < spike_start + spike_len
        vol_mult = spike_vol_mult if in_spike else 1.0
        drift = mu * spike_mu_mult if in_spike else mu

        sigma2[t] = omega + alpha * (base_innov[t - 1] ** 2) + beta * sigma2[t - 1]
        shock = rng.standard_normal()
        base_innov[t] = np.sqrt(sigma2[t]) * shock
        returns[t] = drift + base_innov[t] * vol_mult

    log_price = np.log(start_price) + np.cumsum(returns)
    price = np.exp(log_price)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(
        {
            "date": dates,
            "close": price,
            "log_return": returns,
            "true_sigma": np.sqrt(sigma2),
            "in_spike_regime": [(spike_start <= t < spike_start + spike_len) for t in range(n_days)],
            "source": "synthetic",
        }
    )
    return df


def simulate_fx(
    n_days: int = 1000,
    start_rate: float = 129.0,  # KES per USD, illustrative
    mu: float = 0.00005,
    sigma: float = 0.003,
    seed: int = 11,
) -> pd.DataFrame:
    """Simulate a daily FX rate as a random walk in log space (geometric)."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=n_days)
    shocks[0] = 0.0
    log_rate = np.log(start_rate) + np.cumsum(shocks)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame({"date": dates, "fx_rate": np.exp(log_rate), "source": "synthetic"})


def simulate_village_local_gap(
    n_weeks: int,
    village: str = "Ondera",
    start_gap: float = 0.18,
    long_run_mean: float = 0.18,
    q: float = 0.0005,  # process variance (how fast the true gap drifts week to week)
    reversion: float = 0.07,  # mean-reversion strength (0 = pure random walk)
    seed: int = 23,
) -> pd.DataFrame:
    """Simulate the TRUE (otherwise unobservable) local gap b_v,t for one
    village as a mean-reverting random walk (a discrete Ornstein-Uhlenbeck
    process), so the Kalman filter's recovery of it can later be scored
    against ground truth.

    A PURE random walk (reversion=0) will, over a few hundred steps, often
    wander down to the clip floor and get stuck there for good — an
    artifact, not a realistic "local gap," and it flattens any chart of it.
    A small reversion term keeps the gap drifting slowly (as the brief
    describes) around a structural level set by transport cost and buyer
    power, without ever permanently pinning at the boundary.
    """
    rng = np.random.default_rng(seed)
    gap = np.empty(n_weeks)
    gap[0] = start_gap
    for t in range(1, n_weeks):
        shock = rng.normal(0, np.sqrt(q))
        gap[t] = gap[t - 1] + reversion * (long_run_mean - gap[t - 1]) + shock
    gap = np.clip(gap, 0.02, 0.6)  # a gap near 0 or above ~60% isn't plausible
    weeks = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_weeks, freq="W")
    return pd.DataFrame({"date": weeks, "village": village, "true_local_gap": gap, "source": "synthetic"})


def simulate_farmgate_anchors(
    futures: pd.DataFrame,
    fx: pd.DataFrame,
    true_gap_weekly: pd.DataFrame,
    k: float = DEFAULT_K,
    obs_noise_sd: float = 0.04,
    monthly: bool = True,
    seed: int = 31,
) -> pd.DataFrame:
    """Generate farmgate anchor prices (stand-in for cooperative payout / NCE
    records) FROM the known decomposition log P = log F + log X + log k - b,
    plus observation noise — so decomposition.py's recovered b_v,t can be
    checked against the `true_local_gap` column carried through here.
    """
    rng = np.random.default_rng(seed)
    df = futures[["date", "close"]].merge(fx[["date", "fx_rate"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    gap_series = (
        true_gap_weekly.set_index("date")["true_local_gap"]
        .reindex(df["date"], method="ffill")
        .bfill()
        .values
    )
    df["true_local_gap"] = gap_series

    log_p_farm = np.log(df["close"]) + np.log(df["fx_rate"]) + np.log(k) - df["true_local_gap"]
    noise = rng.normal(0, obs_noise_sd, size=len(df))
    df["farmgate_price_per_kg"] = np.exp(log_p_farm + noise)
    df["source"] = "synthetic"

    if monthly:
        df = df.set_index("date").resample("MS").mean(numeric_only=True).reset_index()
        df["source"] = "synthetic"
    return df


def simulate_region_villages(
    n_weeks: int,
    village_names: list[str] | None = None,
    shared_long_run_mean: float = 0.18,
    region_q: float = 0.0004,
    region_reversion: float = 0.05,
    offset_sd: float = 0.02,
    offset_q: float = 0.0,
    seed: int = 100,
) -> pd.DataFrame:
    """Simulate TRUE local gaps for several villages in one trading region
    ("Vand Chhako" — the pooling unit), as a SHARED regional signal plus a
    small, persistent per-village offset:

        region_level_t  = mean-reverting walk around shared_long_run_mean
        offset_v         ~ N(0, offset_sd^2), drawn ONCE per village (a
                           fixed structural difference — distance to a
                           depot, one dominant trader's margin — not
                           something that should drift on its own).
                           offset_q defaults to 0 for exactly this reason:
                           even a tiny per-week sd compounds, unreverted,
                           into a large value over many weeks (a bug caught
                           during validation — see git history/README) and
                           silently stops the offset from being "mostly
                           persistent" at all. Set it only deliberately.
        true_local_gap_v,t = region_level_t + offset_v,t

    This is deliberately NOT four independent village-level random walks
    (an earlier version of this generator was exactly that, and it made
    partial pooling look useless-to-harmful in validation: with each
    village wandering on its own, a sparse village's true level can drift
    arbitrarily far from its neighbors', and shrinking toward them then
    hurts it — a real property of shrinkage estimators, but not the
    realistic case the architecture doc's "Vand Chhako" pooling targets).

    The realistic case is villages served by overlapping trader networks,
    buying at the same regional commodity prices and fuel costs: MOST of
    the week-to-week movement in b_v,t is common across nearby villages,
    and the fairly stable part that differs is driven by a few persistent
    local factors (distance to a depot, one dominant trader's margin). That
    is exactly what this generator encodes, and it's what makes the region
    an informative pooling target: offset_sd is the tau^2 in the
    partial-pooling formula b_hat_v = w_v*ybar_v + (1-w_v)*b_hat_region,
    w_v = n_v/(n_v + r/tau^2) — see models/kalman.py.

    Returns a long dataframe: columns date, village, true_local_gap,
    region_level (ground truth for scoring the region estimate), source.
    """
    if village_names is None:
        village_names = ["Ondera", "Kiptoo", "Marwa", "Serem"]
    rng = np.random.default_rng(seed)

    region_level = np.empty(n_weeks)
    region_level[0] = shared_long_run_mean
    for t in range(1, n_weeks):
        shock = rng.normal(0, np.sqrt(region_q))
        region_level[t] = region_level[t - 1] + region_reversion * (shared_long_run_mean - region_level[t - 1]) + shock
    weeks = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_weeks, freq="W")

    frames = []
    for name in village_names:
        offset = np.empty(n_weeks)
        offset[0] = rng.normal(0, offset_sd)
        for t in range(1, n_weeks):
            offset[t] = offset[t - 1] + rng.normal(0, np.sqrt(offset_q))
        gap = np.clip(region_level + offset, 0.02, 0.6)
        frames.append(
            pd.DataFrame(
                {
                    "date": weeks,
                    "village": name,
                    "true_local_gap": gap,
                    "region_level": region_level,
                    "source": "synthetic",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def simulate_farmer_reports(
    true_gap_by_village: pd.DataFrame,
    reports_per_week_range: tuple[int, int] = (3, 10),
    obs_noise_sd: float = 0.05,
    sparse_villages: list[str] | None = None,
    sparse_reports_per_week_range: tuple[int, int] = (0, 2),
    adversarial_frac: float = 0.08,
    adversarial_shift: float = 0.15,
    adversarial_sd: float = 0.03,
    seed: int = 200,
) -> pd.DataFrame:
    """Simulate sparse, noisy, sometimes-adversarial farmer-reported local
    gaps — the raw input to the Kalman filter (section 3.3). A farmer's
    "report" here is already expressed as an implied local gap (i.e. we skip
    re-deriving it from a raw buyer-offer price via the section-3.1
    decomposition, which decomposition.py already covers at anchor dates;
    this module picks up from there — a documented simplification).

    Report volume is random and can be ZERO for a village in a given week
    (the realistic case the partial-pooling/Kalman combo exists to handle).
    `sparse_villages` lets you mark specific villages as chronically
    under-reporting (fewer farmers with phones/literacy/trust), which is
    where pooling toward the region estimate should visibly help most.

    A fraction `adversarial_frac` of reports are fake: shifted by
    `adversarial_shift` (here, a fake report claiming a LARGER gap than
    reality — e.g. a rival trader seeding rumors that "everyone's getting
    lowballed here" to scare farmers into selling to them at a still-bad
    price). The sign/magnitude is a configurable scenario knob, not a
    structural assumption — the robust weighting in models/kalman.py doesn't
    care which direction the attack points, only that it's an outlier
    relative to the current filtered estimate. `is_adversarial` is carried
    through as ground truth for scoring robustness ONLY; the filter never
    sees this column.
    """
    rng = np.random.default_rng(seed)
    sparse_villages = set(sparse_villages or [])
    rows = []
    for _, row in true_gap_by_village.iterrows():
        village = row["village"]
        lo, hi = sparse_reports_per_week_range if village in sparse_villages else reports_per_week_range
        n_reports = rng.integers(lo, hi + 1)
        for _ in range(n_reports):
            is_adversarial = rng.random() < adversarial_frac
            if is_adversarial:
                noise = rng.normal(adversarial_shift, adversarial_sd)
            else:
                noise = rng.normal(0, obs_noise_sd)
            rows.append(
                {
                    "date": row["date"],
                    "village": village,
                    "reported_gap": float(row["true_local_gap"] + noise),
                    "true_local_gap": float(row["true_local_gap"]),  # ground truth, not seen by the filter
                    "is_adversarial": bool(is_adversarial),  # ground truth, not seen by the filter
                }
            )
    out = pd.DataFrame(rows)
    out["source"] = "synthetic"
    return out


def build_all(n_days: int = 1000, village: str = "Ondera", seed: int = 7) -> dict[str, pd.DataFrame]:
    futures = simulate_futures_garch(n_days=n_days, seed=seed)
    fx = simulate_fx(n_days=n_days, seed=seed + 4)
    n_weeks = int(n_days / 5) + 2
    true_gap = simulate_village_local_gap(n_weeks=n_weeks, village=village, seed=seed + 9)
    anchors = simulate_farmgate_anchors(futures, fx, true_gap, seed=seed + 13)
    return {
        "futures": futures,
        "fx": fx,
        "true_local_gap": true_gap,
        "farmgate_anchors": anchors,
    }


if __name__ == "__main__":
    data = build_all()
    for name, df in data.items():
        print(name, df.shape)
        print(df.head(3))
        print()
