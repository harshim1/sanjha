"""
Local-gap tracking — Sanjha's math section 3.3, "Vand Chhako" (share the
load): track the local buyer-margin gap b_{v,t} CONTINUOUSLY between the
sparse farmgate anchor dates that models/decomposition.py can solve exactly,
using noisy, sparse, and sometimes-adversarial farmer-reported offers.

State-space model (per village v):
    b_{v,t} = b_{v,t-1} + eta_t,      eta_t  ~ N(0, Q)      (random-walk state)
    y_{v,i} = b_{v,t}    + eps_i,     eps_i  ~ N(0, R)      (one farmer report)

Partial pooling across villages in a region ("Vand Chhako" — sharing signal,
not just load):
    b_hat_v = w_v * ybar_v + (1 - w_v) * b_hat_region
    w_v     = n_v / (n_v + R / tau^2)

tau^2 is the between-village variance of the true local gap AFTER the
shared regional trend is accounted for — i.e. the spread of villages'
persistent idiosyncratic offsets, not their raw levels (see
data/synthetic.py::simulate_region_villages's `offset_sd`). R/n_v is the
variance of village v's OWN report-average this period. A village with
few/no reports this week (n_v small) leans on the regional signal; a
village with lots of reports mostly trusts itself. This is the same
empirical-Bayes shrinkage used in small-area estimation — and it is
genuinely sensitive to tau^2 being in the right ballpark: pooling toward a
region signal only helps when villages truly are mostly similar (small
tau^2) with a shared regional driver underneath. An earlier build-session
draft of the synthetic generator gave each village a fully independent
random walk with no shared component; under that generative story pooling
made things WORSE, not better (confirmed by a 25-seed Monte Carlo
validation), because there was no real shared signal to borrow — see that
module's docstring. Partial pooling is only as good as the assumption that
nearby villages' buyer margins actually move together, which is the
realistic case for Sanjha's "Vand Chhako" design (shared trader networks,
shared fuel/transport costs) but is NOT automatically true, and should be
checked against real reported-offer data before trusting it in production.

Robustness to adversarial/fake reports: before pooling and before the Kalman
update, each village's raw reports for a period are passed through a Tukey
biweight (bisquare) M-estimator, iteratively reweighted and seeded from that
PERIOD'S OWN MEDIAN (not the filter's prediction — see robust_period_mean's
docstring for why that distinction matters). Tukey biweight gives FULL,
hard-zero rejection to points beyond its tuning constant, unlike a Huber
estimator's soft 1/|r| taper, which still lets a tightly-clustered, large,
but numerous fake-report flood tug the mean a little every single period —
small per-period nudges that a Kalman filter, integrating over many weeks,
quietly accumulates into real bias. This is a real, documented limit, not
hidden: the defense is a median-type estimator, so it holds up only while a
village's OWN reports that period are under ~50% fake (its breakdown point);
the flood-test sweep in __main__ below shows robust tracking truth well
under roughly 30-40% adversarial share, then degrading toward (and
eventually matching) the naive baseline as the share climbs past 50%.

This module only tracks b_{v,t}; it doesn't decide anything from it yet —
backtest/run_backtest.py::combine_with_local_gap() is the hook where this
estimate replaces the constant local-gap assumption used in session 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_Q = 0.0005  # process variance — matches data/synthetic.py's village-gap generator
DEFAULT_R = 0.05**2  # single-report observation variance (obs_noise_sd=0.05 in the generator)
DEFAULT_TAU2 = 0.02**2  # between-village offset variance (offset_sd=0.02 in the generator)


def huber_weight(standardized_residual: np.ndarray, c: float = 1.5) -> np.ndarray:
    """Huber's bounded-influence weight: 1 inside +/- c standard deviations,
    falling off as c/|r| outside it — SOFT tapering, never reaches zero for
    a finite residual. Kept here for reference/comparison; `robust_period_mean`
    defaults to the harder-rejecting Tukey biweight below, for the reason
    explained in this module's top docstring.
    """
    r = np.asarray(standardized_residual, dtype=float)
    out = np.ones_like(r)
    mask = np.abs(r) > c
    out[mask] = c / np.abs(r[mask])
    return out


def tukey_biweight_weight(standardized_residual: np.ndarray, c: float = 2.0) -> np.ndarray:
    """Tukey's biweight (bisquare) weight: (1-(r/c)^2)^2 inside +/- c standard
    deviations, HARD ZERO outside it — a point far enough from the current
    robust center contributes nothing at all, not just "less." This is what
    gives the adversarial-report defense a real breakdown point instead of a
    soft, ever-present tug on the estimate.
    """
    r = np.asarray(standardized_residual, dtype=float) / c
    out = np.where(np.abs(r) < 1, (1 - r**2) ** 2, 0.0)
    return out


def robust_period_mean(
    reports: np.ndarray,
    center: float,
    obs_sd: float,
    c: float = 2.0,
    max_iter: int = 5,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Iteratively-reweighted mean of one village's reports in one period.

    IMPORTANT design choice: IRLS is seeded from the PERIOD'S OWN MEDIAN, not
    from the filter's current prediction (`center` is used only as a
    fallback when there's too little data to take a median of). Centering on
    the Kalman prediction instead would be circular against a SUSTAINED,
    one-directional attack: once enough biased reports have already pulled
    the filter's estimate, honest reports start looking like the "outliers"
    relative to that already-biased prediction, and the attack reinforces
    itself. The median has a 50% breakdown point — it stays anchored to the
    honest majority as long as adversarial reports are under half the total
    THAT PERIOD, independent of what the filter currently believes. That's
    the real, documented limit of this defense: it degrades once a village's
    own per-period reports are >=50% fake, which the flood-test sweep in
    __main__ below shows directly (robust tracks truth well under ~30-40%
    adversarial share, then degrades toward the naive baseline as the share
    approaches 50%).

    Returns (robust_mean, effective_n), where effective_n is the sum of
    final IRLS weights (a report downweighted to near-zero barely counts
    toward "how much data do we actually have this period"). With zero
    reports, returns (center, 0.0) so the Kalman step falls back to pure
    prediction (effective_n=0 also drives the pooling weight w_v to 0).
    """
    reports = np.asarray(reports, dtype=float)
    n = len(reports)
    if n == 0:
        return float(center), 0.0
    if n == 1:
        return float(reports[0]), 1.0

    estimate = float(np.median(reports))
    weights = np.ones(n)
    for _ in range(max_iter):
        resid = (reports - estimate) / max(obs_sd, 1e-9)
        weights = tukey_biweight_weight(resid, c=c)
        if weights.sum() < 1e-9:
            # every report rejected (extreme, all-contaminated period) —
            # fall back to the median itself rather than divide by zero.
            estimate = float(np.median(reports))
            weights = np.zeros(n)
            weights[np.argmin(np.abs(reports - estimate))] = 1.0
            break
        new_estimate = float(np.average(reports, weights=weights))
        if abs(new_estimate - estimate) < tol:
            estimate = new_estimate
            break
        estimate = new_estimate
    return estimate, float(weights.sum())


def region_estimate(
    village_means: dict[str, float],
    village_eff_n: dict[str, float],
    fallback: float,
) -> float:
    """Precision-weighted mean of villages that reported something this
    period — the architecture doc's literal, SINGLE-PERIOD version of the
    regional ("Vand Chhako") pooled signal. `run_village_pooling` does NOT
    call this: it uses a time-smoothed region Kalman filter instead, because
    a raw one-period cross-section of a handful of villages is itself too
    noisy to pool into a sparse village without doing more harm than good
    (see this module's top docstring and the 25-seed validation in
    __main__). Kept here as a standalone, directly-readable reference
    implementation of the plain formula. Falls back to `fallback` (e.g. the
    prior region estimate) if nobody in the region reported this period.
    """
    names = [v for v in village_means if village_eff_n.get(v, 0) > 0]
    if not names:
        return float(fallback)
    weights = np.array([village_eff_n[v] for v in names], dtype=float)
    values = np.array([village_means[v] for v in names], dtype=float)
    return float(np.average(values, weights=weights))


def partial_pool(
    village_mean: float,
    village_eff_n: float,
    region_mean: float,
    r: float,
    tau2: float,
) -> tuple[float, float]:
    """b_hat_v = w_v*ybar_v + (1-w_v)*b_hat_region, w_v = n_v/(n_v + r/tau^2)
    — the architecture doc's formula, read literally, for one period. See
    `region_estimate`'s docstring: `run_village_pooling` uses a precision-
    weighted generalization of this same idea across TIME (own Kalman
    posterior vs. region Kalman posterior), not this single-period version.
    Returns (pooled_estimate, w_v). w_v=0 when village_eff_n=0 -> pooled
    estimate collapses to the region signal, as it should with no local data.
    """
    if village_eff_n <= 0:
        return float(region_mean), 0.0
    w = village_eff_n / (village_eff_n + r / tau2)
    pooled = w * village_mean + (1 - w) * region_mean
    return float(pooled), float(w)


def kalman_predict(b: float, p: float, q: float) -> tuple[float, float]:
    """Random-walk state prediction: mean unchanged, variance grows by Q."""
    return b, p + q


def kalman_update(b_pred: float, p_pred: float, obs: float, obs_var: float) -> tuple[float, float]:
    """Standard scalar Kalman update. obs_var should reflect how much
    effective information the observation carries this period (less for a
    pooled/shrunk estimate than for a single clean report) — see
    `run_village_pooling` for how that's set.
    """
    if obs_var <= 0 or not np.isfinite(obs_var):
        return b_pred, p_pred
    k = p_pred / (p_pred + obs_var)
    b_new = b_pred + k * (obs - b_pred)
    p_new = (1 - k) * p_pred
    return float(b_new), float(p_new)


def run_village_pooling(
    reports: pd.DataFrame,
    villages: list[str] | None = None,
    q: float = DEFAULT_Q,
    q_region: float | None = None,
    r: float = DEFAULT_R,
    tau2: float = DEFAULT_TAU2,
    b0: float = 0.18,
    p0: float = 0.05**2,
    robust: bool = True,
    robust_c: float = 2.0,
    report_col: str = "reported_gap",
) -> pd.DataFrame:
    """Run the full per-period pipeline for a region of villages, one time
    step at a time, as a TWO-LEVEL hierarchical Kalman filter:

      - a REGION-level filter, fed each period by the robust mean of EVERY
        village's reports pooled together (large effective n -> smooths out
        fast week-to-week noise -> a slowly-moving structural signal: the
        "Vand Chhako" shared regional level).
      - each VILLAGE's own filter, fed only by its own reports (what section
        3.3 calls b_v,t on its own, with no pooling).
      - the REPORTED estimate for village v is a precision-weighted blend of
        its own posterior and the region's posterior (region variance
        inflated by `tau2`, the between-village spread, since a village's
        true level isn't expected to equal the region's exactly):

            w_v          = (1/P_v) / (1/P_v + 1/(P_region + tau2))
            b_hat_v      = w_v * b_v_own + (1 - w_v) * b_region
            P_hat_v      = 1 / (1/P_v + 1/(P_region + tau2))

        This is the time-integrated generalization of the architecture
        doc's w_v = n_v/(n_v + r/tau2): same shape (precision of the
        village's OWN information vs. the region's), but P_v here already
        reflects everything the village's own Kalman filter has accumulated
        over time, not just this one period's report count. A village that
        has built up a long, tight track record (P_v small) stops needing
        the region; a chronically sparse village (P_v stays large forever)
        keeps leaning on it indefinitely — which is the realistic case
        pooling exists for, not just a cold-start fix.

      EARLIER DESIGN NOTE (fixed): an earlier version computed the region
      signal as a single period's raw cross-sectional average of only the
      OTHER villages reporting that week. With few villages and few reports
      each, that estimate was itself noisy, and blending noise into a
      sparse village's filter made its RMSE worse, not better, in testing.
      Smoothing the region signal through its own Kalman filter (first
      bullet above) is what fixes that.

    Within each period, robust (or plain) per-village/-region summaries are
    computed via Tukey biweight IRLS (see robust_period_mean) centered on
    each filter's own prediction. `robust=False` reproduces a naive version
    (plain sample mean, no downweighting) for comparison — the ablation the
    fake-report-flood test (build-plan hours 8-11 demo artifact) runs.

    Returns a long dataframe: date, village, b_hat, p_hat, n_reports,
    effective_n, pool_weight, region_estimate, region_p.
    """
    if villages is None:
        villages = sorted(reports["village"].unique())
    if q_region is None:
        # A region-wide average of ~independent per-village idiosyncratic
        # drift has smaller variance than any one village's own drift —
        # roughly by a factor of the number of villages, under the
        # (documented, simplifying) assumption that village-level process
        # noise is independent across villages.
        q_region = q / max(1, len(villages))

    dates = sorted(reports["date"].unique())
    own_state = {v: b0 for v in villages}
    own_var = {v: p0 for v in villages}
    state = dict(own_state)  # the REPORTED (pooled) estimate carried forward as each village's prior
    var = dict(own_var)
    region_state, region_var = b0, p0

    out_rows = []
    for date in dates:
        period = reports[reports["date"] == date]

        # 1. predict region and each village forward (region from its own
        #    previous posterior; each village from its previous REPORTED/
        #    pooled posterior, so pooling's benefit carries into the next
        #    period's prior rather than being re-derived from scratch).
        region_pred_b, region_pred_p = kalman_predict(region_state, region_var, q_region)
        pred_b, pred_p = {}, {}
        for v in villages:
            pred_b[v], pred_p[v] = kalman_predict(state[v], var[v], q)

        # 2. robust (or plain) per-village summary for this period
        means, eff_n, raw_n = {}, {}, {}
        for v in villages:
            vals = period.loc[period["village"] == v, report_col].to_numpy()
            raw_n[v] = len(vals)
            if robust:
                m, n_eff = robust_period_mean(vals, center=pred_b[v], obs_sd=np.sqrt(r), c=robust_c)
            else:
                m = float(vals.mean()) if len(vals) else pred_b[v]
                n_eff = float(len(vals))
            means[v], eff_n[v] = m, n_eff

        # 3. region observation: ALL villages' reports pooled together this
        #    period, robustly summarized the same way, then Kalman-updated
        #    into the region's own (slow-moving) filter.
        all_vals = period[report_col].to_numpy()
        if robust:
            region_obs, region_eff_n = robust_period_mean(all_vals, center=region_pred_b, obs_sd=np.sqrt(r), c=robust_c)
        else:
            region_obs = float(all_vals.mean()) if len(all_vals) else region_pred_b
            region_eff_n = float(len(all_vals))
        region_obs_var = r / max(region_eff_n, 1e-6)
        if region_eff_n > 0:
            region_state, region_var = kalman_update(region_pred_b, region_pred_p, region_obs, region_obs_var)
        else:
            region_state, region_var = region_pred_b, region_pred_p

        # 4. each village: its own-only Kalman posterior for this period,
        #    then precision-weighted blend with the (now-updated) region.
        for v in villages:
            if eff_n[v] > 0:
                own_b, own_p = kalman_update(pred_b[v], pred_p[v], means[v], r / eff_n[v])
            else:
                own_b, own_p = pred_b[v], pred_p[v]
            own_state[v], own_var[v] = own_b, own_p

            region_var_eff = region_var + tau2
            prec_own = 1.0 / max(own_p, 1e-12)
            prec_region = 1.0 / max(region_var_eff, 1e-12)
            w = prec_own / (prec_own + prec_region)
            b_hat = w * own_b + (1 - w) * region_state
            p_hat = 1.0 / (prec_own + prec_region)
            state[v], var[v] = b_hat, p_hat

            out_rows.append(
                {
                    "date": date,
                    "village": v,
                    "b_hat": b_hat,
                    "p_hat": p_hat,
                    "n_reports": raw_n[v],
                    "effective_n": eff_n[v],
                    "pool_weight": w,
                    "region_estimate": region_state,
                    "region_p": region_var,
                }
            )

    return pd.DataFrame(out_rows).sort_values(["village", "date"]).reset_index(drop=True)


def score_against_truth(result: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """RMSE/bias of b_hat vs. the known true_local_gap, overall and broken
    out by village — the only way to validate a Kalman filter's hidden-state
    recovery is against synthetic ground truth (section 3.3's local gap is
    never directly observed in the real system).
    """
    merged = result.merge(
        truth[["date", "village", "true_local_gap"]], on=["date", "village"], how="left"
    )
    err = merged["b_hat"] - merged["true_local_gap"]
    by_village = (
        merged.assign(err=err)
        .groupby("village")["err"]
        .apply(lambda e: float(np.sqrt((e**2).mean())))
        .to_dict()
    )
    return {
        "rmse_overall": float(np.sqrt((err**2).mean())),
        "bias_overall": float(err.mean()),
        "rmse_by_village": by_village,
        "n": int(len(merged)),
    }


if __name__ == "__main__":
    from data.synthetic import simulate_farmer_reports, simulate_region_villages

    n_weeks = 150
    sparse = ["Serem"]

    # --- Demo 1: one concrete run, same shape as the other modules' smoke tests ---
    true_gaps = simulate_region_villages(n_weeks=n_weeks, seed=100)
    reports = simulate_farmer_reports(true_gaps, sparse_villages=sparse, adversarial_frac=0.15, seed=200)

    print(f"Villages: {sorted(true_gaps['village'].unique())}  (sparse/chronically-under-reporting: {sparse})")
    print(f"Total farmer reports simulated: {len(reports)}  ({reports['is_adversarial'].mean():.1%} adversarial)")
    print()

    robust_result = run_village_pooling(reports, robust=True)
    naive_result = run_village_pooling(reports, robust=False)
    robust_score = score_against_truth(robust_result, true_gaps)
    naive_score = score_against_truth(naive_result, true_gaps)

    print("Full pipeline — robust (Tukey biweight) weighting + partial pooling:")
    print(f"  overall RMSE={robust_score['rmse_overall']:.4f}  bias={robust_score['bias_overall']:+.4f}")
    for v, rmse in robust_score["rmse_by_village"].items():
        print(f"    {v:10s} RMSE={rmse:.4f}")
    print()
    print("Naive (plain mean, no robust weighting) for comparison:")
    print(f"  overall RMSE={naive_score['rmse_overall']:.4f}  bias={naive_score['bias_overall']:+.4f}")
    for v, rmse in naive_score["rmse_by_village"].items():
        print(f"    {v:10s} RMSE={rmse:.4f}")

    # --- Demo 2: fake-report flood sweep (the build-plan's hours 8-11 demo artifact) ---
    print()
    print("=== Fake-report flood test: robust vs naive as adversarial share rises ===")
    print(f"{'frac':>6} {'robust_rmse':>12} {'naive_rmse':>11}")
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        rep = simulate_farmer_reports(true_gaps, sparse_villages=sparse, adversarial_frac=frac, seed=200)
        rb = score_against_truth(run_village_pooling(rep, robust=True), true_gaps)["rmse_overall"]
        nv = score_against_truth(run_village_pooling(rep, robust=False), true_gaps)["rmse_overall"]
        print(f"{frac:>6.1f} {rb:>12.4f} {nv:>11.4f}")
    print(
        "Robust weighting should track truth noticeably better than naive while a village's\n"
        "own per-period reports stay under ~30-40% fake, then degrade toward (and eventually\n"
        "past) the naive baseline as the share nears the ~50% breakdown point of a median-\n"
        "seeded M-estimator — that crossover is the expected, documented limit, not a bug."
    )

    # --- Demo 3: does pooling actually help the chronically sparse village? ---
    # A single seed is not a fair test of a shrinkage estimator (whether pooling helps
    # ANY ONE village depends on how close that village's true offset happens to be to
    # its neighbors' this run) — so this averages over many random regions, which is
    # the metric that actually matters for "should Sanjha pool by default."
    print()
    print("=== Partial pooling benefit, averaged over 25 random regions (no adversarial noise) ===")
    n_runs = 25
    pooled_overall, unpooled_overall, pooled_sparse, unpooled_sparse = [], [], [], []
    for i in range(n_runs):
        seed = 2000 + i
        tg = simulate_region_villages(n_weeks=n_weeks, seed=seed)
        rep = simulate_farmer_reports(
            tg, sparse_villages=sparse, sparse_reports_per_week_range=(0, 1), adversarial_frac=0.0, seed=seed + 9000
        )
        pooled = score_against_truth(run_village_pooling(rep, robust=True), tg)
        unpooled = score_against_truth(run_village_pooling(rep, robust=True, tau2=1e6), tg)
        pooled_overall.append(pooled["rmse_overall"])
        unpooled_overall.append(unpooled["rmse_overall"])
        pooled_sparse.append(pooled["rmse_by_village"][sparse[0]])
        unpooled_sparse.append(unpooled["rmse_by_village"][sparse[0]])
    print(f"  region-wide mean RMSE   pooled={np.mean(pooled_overall):.4f}  unpooled={np.mean(unpooled_overall):.4f}")
    print(f"  {sparse[0]}-only mean RMSE   pooled={np.mean(pooled_sparse):.4f}  unpooled={np.mean(unpooled_sparse):.4f}")
