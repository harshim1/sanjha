"""
Fill the demo database with SYNTHETIC shared offers (source="synthetic"), so
the dashboard and the village estimates are not empty on first run. Each
goes through exactly the same filter as a real SMS report.

    python -m app.seed            (wipes app/sanjha.db first)
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from app import build_model, store

# farmers sharing per village — Serem stays under the k-anonymity threshold on
# purpose, to show a sparse village borrowing the regional estimate
FARMERS = {"Ondera": 9, "Kiptoo": 7, "Marwa": 6, "Serem": 2}
VILLAGE_OFFSET = {"Ondera": 0.02, "Kiptoo": -0.015, "Marwa": 0.0, "Serem": 0.03}
NOISE_SD = 0.05


def seed(seed_value: int = 5) -> None:
    if os.path.exists(store.DB_PATH):
        os.remove(store.DB_PATH)
    con, model = store.connect(), build_model.load()
    rng = np.random.default_rng(seed_value)
    ref = model["futures_usd_lb"] * model["fx"] * model["k"]
    now = time.time()
    for village, n in FARMERS.items():
        for i in range(n):
            gap = model["kalman"]["region_gap0"] + VILLAGE_OFFSET[village] + rng.normal(0, NOISE_SD)
            price = round(ref * math.exp(-gap) / 5) * 5
            ts = now - rng.uniform(0.5, 9) * 24 * 3600
            store.add_report(con, model, store.hash_phone(f"synthetic:{village}:{i}"), village, price, source="synthetic", now=ts)
    for v in store.VILLAGES:
        print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in store.gap_estimate(con, model, v).items()})


if __name__ == "__main__":
    seed()
