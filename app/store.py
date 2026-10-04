"""
SQLite store + the live local-gap filter.

Privacy: phone numbers are never stored — only a salted SHA-256 hash (set
SANJHA_SALT in production; the default is a development salt). A village's
own estimate is used only once at least K_ANONYMITY distinct farmers have
reported there; until then the village gets the regional estimate.

The gap filter is the scalar Kalman filter from models/kalman.py, run one
report at a time:

    predict   P <- P + q * weeks_since_last_update
    robust    w = tukey_biweight((y - b) / sqrt(P + r));  r_eff = r / w
              (w = 0: the report is stored but does not move the estimate)
    update    K = P / (P + r_eff);  b <- b + K (y - b);  P <- (1 - K) P

where y = log(F * X * k) - log(offer) is the gap that one reported offer
implies. Every report also updates the region-level filter ("_region"),
which is what a village with few reports borrows from.
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import time

from models.kalman import kalman_update, tukey_biweight_weight

DB_PATH = os.environ.get("SANJHA_DB", os.path.join(os.path.dirname(__file__), "sanjha.db"))
SALT = os.environ.get("SANJHA_SALT", "dev-salt-change-me")
K_ANONYMITY = 5
REGION = "_region"
VILLAGES = ["Ondera", "Kiptoo", "Marwa", "Serem"]
DEFAULT_VILLAGE = "Ondera"
ROBUST_C = 3.0  # wider than kalman.py's batch default (2.0): a single honest low offer must still count a little
OFFICER_WEIGHT = 2.0  # reports entered by a cooperative officer for a verified member count double
REPORT_COOLDOWN_S = 24 * 3600  # one counted report per phone per day
WEEK_S = 7 * 24 * 3600

SCHEMA = """
CREATE TABLE IF NOT EXISTS farmers (phone_hash TEXT PRIMARY KEY, village TEXT, opted_out INTEGER DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY, phone_hash TEXT, village TEXT, price REAL, gap_obs REAL,
                                    weight REAL, source TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS gap_state (village TEXT PRIMARY KEY, b REAL, p REAL, updated REAL);
CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, phone_hash TEXT, direction TEXT, channel TEXT, text TEXT,
                                     intent TEXT, ts REAL);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(path or DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def hash_phone(phone: str) -> str:
    return hashlib.sha256((SALT + phone.strip()).encode()).hexdigest()[:20]


def get_farmer(con, phone_hash: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM farmers WHERE phone_hash=?", (phone_hash,)).fetchone()
    if row is None:
        con.execute("INSERT INTO farmers VALUES (?,?,0,?)", (phone_hash, DEFAULT_VILLAGE, time.time()))
        con.commit()
        row = con.execute("SELECT * FROM farmers WHERE phone_hash=?", (phone_hash,)).fetchone()
    return row


def set_farmer(con, phone_hash: str, village: str | None = None, opted_out: bool | None = None) -> None:
    get_farmer(con, phone_hash)
    if village is not None:
        con.execute("UPDATE farmers SET village=? WHERE phone_hash=?", (village, phone_hash))
    if opted_out is not None:
        con.execute("UPDATE farmers SET opted_out=? WHERE phone_hash=?", (int(opted_out), phone_hash))
    con.commit()


def log_message(con, phone_hash: str, direction: str, channel: str, text: str, intent: str = "") -> None:
    con.execute("INSERT INTO messages (phone_hash,direction,channel,text,intent,ts) VALUES (?,?,?,?,?,?)",
                (phone_hash, direction, channel, text, intent, time.time()))
    con.commit()


def _state(con, model: dict, village: str, now: float) -> tuple[float, float]:
    """Predicted (b, P) for `village` at time `now`."""
    row = con.execute("SELECT * FROM gap_state WHERE village=?", (village,)).fetchone()
    k = model["kalman"]
    if row is None:
        return k["region_gap0"], k["region_var0"]
    return row["b"], row["p"] + k["q"] * max(0.0, now - row["updated"]) / WEEK_S


def _filter_step(con, model: dict, village: str, y: float, weight: float, now: float) -> float:
    b, p = _state(con, model, village, now)
    r = model["kalman"]["r"]
    w = float(tukey_biweight_weight((y - b) / math.sqrt(p + r), c=ROBUST_C)) * weight
    if w > 1e-6:
        b, p = kalman_update(b, p, y, r / w)
    con.execute("INSERT INTO gap_state VALUES (?,?,?,?) ON CONFLICT(village) DO UPDATE SET b=excluded.b,p=excluded.p,updated=excluded.updated",
                (village, b, p, now))
    return w


def add_report(con, model: dict, phone_hash: str, village: str, price: float, source: str = "sms", now: float | None = None) -> dict:
    """Store a reported offer and run it through the village and region
    filters. Returns the weight it was given (0 = stored, but ignored)."""
    now = time.time() if now is None else now
    y = math.log(model["futures_usd_lb"] * model["fx"] * model["k"]) - math.log(price)
    recent = con.execute("SELECT 1 FROM reports WHERE phone_hash=? AND ts>? AND weight>0 AND source!='officer'",
                         (phone_hash, now - REPORT_COOLDOWN_S)).fetchone()
    if recent and source != "officer":
        weight = 0.0  # this phone already counted today: a flood from one number moves nothing
    else:
        base = OFFICER_WEIGHT if source == "officer" else 1.0
        weight = _filter_step(con, model, village, y, base, now)
        _filter_step(con, model, REGION, y, base, now)
    con.execute("INSERT INTO reports (phone_hash,village,price,gap_obs,weight,source,ts) VALUES (?,?,?,?,?,?,?)",
                (phone_hash, village, price, y, weight, source, now))
    con.commit()
    return {"gap_obs": y, "weight": weight, "rate_limited": bool(recent and source != "officer")}


def reporters(con, village: str) -> int:
    return con.execute("SELECT COUNT(DISTINCT phone_hash) FROM reports WHERE village=?", (village,)).fetchone()[0]


def gap_estimate(con, model: dict, village: str, now: float | None = None) -> dict:
    """The (gap, variance) that goes on `village`'s card: its own filter
    blended with the region's by precision once K_ANONYMITY farmers have
    reported, the region's alone before that."""
    now = time.time() if now is None else now
    tau2 = model["kalman"]["tau2"]
    b_r, p_r = _state(con, model, REGION, now)
    p_r += tau2  # a village differs from the region by a persistent offset
    n = reporters(con, village)
    if n < K_ANONYMITY:
        return {"gap": b_r, "var": p_r, "basis": "region", "reporters": n}
    b_v, p_v = _state(con, model, village, now)
    prec = 1 / p_v + 1 / p_r
    return {"gap": (b_v / p_v + b_r / p_r) / prec, "var": 1 / prec, "basis": "village+region", "reporters": n}
