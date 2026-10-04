"""
Card builder — "the SMS card is the model". Compiles the forecast state
(app/model.json) plus a local-gap estimate into the numbers on the card and
the fixed-template text that carries them.

    fair price   = F * X * k * exp(-b)             (decomposition.py)
    range        = fair * exp(-/+ q * sigma)        80% band, conformal q
    floor        = fair * exp(-q * sigma * 1.2835)  the band's 5% lower edge
    sigma^2      = futures + FX weekly variance + gap drift q + gap estimate variance P

Prices are rounded OUTWARD to the nearest 10 KES (range) and DOWN (floor), so
rounding never makes the card claim more precision than the model has.
"""
from __future__ import annotations

import math

import numpy as np

from models.decision import DecisionParams, evaluate_choices
from sms.templates import render, render_english

FLOOR_STRETCH = 1.645 / 1.2816  # 90% vs 80% two-sided Gaussian quantile ratio
ROUND_TO = 10


def card_numbers(model: dict, gap: float, gap_var: float) -> dict:
    fair = model["futures_usd_lb"] * model["fx"] * model["k"] * math.exp(-gap)
    sigma = math.sqrt(model["sigma_week_global"] ** 2 + model["kalman"]["q"] + gap_var)
    q = model["conformal_q"]
    lo, hi = fair * math.exp(-q * sigma), fair * math.exp(q * sigma)
    floor = fair * math.exp(-q * sigma * FLOOR_STRETCH)
    return {
        "fair": fair,
        "sigma_week": sigma,
        "lo": int(math.floor(lo / ROUND_TO) * ROUND_TO),
        "hi": int(math.ceil(hi / ROUND_TO) * ROUND_TO),
        "floor": int(math.floor(floor / ROUND_TO) * ROUND_TO),
        "hits": model["track"]["hits"],
        "n": model["track"]["n"],
        "abstain": model["relative_width"] > model["w_star"],
    }


def build_card(model: dict, gap: float, gap_var: float) -> dict:
    nums = card_numbers(model, gap, gap_var)
    name = "abstain" if nums["abstain"] else "card"
    fields = {k: nums[k] for k in ("lo", "hi", "floor", "hits", "n")}
    return {**nums, "template": name, "text": render(name, **fields), "english": render_english(name, **fields)}


def report_reply(model: dict, gap: float, gap_var: float, price: float) -> dict:
    nums = card_numbers(model, gap, gap_var)
    name = "report_abstain" if nums["abstain"] else ("report_low" if price < nums["lo"] else "report_ok")
    fields = {"price": int(round(price)), **{k: nums[k] for k in ("lo", "hi", "hits", "n")}}
    return {**nums, "template": name, "text": render(name, **fields), "english": render_english(name, **fields)}


def wait_reply(model: dict, gap: float, gap_var: float, h_weeks: int = 4, offer: float | None = None) -> dict:
    nums = card_numbers(model, gap, gap_var)
    if nums["abstain"]:
        return {**nums, "template": "abstain", "text": render("abstain"), "english": render_english("abstain")}
    out = evaluate_choices(
        nums["fair"],
        nums["sigma_week"],
        horizons=(h_weeks,),
        params=DecisionParams(**model["decision_params"]),
        offer=offer,
        z_pool=np.asarray(model["z_pool"]),
    )["choices"]
    wait = out[out["choice"] == "wait"].iloc[0]
    fields = {"h": h_weeks, "p": int(round(100 * wait["p_beats_selling_now"])), "bad": int(round(wait["bad_case_p10"]))}
    return {**nums, **fields, "template": "wait", "text": render("wait", **fields), "english": render_english("wait", **fields)}
