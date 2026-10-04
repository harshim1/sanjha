"""
Every word Sanjha ever sends. Fixed templates only — no text is generated at
run time, so the system cannot hallucinate a price or an instruction. None of
these says "sell" or "hold"; the strongest thing a message does is point to
the cooperative.

NOT YET CHECKED BY A NATIVE SWAHILI SPEAKER. The first two are from the build
reference; the rest were drafted for this build and must be reviewed before
any farmer sees them.
"""
from __future__ import annotations

SMS_LIMIT = 160

TEMPLATES = {
    # (Fair price this week: lo-hi/kg. Below floor? Ask the cooperative first. Right hits of the last n weeks.)
    "card": "BEI SAWA wiki hii: {lo}-{hi}/kg. Chini ya {floor}? Uliza ushirika kwanza. Sahihi wiki {hits} kati ya {n}.",
    # (Prices are swinging a lot this week. I'm not sure - ask the cooperative.)
    "abstain": "Bei inayumba sana wiki hii. Sina uhakika - uliza ushirika.",
    # (Thanks for sharing price/kg. Fair price this week: lo-hi/kg. Right hits of the last n weeks.)
    "report_ok": "Asante kwa kushiriki {price}/kg. Bei sawa wiki hii: {lo}-{hi}/kg. Sahihi wiki {hits} kati ya {n}.",
    # (Thanks for sharing. price/kg is below the fair price (lo-hi/kg). Ask the cooperative first.)
    "report_low": "Asante kwa kushiriki. {price}/kg iko chini ya bei sawa ({lo}-{hi}/kg). Uliza ushirika kwanza.",
    # (Thanks for sharing price/kg. Prices are swinging a lot this week - ask the cooperative.)
    "report_abstain": "Asante kwa kushiriki {price}/kg. Bei inayumba sana wiki hii - uliza ushirika.",
    # (I did not see a price. Send the price you were offered, e.g.: wamenipa 340)
    "report_no_price": "Sijaona bei. Tuma bei uliyopewa kwa kilo, mfano: wamenipa 340",
    # (Waiting h weeks: p% chance of getting more than today. Bad case: bad/kg. The decision is yours.)
    "wait": "Kusubiri wiki {h}: nafasi {p}% ya kupata zaidi ya leo. Hali mbaya: {bad}/kg. Uamuzi ni wako.",
    # (Send BEI for the fair price. Send the price you were offered, e.g. wamenipa 340. Send ACHA to leave.)
    "help": "SANJHA: Tuma BEI kupata bei sawa. Tuma bei uliyopewa, mfano: wamenipa 340. Tuma ACHA kujiondoa.",
    # (You have left. You will get no more messages. Send JIUNGE to return.)
    "opt_out": "Umejiondoa. Hutapokea ujumbe tena. Tuma JIUNGE kurudi.",
    # (Welcome to Sanjha, village. Send BEI for the fair price.)
    "join": "Karibu Sanjha, {village}. Tuma BEI kupata bei sawa ya wiki hii.",
    # (Sorry, I did not understand. Send BEI for the price, or MSAADA.)
    "unknown": "Samahani, sijaelewa. Tuma BEI kupata bei, au MSAADA.",
}

ENGLISH = {
    "card": "Fair price this week: {lo}-{hi}/kg. Below {floor}? Ask the cooperative first. Right {hits} of the last {n} weeks.",
    "abstain": "Prices are swinging a lot this week. I'm not sure - ask the cooperative.",
    "report_ok": "Thanks for sharing {price}/kg. Fair price this week: {lo}-{hi}/kg. Right {hits} of the last {n} weeks.",
    "report_low": "Thanks for sharing. {price}/kg is below the fair price ({lo}-{hi}/kg). Ask the cooperative first.",
    "report_abstain": "Thanks for sharing {price}/kg. Prices are swinging a lot this week - ask the cooperative.",
    "report_no_price": "I did not see a price. Send the price per kg you were offered, e.g.: wamenipa 340",
    "wait": "Waiting {h} weeks: {p}% chance of getting more than today. Bad case: {bad}/kg. The decision is yours.",
    "help": "SANJHA: Send BEI for the fair price. Send the price you were offered, e.g. wamenipa 340. Send ACHA to leave.",
    "opt_out": "You have left. You will get no more messages. Send JIUNGE to return.",
    "join": "Welcome to Sanjha, {village}. Send BEI for this week's fair price.",
    "unknown": "Sorry, I did not understand. Send BEI for the price, or MSAADA.",
}


def render(name: str, **fields) -> str:
    text = TEMPLATES[name].format(**fields)
    if len(text) > SMS_LIMIT:
        raise ValueError(f"template {name!r} renders to {len(text)} chars, over the {SMS_LIMIT}-char SMS limit")
    return text


def render_english(name: str, **fields) -> str:
    return ENGLISH[name].format(**fields)
