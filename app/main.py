"""
Sanjha's live loop: SMS and USSD callbacks in the shape Africa's Talking
sends them, the card/decision API the offline web app caches, and a
cooperative dashboard. Run:

    uvicorn app.main:app --port 8000        (after `python -m app.build_model`)

No LLM and no model fitting happen here — every reply is a fixed template
filled with numbers from app/model.json and the SQLite gap filter.

Africa's Talking: point the sandbox's SMS callback at POST /sms/incoming and
the USSD callback at POST /ussd. Replies to SMS are sent through the AT
messaging API only if AT_USERNAME and AT_API_KEY are set; otherwise they are
just logged and returned in the HTTP response (which is what the built-in
simulator at /#sim uses).
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Form, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app import build_model, store
from sms import card, classifier
from sms.templates import render, render_english

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
WAIT_HORIZONS = {"1": 2, "2": 4, "3": 8}

app = FastAPI(title="Sanjha")
MODEL = build_model.load()
CON = store.connect()


def _reply(name: str, **fields) -> dict:
    return {"template": name, "text": render(name, **fields), "english": render_english(name, **fields)}


def _send_sms(phone: str, text: str) -> str:
    """Deliver via Africa's Talking if credentials are configured."""
    user, key = os.environ.get("AT_USERNAME"), os.environ.get("AT_API_KEY")
    if not (user and key):
        return "logged-only"
    import httpx

    host = "api.sandbox.africastalking.com" if user == "sandbox" else "api.africastalking.com"
    try:
        r = httpx.post(f"https://{host}/version1/messaging", data={"username": user, "to": phone, "message": text},
                       headers={"apiKey": key, "Accept": "application/json"}, timeout=10)
        return f"africastalking:{r.status_code}"
    except httpx.HTTPError as exc:
        return f"africastalking-failed:{type(exc).__name__}"


def handle_text(phone: str, text: str, channel: str = "sms") -> dict:
    """One incoming message -> one fixed-template reply."""
    ph = store.hash_phone(phone)
    farmer = store.get_farmer(CON, ph)
    first = text.strip().split(" ")[0].lower() if text.strip() else ""

    if first in ("jiunge", "join"):
        rest = text.strip().split(" ", 1)[1].strip().title() if " " in text.strip() else ""
        village = rest if rest in store.VILLAGES else farmer["village"]
        store.set_farmer(CON, ph, village=village, opted_out=False)
        intent, conf, out = "join", 1.0, _reply("join", village=village)
    else:
        intent, conf = classifier.predict(MODEL["classifier"], text)
        village = farmer["village"]
        if intent == "opt_out":
            store.set_farmer(CON, ph, opted_out=True)
            out = _reply("opt_out")
        elif intent == "help":
            out = _reply("help")
        elif intent == "report_offer":
            price = classifier.extract_price(text)
            if price is None:
                out = _reply("report_no_price")
            else:
                info = store.add_report(CON, MODEL, ph, village, price, source=channel)
                g = store.gap_estimate(CON, MODEL, village)
                out = {**card.report_reply(MODEL, g["gap"], g["var"], price), "report": info, "basis": g["basis"]}
        elif intent in ("ask_price", "ask_wait"):
            g = store.gap_estimate(CON, MODEL, village)
            build = card.build_card if intent == "ask_price" else card.wait_reply
            out = {**build(MODEL, g["gap"], g["var"]), "basis": g["basis"]}
        else:
            out = _reply("unknown")

    store.log_message(CON, ph, "in", channel, text, intent)
    store.log_message(CON, ph, "out", channel, out["text"], out["template"])
    return {"intent": intent, "confidence": round(conf, 3), "village": village, "reply": out["text"],
            "reply_english": out["english"], "template": out["template"], "detail": {k: v for k, v in out.items() if k in ("lo", "hi", "floor", "fair", "basis", "report", "abstain")}}


@app.post("/sms/incoming")
def sms_incoming(from_: str = Form(..., alias="from"), text: str = Form("")):
    result = handle_text(from_, text, "sms")
    result["delivery"] = _send_sms(from_, result["reply"])
    return result


@app.post("/ussd", response_class=PlainTextResponse)
def ussd(phoneNumber: str = Form(...), text: str = Form("")):
    """Africa's Talking USSD: `text` is the whole input so far, '*'-separated.
    'CON ' keeps the session open, 'END ' closes it."""
    steps = [s for s in text.split("*") if s != ""]
    if not steps:
        return "CON Sanjha\n1. Bei sawa wiki hii\n2. Ripoti bei uliyopewa\n3. Nisubiri au niuze?\n4. Msaada"
    if steps[0] == "1":
        return "END " + handle_text(phoneNumber, "bei", "ussd")["reply"]
    if steps[0] == "2":
        if len(steps) == 1:
            return "CON Andika bei uliyopewa kwa kilo:"
        return "END " + handle_text(phoneNumber, f"wamenipa {steps[1]}", "ussd")["reply"]
    if steps[0] == "3":
        if len(steps) == 1:
            return "CON Kusubiri wiki ngapi?\n1. Wiki 2\n2. Wiki 4\n3. Wiki 8"
        ph = store.hash_phone(phoneNumber)
        g = store.gap_estimate(CON, MODEL, store.get_farmer(CON, ph)["village"])
        out = card.wait_reply(MODEL, g["gap"], g["var"], WAIT_HORIZONS.get(steps[1], 4))
        store.log_message(CON, ph, "out", "ussd", out["text"], out["template"])
        return "END " + out["text"]
    if steps[0] == "4":
        return "END " + render("help")
    return "END " + render("unknown")


@app.get("/api/card")
def api_card(village: str = store.DEFAULT_VILLAGE):
    if village not in store.VILLAGES:
        raise HTTPException(404, "unknown village")
    g = store.gap_estimate(CON, MODEL, village)
    return {"village": village, "week_of": MODEL["week_of"], **g, **card.build_card(MODEL, g["gap"], g["var"])}


@app.get("/api/model.json")
def api_model():
    """Everything the offline web app needs (the classifier stays server-side)."""
    return {k: v for k, v in MODEL.items() if k != "classifier"}


@app.get("/api/dashboard")
def api_dashboard():
    villages = []
    for v in store.VILLAGES:
        g = store.gap_estimate(CON, MODEL, v)
        c = card.card_numbers(MODEL, g["gap"], g["var"])
        n_reports = CON.execute("SELECT COUNT(*), SUM(weight=0) FROM reports WHERE village=?", (v,)).fetchone()
        villages.append({"village": v, "reporters": g["reporters"], "reports": n_reports[0], "ignored": n_reports[1] or 0,
                         "basis": g["basis"], "lo": c["lo"], "hi": c["hi"], "fair": round(c["fair"])})
    recent = [dict(r) for r in CON.execute(
        "SELECT village, price, round(weight,2) AS weight, source, ts FROM reports ORDER BY id DESC LIMIT 12")]
    counts = {r[0]: r[1] for r in CON.execute("SELECT intent, COUNT(*) FROM messages WHERE direction='in' GROUP BY intent")}
    return {"week_of": MODEL["week_of"], "sources": MODEL["sources"], "k_anonymity": store.K_ANONYMITY,
            "villages": villages, "recent_reports": recent, "incoming_by_intent": counts,
            "classifier": MODEL["classifier"]["scores"]}


@app.post("/api/officer/report")
def officer_report(village: str = Form(...), price: float = Form(...), member: str = Form(...),
                   x_officer_token: str = Header("")):
    """A cooperative officer entering an offer on a member's behalf (the
    mitigation for who does and doesn't own a phone). Needs OFFICER_TOKEN."""
    token = os.environ.get("OFFICER_TOKEN")
    if not token or x_officer_token != token:
        raise HTTPException(403, "officer token required")
    if village not in store.VILLAGES:
        raise HTTPException(404, "unknown village")
    return store.add_report(CON, MODEL, store.hash_phone("member:" + member), village, price, source="officer")


@app.get("/sw.js")
def service_worker():
    return FileResponse(os.path.join(WEB_DIR, "sw.js"), media_type="application/javascript", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
