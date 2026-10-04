"""End-to-end check of the live loop against a throwaway database:

    python -m tests.test_live_loop
"""
import os
import tempfile

os.environ["SANJHA_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["OFFICER_TOKEN"] = "test-token"
os.environ.pop("AT_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from app import store  # noqa: E402
from app.main import MODEL, app  # noqa: E402
from sms.templates import SMS_LIMIT, TEMPLATES  # noqa: E402

client = TestClient(app)


def sms(text, phone="+254700000001"):
    r = client.post("/sms/incoming", data={"from": phone, "text": text})
    assert r.status_code == 200, r.text
    return r.json()


def main():
    # every reply is one of the fixed templates, and fits in one SMS
    first = sms("bei ngapi leo?")
    assert first["intent"] == "ask_price" and first["template"] in ("card", "abstain"), first
    for text, intent in [("wamenipa 340", "report_offer"), ("nisubiri au niuze?", "ask_wait"), ("msaada", "help"), ("mvua inanyesha", "other")]:
        out = sms(text)
        assert out["intent"] == intent, (text, out)
        assert out["template"] in TEMPLATES and len(out["reply"]) <= SMS_LIMIT
        for banned in ("uza", "sell", "hold"):  # never an instruction
            assert banned not in out["reply"].lower().split(), out["reply"]
        print(f"{text!r:24} -> [{out['intent']}] {out['reply']}")

    # a low offer is flagged against the range, never with a command
    low = sms("wamenipa 250", phone="+254700000002")
    assert low["template"] == "report_low", low

    # sharing narrows the range: 6 farmers report near the fair price
    wide = client.get("/api/card?village=Kiptoo").json()
    for i in range(6):
        sms("jiunge Kiptoo", phone=f"+25471100000{i}")
        sms(f"nimepewa {int(wide['fair']) + 5 * (i - 3)}", phone=f"+25471100000{i}")
    narrow = client.get("/api/card?village=Kiptoo").json()
    assert narrow["basis"] == "village+region" and narrow["var"] < wide["var"], (wide, narrow)
    assert narrow["hi"] - narrow["lo"] < wide["hi"] - wide["lo"]
    print(f"range Kiptoo: {wide['lo']}-{wide['hi']} -> {narrow['lo']}-{narrow['hi']} after 6 farmers shared")

    # k-anonymity: a village with under 5 reporters only ever gets the regional estimate
    assert client.get("/api/card?village=Serem").json()["basis"] == "region"

    # a flood of fake low reports from one phone moves nothing after the first
    before = client.get("/api/card?village=Kiptoo").json()["gap"]
    floods = [sms("wamenipa 150", phone="+254799999999" if i else "+254799999999")["detail"]["report"] for i in range(10)]
    assert sum(1 for f in floods if f["weight"] > 0) <= 1
    # ...and 10 different phones all reporting an absurd price are rejected by the robust filter
    for i in range(10):
        sms("jiunge Kiptoo", phone=f"+25472200000{i}")
        sms("wamenipa 120", phone=f"+25472200000{i}")
    after = client.get("/api/card?village=Kiptoo").json()["gap"]
    assert abs(after - before) < 0.02, (before, after)
    print(f"flood test: Kiptoo gap {before:.4f} -> {after:.4f} after 20 fake low reports")

    # opt-out, USSD, officer entry, privacy
    assert sms("acha")["template"] == "opt_out"
    menu = client.post("/ussd", data={"phoneNumber": "+254700000003", "text": ""}).text
    assert menu.startswith("CON ") and "1." in menu
    assert client.post("/ussd", data={"phoneNumber": "+254700000003", "text": "1"}).text.startswith("END ")
    assert client.post("/ussd", data={"phoneNumber": "+254700000003", "text": "2"}).text.startswith("CON ")
    assert "350" in client.post("/ussd", data={"phoneNumber": "+254700000003", "text": "2*350"}).text
    print("ussd 3*2 ->", client.post("/ussd", data={"phoneNumber": "+254700000003", "text": "3*2"}).text)
    assert client.post("/api/officer/report", data={"village": "Serem", "price": 380, "member": "M-17"}).status_code == 403
    ok = client.post("/api/officer/report", data={"village": "Serem", "price": 380, "member": "M-17"}, headers={"x-officer-token": "test-token"})
    assert ok.status_code == 200 and ok.json()["weight"] > 0
    con = store.connect()
    dump = " ".join(str(tuple(r)) for t in ("farmers", "reports", "messages") for r in con.execute(f"SELECT * FROM {t}"))
    assert "+2547" not in dump, "a raw phone number reached the database"

    # abstention path: a week wider than anything backtested
    MODEL["relative_width"] = MODEL["w_star"] + 0.01
    assert sms("bei", phone="+254700000004")["template"] == "abstain"
    assert sms("nisubiri?", phone="+254700000004")["template"] == "abstain"
    assert client.get("/").status_code == 200 and client.get("/sw.js").status_code == 200
    assert "classifier" not in client.get("/api/model.json").json()
    print("all live-loop checks passed")


if __name__ == "__main__":
    main()
