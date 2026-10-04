"""
Intent classifier for incoming SMS — the "understanding messy Swahili SMS"
capability. TF-IDF (word 1-2 grams) + multinomial logistic regression,
trained with scikit-learn and exported as a few KB of JSON; inference is the
~25 lines of plain Python in `predict()` (no scikit-learn at run time, and
directly portable to the phone-side JS).

Intents: report_offer, ask_price, ask_wait, help, opt_out, other.

TRAINING DATA IS SYNTHETIC: template-expanded Swahili / English / mixed
messages with filler words and typos, generated below. Amazon MASSIVE
(sw-KE) is NOT used yet — its intents don't map onto these, though it would
make a good source of real "other" messages. The held-out set at the bottom
was hand-written separately from the templates; it is small, and written by
the same non-native author, so treat its accuracy as an upper bound.
"""
from __future__ import annotations

import json
import math
import random
import re

INTENTS = ["report_offer", "ask_price", "ask_wait", "help", "opt_out", "other"]
MIN_CONFIDENCE = 0.45  # below this the reply is the "I did not understand" template
PRICE_RANGE = (50, 3000)  # plausible KES per kg; numbers outside are not treated as prices

_TOKEN = re.compile(r"<num>|[a-z']+|\?")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def tokenize(text: str) -> list[str]:
    text = _NUMBER.sub(" <num> ", text.lower())
    return _TOKEN.findall(text)


def ngrams(tokens: list[str]) -> list[str]:
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def extract_price(text: str) -> float | None:
    """First number in the message that could plausibly be a price per kg."""
    for m in _NUMBER.finditer(text):
        value = float(m.group().replace(",", "."))
        if PRICE_RANGE[0] <= value <= PRICE_RANGE[1]:
            return value
    return None


def predict(model: dict, text: str) -> tuple[str, float]:
    """(intent, confidence). Pure Python: tf * idf, L2-normalise, dot with the
    class weights, softmax."""
    counts: dict[int, float] = {}
    for g in ngrams(tokenize(text)):
        idx = model["vocab"].get(g)
        if idx is not None:
            counts[idx] = counts.get(idx, 0.0) + 1.0
    if not counts:
        return "other", 0.0
    vec = {i: c * model["idf"][i] for i, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    scores = [b + sum(w[i] * v / norm for i, v in vec.items()) for w, b in zip(model["coef"], model["intercept"])]
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    probs = [e / sum(exps) for e in exps]
    best = max(range(len(probs)), key=probs.__getitem__)
    if probs[best] < MIN_CONFIDENCE:
        return "other", probs[best]
    return model["intents"][best], probs[best]


# ----------------------------------------------------------------------------
# Synthetic training data
# ----------------------------------------------------------------------------
_SEEDS = {
    "report_offer": [
        "wamenipa {n}", "nimepewa {n}", "wananipa {n} kwa kilo", "mnunuzi anatoa {n}", "nimeambiwa {n}",
        "ofa {n}", "{n}", "dalali amesema {n}", "wamenipa shilingi {n}", "bei {n}", "bei niliyopewa ni {n}",
        "amenipa {n} kwa kg", "wanataka kununua kwa {n}", "leo wamenipa {n}", "mnunuzi amesema {n} kilo",
        "nimepata ofa ya {n}", "they offered {n}", "offer {n}", "got {n} per kg", "buyer says {n}",
        "wameniambia {n} tu", "sh {n}", "kes {n} kwa kilo", "{n} kwa kilo", "{n}/kg", "wamenipa {n} bob",
        "bei ya dalali {n}", "nimeletewa bei ya {n}", "i was offered {n}", "wanatoa {n} leo",
    ],
    "ask_price": [
        "bei", "bei ngapi", "bei ngapi leo?", "bei ya kahawa leo", "bei sawa ni ngapi", "nipe bei",
        "kahawa ni bei gani", "bei ya wiki hii", "price", "price today", "what is the price", "nataka kujua bei",
        "bei ikoje", "soko likoje leo", "bei gani sasa", "kahawa inauzwa ngapi", "bei ya leo ni ngapi?",
        "naomba bei", "tuma bei", "bei sawa", "how much is coffee", "kilo ni ngapi", "bei ya kahawa ni gani?",
        "nijulishe bei", "bei ya soko",
    ],
    "ask_wait": [
        "nisubiri?", "niuze sasa au nisubiri", "nisubiri au niuze", "ni bora kusubiri?", "should i wait",
        "subiri", "nikisubiri nitapata zaidi?", "niuze leo?", "ningoje au niuze", "bei itapanda?",
        "nisubiri wiki ngapi", "wait or sell", "je nisubiri", "kusubiri kuna faida?", "bei itashuka?",
        "ningoje bei ipande?", "niuze nusu?", "should i sell now", "ningoje",
    ],
    "help": [
        "msaada", "saidia", "nisaidie", "help", "sielewi", "hii ni nini", "jinsi ya kutumia", "maelezo",
        "nifanyeje", "how does this work", "info", "naomba msaada", "sijui kutumia", "nieleze", "menu",
        "hii inafanyaje kazi", "nahitaji msaada",
    ],
    "opt_out": [
        "acha", "stop", "ondoa", "nitoe", "sitaki ujumbe", "usinitumie tena", "jiondoa", "simamisha",
        "unsubscribe", "toka", "acha kunitumia sms", "sitaki tena", "niondoe", "acha ujumbe", "stop sms",
        "usitume tena", "nimechoka na ujumbe",
    ],
    "other": [
        "habari", "asante", "sawa", "mvua inanyesha", "nani wewe", "hello", "ok", "mambo", "nitakuja kesho",
        "shamba langu ni kubwa", "mungu akubariki", "habari za asubuhi", "poa", "nimefika", "kesho ni mkutano",
        "nipigie simu", "hi", "shikamoo", "watoto wako shuleni", "namba yangu ni {n}", "nina miti {n}",
        "tutaonana", "safi sana", "nimevuna magunia {big}", "lala salama", "uko wapi", "thanks", "good morning",
    ],
}
_PREFIX = ["", "", "", "", "habari ", "tafadhali ", "jamani ", "sasa ", "leo ", "pls ", "eti "]
_SUFFIX = ["", "", "", "", " leo", " tafadhali", " sasa", " asante", " jamani", " pls", " ?"]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1 :] if rng.random() < 0.5 else word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def synthetic_messages(per_intent: int = 400, seed: int = 0) -> list[tuple[str, str]]:
    """(text, intent) pairs. Every one is synthetic."""
    rng = random.Random(seed)
    out = []
    for intent, seeds in _SEEDS.items():
        for _ in range(per_intent):
            text = rng.choice(seeds).format(n=rng.randrange(150, 900, 5), big=rng.randrange(3, 40))
            text = rng.choice(_PREFIX) + text + rng.choice(_SUFFIX)
            if rng.random() < 0.25:
                words = text.split()
                j = rng.randrange(len(words))
                words[j] = _typo(words[j], rng)
                text = " ".join(words)
            out.append((text, intent))
    return out


# Hand-written, not drawn from the templates above.
HELD_OUT = [
    ("wamenipa 340", "report_offer"), ("dalali ameniambia 310 kwa kilo moja", "report_offer"),
    ("nimepewa bei ya 365 leo asubuhi", "report_offer"), ("offer ni 290 tu jamani", "report_offer"),
    ("mnunuzi wa kijiji anatoa 400", "report_offer"), ("355", "report_offer"),
    ("wanasema watanipa 330 kwa kg", "report_offer"), ("buyer offered me 375", "report_offer"),
    ("bei ngapi leo?", "ask_price"), ("kahawa bei gani wiki hii", "ask_price"),
    ("naomba kujua bei sawa", "ask_price"), ("price ya kahawa", "ask_price"),
    ("bei ya kahawa iko aje", "ask_price"), ("nitumie bei", "ask_price"), ("bei?", "ask_price"),
    ("nisubiri au niuze sasa hivi", "ask_wait"), ("je bei itapanda wiki ijayo?", "ask_wait"),
    ("ni vizuri kungoja?", "ask_wait"), ("should i wait or sell", "ask_wait"), ("niuze leo ama ningoje", "ask_wait"),
    ("msaada tafadhali", "help"), ("sielewi hii kitu", "help"), ("nisaidie jinsi ya kutumia", "help"),
    ("help me", "help"), ("naomba maelezo", "help"),
    ("acha", "opt_out"), ("sitaki ujumbe tena", "opt_out"), ("stop", "opt_out"), ("niondoe kwa orodha", "opt_out"),
    ("usinitumie sms", "opt_out"),
    ("habari yako", "other"), ("asante sana", "other"), ("nitakuja kesho shambani", "other"),
    ("mvua imenyesha sana", "other"), ("hello there", "other"), ("sawa sawa", "other"),
]


def train(per_intent: int = 400, seed: int = 0) -> dict:
    """Fit with scikit-learn, return the JSON-serialisable model plus scores."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    texts, labels = zip(*synthetic_messages(per_intent, seed))
    x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=seed, stratify=labels)
    vec = TfidfVectorizer(analyzer=lambda t: ngrams(tokenize(t)), min_df=2, max_features=600)
    clf = LogisticRegression(C=10.0, max_iter=2000)
    clf.fit(vec.fit_transform(x_tr), y_tr)

    model = {
        "intents": [str(c) for c in clf.classes_],
        "vocab": {k: int(v) for k, v in vec.vocabulary_.items()},
        "idf": [round(float(v), 3) for v in vec.idf_],
        "coef": [[round(float(v), 3) for v in row] for row in clf.coef_],
        "intercept": [round(float(v), 3) for v in clf.intercept_],
        "training_data": "synthetic (template-expanded), see sms/classifier.py",
    }
    synth_acc = sum(predict(model, t)[0] == y for t, y in zip(x_te, y_te)) / len(x_te)
    held = [(t, y, predict(model, t)[0]) for t, y in HELD_OUT]
    model["scores"] = {
        "synthetic_test_accuracy": round(synth_acc, 3),
        "held_out_accuracy": round(sum(y == p for _, y, p in held) / len(held), 3),
        "held_out_n": len(held),
        "n_train": len(x_tr),
    }
    model["_held_out_errors"] = [(t, y, p) for t, y, p in held if y != p]
    return model


if __name__ == "__main__":
    m = train()
    errors = m.pop("_held_out_errors")
    print(m["scores"], f"size: {len(json.dumps(m)) / 1024:.1f} KB")
    for t, y, p in errors:
        print(f"  MISSED  {t!r}: wanted {y}, got {p}")
    for t in ("wamenipa 340", "bei ngapi leo?", "nisubiri?", "acha"):
        print(f"  {t!r} -> {predict(m, t)} price={extract_price(t)}")
