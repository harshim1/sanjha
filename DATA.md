# Sanjha — data sheet

Every dataset the tool uses or cites, with source, licence, size, and what it
does **not** cover (section 7.2 of the challenge brief). Lines marked
**CONFIRM** are licence or access terms that were not verified during the
build and must be checked before submission.

## 1. Data the tool is built with

| Dataset | Used for | Source | Size | Licence / terms | Real or synthetic |
|---|---|---|---|---|---|
| Arabica coffee futures, front month (KC=F), daily close | Global price forecast, volatility model, the whole backtest | Yahoo Finance via the `yfinance` library; prices originate from ICE Futures U.S. | 3,017 trading days, 6 Oct 2014 – 2 Oct 2026 | Yahoo terms of use: unofficial access, personal/research use; not licensed for redistribution or a commercial product. **CONFIRM.** Production needs a licensed feed. | Real |
| KES per USD exchange rate, daily | Converting the futures price to shillings | Intended: FRED (series id in `data/fetch.py` is unverified). Actually used: `data/synthetic.py` | 3,077 simulated days | n/a (generated) | **Synthetic** — a random walk pinned to end at 129 KES/USD |
| Farmgate anchor prices (cooperative payouts) | Level of the local gap | Intended: Nairobi Coffee Exchange results / UCDA monthly reports / the cooperative's own payout records. Actually used: `data/synthetic.py` | 143 monthly values | n/a (generated) | **Synthetic** — generated from the decomposition formula with a known gap |
| Farmer-reported offers | Tracking the local gap week to week | Intended: farmers' own SMS/USSD reports. Actually used: `app/seed.py`, `data/synthetic.py` | 24 seeded offers across 4 villages (demo database); 4 villages × 150 weeks for the Kalman validation | n/a (generated) | **Synthetic** |
| Swahili/English SMS messages labelled by intent | Training the intent classifier | Template-expanded in `sms/classifier.py` (no LLM used) | 2,400 messages, 6 intents; plus 36 hand-written test messages | n/a (generated; test set written by the author) | **Synthetic** |
| Swahili voice clips | Reading the card aloud offline | ElevenLabs text-to-speech, generated at build time by `voice/build_clips.py` | 35 clips (not yet generated — needs an API key) | Depends on the ElevenLabs plan. **CONFIRM** commercial-use terms for the tier used. | Synthetic voice |

Unit conversion `k = 1.35` (USD/lb green coffee to KES/kg parchment, with the
exchange rate applied separately) is a **placeholder constant**, not a
measured parchment-to-green outturn ratio.

Decision-layer defaults (0.5%/month storage loss, KES 0.5/kg/week storage
cost, 5%/month cost of cash) are **placeholders**, not measured values.

## 2. What the data does not cover

- **No real farmgate price has been seen by this system.** The futures are
  real; everything local (exchange rate, cooperative payouts, village offers)
  is synthetic. The forecast-band backtest is therefore evidence about the
  global coffee price only. The local-gap filter has been validated only
  against a simulator whose assumptions (villages share a regional driver;
  report noise of 5%) were chosen by the author.
- **The futures series is the expiring contract about 39% of days.** Yahoo's
  KC=F follows the front month through its delivery period, when it is thinly
  traded and can jump 6–14% in a day. Physical coffee is priced off the most
  active contract, which this series is not on those days.
- **Arabica only, one exchange.** No robusta, no quality or grade
  differential, no certification premium, no cherry-versus-parchment
  distinction. A farmer selling cherry, or a lower grade, would see a
  different fair price.
- **Reported offers are opening bids, not transaction prices,** and will come
  disproportionately from farmers who own a phone. GSMA's Mobile Gender Gap
  reports document that women are less likely to own one; reports entered by
  a cooperative officer are the mitigation, untested.
- **The classifier has never seen a real farmer's message.** Its training
  data is synthetic and its 36-message test set was written by the same
  non-native author. Real messages will include spelling, Sheng and
  code-switching the templates do not anticipate. Amazon MASSIVE (sw-KE,
  CC BY 4.0 — **CONFIRM**) is named in the brief and is **not used**.
- **Swahili only, and unreviewed.** Nine of eleven message templates were
  drafted by the author and have not been checked by a native speaker. Noor's
  home language is not covered: it would need the templates translated by a
  speaker, the classifier retrained on that language, and recorded human
  voice prompts (ElevenLabs does not list most Kenyan local languages).
- **No weather, yield, soil or disease data.** The tool addresses the
  price-at-harvest half of the agriculture brief, not the falling-yield half.
- **One country's units.** Prices are in KES per kg; nothing has been run on
  another market (the Agmarknet rerun was not built).

## 3. Evidence that the problem is real

To be completed with figures — each needs **source, year, country** (brief,
section 7.2). The module meant to pull these (`data/grounding.py`) returns
placeholders because the sites were unreachable during the build.

| Claim to support | Where to get the figure | Status |
|---|---|---|
| Farmers sell at harvest when prices are low because they need cash; credit at harvest changes when they sell | Burke, Bergquist & Miguel, "Sell Low and Buy High: Arbitrage and Local Price Effects in Kenyan Markets", *Quarterly Journal of Economics*, 2019 (Kenya; maize, not coffee) | Cited; no figure extracted |
| Farmgate price is a small and variable share of the export price | FAOSTAT producer prices (coffee, Kenya) against ICE futures; Nairobi Coffee Exchange auction results | **Not pulled** |
| Local market prices are available as an independent reference | WFP food prices via HDX (Kenya) — covers staples such as maize and beans, **not coffee** | **Not pulled** |
| Women are less likely to own a phone, and far less likely a smartphone | GSMA Mobile Gender Gap Report (latest year, Kenya or Sub-Saharan Africa) | Cited; no figure extracted |
| Mobile money and basic-phone reach | World Bank Global Findex (Kenya) | **Not pulled** |

## 4. Methods cited

- Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS 2021 — the calibrated range.
- Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation", *JASA*, 2007 — pinball loss, CRPS, Brier score.
- Diebold & Mariano, "Comparing Predictive Accuracy", *JBES*, 1995 — the test against the naive baseline.
- Bollerslev, "Generalized Autoregressive Conditional Heteroskedasticity", *Journal of Econometrics*, 1986 — GARCH(1,1), via the `arch` package.

## 5. Privacy

Phone numbers are stored only as salted hashes. A village's own estimate is
used only after five distinct farmers have reported there. The SMS card
contains no personal data. Consent wording for joining ("JIUNGE") and the
retention period for stored offers are **not yet defined**.
