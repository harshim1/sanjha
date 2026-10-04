"""
Secondary grounding data — WFP food prices (via HDX) and FAOSTAT, per the
official World Bank "Small AI for Development" brief's Annex B (Agriculture)
dataset list. These are NOT used as inputs to the GARCH/conformal forecasting
engine (that stays on global arabica futures, per the "borrow the market's
answer" design — see models/volatility.py). They serve a different job: the
brief's Section 7.2 scores "data that shows the problem and gap being filled
by the solution" separately from "data you build with." WFP food prices and
FAOSTAT production/yield/price statistics are exactly that problem-is-real
evidence — a local, independently-sourced market price series and country-
level production context, which is also a much better real-world analogue
for farmgate "anchor" prices than the synthetic ones data/synthetic.py has
to generate.

NETWORK NOTE: as of this build, data.humdata.org, fenixservices.fao.org and
bulks-faostat.fao.org are all blocked by this account's egress policy — not
just from the cloud sandbox, but from the user's own machine too (same proxy
policy applies there). This is an account-level allowlist, not a sandbox
limitation; check the policy or try from an unmanaged network/the hackathon
venue before the deadline. Every function here takes a `local_path` so a
CSV/ZIP downloaded by hand (in a browser, on any machine) can be dropped in
and used with zero code changes.

URL patterns below follow HDX's and FAOSTAT's documented, stable naming
conventions, but neither has been verified reachable from this environment
(network blocked) — confirm the exact resource slug/URL resolves before
relying on it, e.g. from the HDX dataset page in a browser, before the demo.
"""
from __future__ import annotations

import io
import os
import sys
import warnings
import zipfile

import pandas as pd

# HDX dataset slug convention: data.humdata.org/dataset/wfp-food-prices-for-<country>
# (lowercase, hyphenated). The page lists its current CSV resource; the resource
# URL itself changes with each WFP update, so this constant is the DATASET page,
# not a direct file link — resolve the actual CSV link from that page.
HDX_WFP_DATASET_PAGE = "https://data.humdata.org/dataset/wfp-food-prices-for-{country_slug}"

# FAOSTAT bulk downloads: stable domain, versioned filenames per domain.
# "Prices" domain (producer/farmgate prices by country/commodity/year) and
# "Production" domain (yield/production by crop). Confirm exact filenames
# against https://www.fao.org/faostat/en/#data before relying on them.
FAOSTAT_PRICES_BULK_URL = "https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip"
FAOSTAT_PRODUCTION_BULK_URL = (
    "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
)


def _warn_unreachable(what: str, local_path_hint: str) -> None:
    warnings.warn(
        f"[Sanjha grounding] Could not reach {what} (network blocked in this "
        f"environment). Falling back to a clearly-labeled synthetic stand-in. "
        f"Pass local_path='{local_path_hint}' once you've downloaded the real "
        f"file (by hand, from any machine with network access) to use it with "
        f"no code changes.",
        stacklevel=2,
    )


def fetch_wfp_food_prices(
    country: str = "Kenya",
    commodity_filter: str = "Coffee",
    local_path: str | None = None,
) -> pd.DataFrame:
    """WFP food price time series for a country, filtered to a commodity.
    Real pull: resolves the HDX dataset page, finds its CSV resource, downloads
    it. Falls back to a labeled synthetic local-market series (log-normal
    noise around a level, NOT GARCH-driven like the futures series, since WFP
    series are local retail/wholesale observations, not a liquid market) if
    unreachable or `local_path` isn't supplied.
    """
    if local_path and os.path.exists(local_path):
        df = pd.read_csv(local_path)
        df["source"] = f"wfp_hdx_local_file:{os.path.basename(local_path)}"
        return df

    try:
        import requests

        country_slug = country.lower().replace(" ", "-")
        page_url = HDX_WFP_DATASET_PAGE.format(country_slug=country_slug)
        resp = requests.get(page_url, timeout=10)
        resp.raise_for_status()
        # A real implementation parses the HDX CKAN API (package_show) for the
        # resource's direct download URL rather than scraping the HTML page;
        # left as a TODO since this environment can't reach HDX to test it.
        raise RuntimeError("HDX resource resolution not implemented — see TODO in source")
    except Exception as exc:  # noqa: BLE001
        print(f"WFP/HDX fetch failed: {exc}", file=sys.stderr)
        _warn_unreachable("WFP food prices (HDX)", f"data/raw/wfp_{country.lower()}_prices.csv")

    import numpy as np

    rng = np.random.default_rng(41)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=48, freq="MS")
    level = 420.0  # illustrative KES/kg level, NOT a real WFP figure
    noise = rng.normal(0, 0.06, size=len(dates))
    price = level * np.exp(np.cumsum(noise) * 0.3)
    return pd.DataFrame(
        {
            "date": dates,
            "country": country,
            "commodity": commodity_filter,
            "price_local_currency": price,
            "source": "synthetic_placeholder_not_wfp",
        }
    )


def fetch_faostat_coffee(country: str = "Kenya", local_path: str | None = None) -> pd.DataFrame:
    """FAOSTAT production/yield/price-by-country-and-year for coffee (green).
    Real pull: downloads and filters the bulk "Production" and "Prices" zips
    (large — tens of MB; filter to the coffee item code and country after
    extraction). Falls back to a labeled placeholder if unreachable.
    """
    if local_path and os.path.exists(local_path):
        if local_path.endswith(".zip"):
            with zipfile.ZipFile(local_path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(name) as f:
                    df = pd.read_csv(io.BytesIO(f.read()), encoding="latin-1")
        else:
            df = pd.read_csv(local_path)
        df["source"] = f"faostat_local_file:{os.path.basename(local_path)}"
        return df

    try:
        import requests

        resp = requests.get(FAOSTAT_PRODUCTION_BULK_URL, timeout=10, stream=True)
        resp.raise_for_status()
        raise RuntimeError("FAOSTAT bulk parsing not implemented in this pass — see TODO in source")
    except Exception as exc:  # noqa: BLE001
        print(f"FAOSTAT fetch failed: {exc}", file=sys.stderr)
        _warn_unreachable("FAOSTAT coffee production/yield", "data/raw/faostat_coffee.zip")

    return pd.DataFrame(
        {
            "country": [country] * 5,
            "year": [2021, 2022, 2023, 2024, 2025],
            "item": ["Coffee, green"] * 5,
            "production_tonnes": [None] * 5,
            "yield_hg_per_ha": [None] * 5,
            "note": ["FAOSTAT unreachable from this environment — fill in from a manual download"] * 5,
            "source": "unavailable_placeholder",
        }
    )


def grounding_summary(country: str = "Kenya") -> str:
    """A short markdown note for the 'problem is real' evidence the brief
    scores under Data grounding (Section 09, 15%). Honest about what is and
    isn't real data right now.
    """
    wfp = fetch_wfp_food_prices(country=country)
    fao = fetch_faostat_coffee(country=country)
    wfp_is_real = not wfp["source"].iloc[0].startswith("synthetic")
    fao_is_real = not fao["source"].iloc[0] == "unavailable_placeholder"

    lines = [
        f"## Grounding data — {country} coffee",
        "",
        f"- WFP food prices (HDX): {'real data loaded' if wfp_is_real else '**not yet pulled** — network blocked in this environment; see data/grounding.py docstring'}",
        f"- FAOSTAT production/yield: {'real data loaded' if fao_is_real else '**not yet pulled** — same network constraint'}",
        "",
        "Before presenting to judges: pull both from a network that can reach "
        "data.humdata.org and bulks-faostat.fao.org (try the hackathon venue, "
        "or ask your org admin to allowlist them), or download the files by "
        "hand and pass `local_path` to the fetch functions in data/grounding.py.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(grounding_summary())
    print()
    print(fetch_wfp_food_prices().head())
    print()
    print(fetch_faostat_coffee())
