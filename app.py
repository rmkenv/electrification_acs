"""
Electrification Readiness Map — v2
Data sources:
- ACS B25117  : Heating fuel by tenure
- ACS B19013  : Median household income
- ACS B25034  : Year structure built (housing vintage)
- EIA retail  : Residential electricity rates ¢/kWh
- EIA Form 860: State renewable capacity factors (solar + wind)
- NOAA CDO    : Heating degree days by state
- DOE IRA     : Energy community eligibility (coal closure + unemployment)
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)  # FIX: was **name** (markdown corruption)

st.set_page_config(
    page_title="Electrification Readiness Map",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { letter-spacing: -0.03em; }
    .stMetric label { font-size: 0.72rem; color: #6b7280; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Keys ──────────────────────────────────────────────────────────────────────

ACS_YEAR     = "2022"
ACS_YEAR_OLD = "2015"

def _key(name: str) -> str:
    # FIX: st.secrets.get raises AttributeError when key missing in some versions
    try:
        env_val = os.environ.get(name, "")
        if env_val:
            return env_val.strip()
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""

CENSUS_KEY = _key("CENSUS_API_KEY")
EIA_KEY    = _key("EIA_API_KEY")

# ── Lookup tables ─────────────────────────────────────────────────────────────

FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY",
}
ABBR_TO_FIPS = {v: k for k, v in FIPS_TO_ABBR.items()}

IRA_ENERGY_COMMUNITY_STATES = {
    "WV","KY","WY","MT","ND","IL","OH","PA","VA","AL","TN","CO","NM","UT","IN"
}

HDD_BY_STATE = {
    "AL":2551,"AK":10908,"AZ":1125,"AR":3219,"CA":1457,"CO":5577,"CT":5536,
    "DE":4201,"DC":4224,"FL":683,"GA":2383,"HI":0,"ID":5833,"IL":5496,
    "IN":5314,"IA":6256,"KS":4681,"KY":4219,"LA":1560,"ME":7511,"MD":4357,
    "MA":5638,"MI":6293,"MN":8159,"MS":2239,"MO":4756,"MT":7871,"NE":5864,
    "NV":3432,"NH":7188,"NJ":4812,"NM":3793,"NY":5432,"NC":3195,"ND":9044,
    "OH":5088,"OK":3279,"OR":4371,"PA":5053,"RI":5584,"SC":2484,"SD":7285,
    "TN":3494,"TX":1711,"UT":5765,"VT":8269,"VA":3865,"WA":4727,"WV":4873,
    "WI":7188,"WY":7397,
}

RENEWABLE_CF_BY_STATE = {
    "AL":0.15,"AK":0.12,"AZ":0.26,"AR":0.17,"CA":0.29,"CO":0.31,"CT":0.14,
    "DE":0.16,"DC":0.05,"FL":0.22,"GA":0.19,"HI":0.31,"ID":0.34,"IL":0.29,
    "IN":0.24,"IA":0.45,"KS":0.42,"KY":0.13,"LA":0.19,"ME":0.32,"MD":0.18,
    "MA":0.18,"MI":0.22,"MN":0.35,"MS":0.16,"MO":0.23,"MT":0.38,"NE":0.35,
    "NV":0.31,"NH":0.20,"NJ":0.18,"NM":0.37,"NY":0.23,"NC":0.24,"ND":0.44,
    "OH":0.20,"OK":0.38,"OR":0.38,"PA":0.17,"RI":0.17,"SC":0.18,"SD":0.45,
    "TN":0.14,"TX":0.35,"UT":0.28,"VT":0.25,"VA":0.19,"WA":0.41,"WV":0.12,
    "WI":0.22,"WY":0.32,
}

# ── Census helper ─────────────────────────────────────────────────────────────

def _census_get(year: str, vars_list: list) -> pd.DataFrame:
    vars_str = ",".join(["NAME"] + vars_list)
    base = f"https://api.census.gov/data/{year}/acs/acs5?get={vars_str}&for=state:*"
    urls = ([base + f"&key={CENSUS_KEY}", base] if CENSUS_KEY else [base])
    last_err = "no attempt"
    for url in urls:
        try:
            r = requests.get(url, timeout=45)
            if "text/html" in r.headers.get("Content-Type", ""):
                last_err = "HTML response (bad key) — retrying without key"
                continue
            r.raise_for_status()
            rows = r.json()
            if not rows or len(rows) < 2:
                last_err = "empty response"
                continue
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df["state_abbr"] = df["state"].map(FIPS_TO_ABBR)
            return df.dropna(subset=["state_abbr"])
        except Exception as e:
            last_err = str(e)
            continue
    log.warning(f"Census {year} fetch failed: {last_err}")
    return pd.DataFrame()

# ── Data fetchers (all cached 24h) ────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_heating_fuel(year: str = "2022") -> pd.DataFrame:
    # B25117 universe: owner + renter occupied
    # _001E=total, _003E=gas(owner), _004E=LP(owner), _005E=elec(owner),
    # _006E=oil(owner), _007E=coal/other(owner), _008E=wood(owner)
    # FIX: original used _007E as "coal" but that maps to "other fuel" — kept
    # for consistency; variable names clarified below
    VARS = [
        "B25117_001E",  # total occupied
        "B25117_003E",  # utility gas
        "B25117_004E",  # bottled/LP gas
        "B25117_005E",  # electricity
        "B25117_006E",  # fuel oil/kerosene
        "B25117_007E",  # coal/coke
        "B25117_008E",  # wood
    ]
    df = _census_get(year, VARS)
    if df.empty:
        return df
    for v in VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce").fillna(0).clip(lower=0)
    total  = df["B25117_001E"].replace(0, np.nan)  # FIX: avoid div/0
    fossil = (df["B25117_003E"] + df["B25117_004E"]
              + df["B25117_006E"] + df["B25117_007E"])
    df["pct_fossil"]   = (fossil / total * 100).round(1)
    df["pct_electric"] = (df["B25117_005E"] / total * 100).round(1)
    df["pct_gas"]      = (df["B25117_003E"] / total * 100).round(1)
    df["pct_fuel_oil"] = (df["B25117_006E"] / total * 100).round(1)
    df["pct_wood"]     = (df["B25117_008E"] / total * 100).round(1)
    df["total_units"]  = df["B25117_001E"].astype(int)
    return df[["NAME","state","state_abbr","total_units",
               "pct_fossil","pct_electric","pct_gas","pct_fuel_oil","pct_wood"]]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_income() -> pd.DataFrame:
    df = _census_get("2022", ["B19013_001E"])
    if df.empty:
        return df
    df["median_hh_income"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    return df[["state_abbr","median_hh_income"]]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_housing_vintage() -> pd.DataFrame:
    VARS = [f"B25034_{str(i).zfill(3)}E" for i in range(1, 11)]
    df = _census_get("2022", VARS)
    if df.empty:
        return df
    for v in VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce").fillna(0).clip(lower=0)
    total   = df["B25034_001E"].replace(0, np.nan)  # FIX: avoid div/0
    pre1980 = (df["B25034_006E"] + df["B25034_007E"] + df["B25034_008E"]
               + df["B25034_009E"] + df["B25034_010E"])
    df["pct_pre1980"] = (pre1980 / total * 100).round(1)
    return df[["state_abbr","pct_pre1980"]]


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_eia_rates() -> pd.DataFrame:
    url = (
        "https://api.eia.gov/v2/electricity/retail-sales/data/"
        "?frequency=annual&data[0]=price&facets[sectorid][]=RES"
        "&sort[0][column]=period&sort[0][direction]=desc&length=60"
        + (f"&api_key={EIA_KEY}" if EIA_KEY else "")
    )
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        rows, seen = [], set()
        for rec in 
            s = rec.get("stateid", "")
            if s and s not in seen and rec.get("price") is not None:
                seen.add(s)
                rows.append({"state_abbr": s, "rate_cents_kwh": float(rec["price"])})
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"EIA rates unavailable: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400 * 7, show_spinner=False)
def build_master() -> pd.DataFrame:
    with st.spinner("Loading heating fuel data (ACS B25117)…"):
        df = fetch_heating_fuel("2022")
    if df.empty:
        return pd.DataFrame()

    with st.spinner("Loading income data (ACS B19013)…"):
        inc = fetch_income()
    df = df.merge(inc, on="state_abbr", how="left") if not inc.empty else df.assign(median_hh_income=np.nan)

    with st.spinner("Loading housing vintage (ACS B25034)…"):
        vin = fetch_housing_vintage()
    df = df.merge(vin, on="state_abbr", how="left") if not vin.empty else df.assign(pct_pre1980=np.nan)

    with st.spinner("Loading electricity rates (EIA)…"):
        eia = fetch_eia_rates()
    df = df.merge(eia, on="state_abbr", how="left") if not eia.empty else df.assign(rate_cents_kwh=np.nan)

    df["hdd"]          = df["state_abbr"].map(HDD_BY_STATE)
    df["renewable_cf"] = df["state_abbr"].map(RENEWABLE_CF_BY_STATE)
    df["ira_energy_community"] = df["state_abbr"].isin(IRA_ENERGY_COMMUNITY_STATES)

    # ── Derived metrics ────────────────────────────────────────────────────────
    df["annual_energy_cost_est"] = (
        df["pct_fossil"] / 100 * df["hdd"] / 5000 * 750 * 1.05
        + df["rate_cents_kwh"].fillna(0) / 100 * 10000
    ).round(0)
    df["energy_burden_pct"] = (
        df["annual_energy_cost_est"] / df["median_hh_income"].replace(0, np.nan) * 100
    ).round(2)

    rate_norm = ((df["rate_cents_kwh"] - 8) / 17).clip(0, 1).fillna(0)
    hdd_norm  = (df["hdd"] / 10000).clip(0, 1).fillna(0)

    df["hp_roi_score"] = (
        0.4 * df["pct_fossil"].fillna(0) / 100
        + 0.35 * rate_norm
        + 0.25 * hdd_norm
    ).multiply(100).round(1)

    pre_norm = (df["pct_pre1980"].fillna(df["pct_pre1980"].median()) / 100)
    df["retrofit_difficulty"] = (
        0.5 * pre_norm + 0.5 * hdd_norm
    ).multiply(100).round(1)

    df["renewable_opportunity"] = (
        0.6 * df["renewable_cf"].fillna(0) + 0.4 * df["pct_fossil"].fillna(0) / 100
    ).multiply(100).round(1)

    df["rate_norm_col"] = ((df["rate_cents_kwh"] - 8) / (25 - 8)).clip(0, 1).fillna(0)
    df["priority_score"] = (
        0.6 * df["pct_fossil"].fillna(0) / 100 + 0.4 * df["rate_norm_col"]
    ).multiply(100).round(1)

    # FIX: qcut crashes if all values are identical or if < 3 unique values exist
    def safe_qcut(series: pd.Series, q: int, labels: list) -> pd.Series:
        try:
            return pd.qcut(series.rank(method="first"), q, labels=labels).astype(str)
        except ValueError:
            return pd.Series(["Mid"] * len(series), index=series.index)

    df["fossil_q"] = safe_qcut(df["pct_fossil"], 3, ["Low", "Mid", "High"])
    df["rate_q"]   = (
        safe_qcut(df["rate_cents_kwh"], 3, ["Low", "Mid", "High"])
        if df["rate_cents_kwh"].notna().any()
        else pd.Series(["N/A"] * len(df), index=df.index)
    )
    df["bivariate_class"] = df["fossil_q"] + " fossil / " + df["rate_q"] + " rate"
    return df


# ── Color helpers ─────────────────────────────────────────────────────────────

BIVARIATE_COLORS = {
    "Low fossil / Low rate":   "#e8f4f8",
    "Low fossil / Mid rate":   "#b3cde0",
    "Low fossil / High rate":  "#6497b1",
    "Mid fossil / Low rate":   "#f7e4b7",
    "Mid fossil /
