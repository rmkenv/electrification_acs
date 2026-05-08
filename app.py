

import os
import logging
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(**name**)

st.set_page_config(
page_title=“Electrification Readiness Map”,
page_icon=“⚡”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

st.markdown(”””

<style>
    .block-container { padding-top: 1.5rem; }
    h1 { letter-spacing: -0.03em; }
    .stMetric label { font-size: 0.72rem; color: #6b7280; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>

“””, unsafe_allow_html=True)

# ── Keys ──────────────────────────────────────────────────────────────────────

ACS_YEAR     = “2022”
ACS_YEAR_OLD = “2015”

def _key(name):
try:
v = os.environ.get(name) or st.secrets.get(name, “”)
return v.strip() if v else “”
except Exception:
return “”

CENSUS_KEY = _key(“CENSUS_API_KEY”)
EIA_KEY    = _key(“EIA_API_KEY”)

# ── Lookup tables ─────────────────────────────────────────────────────────────

FIPS_TO_ABBR = {
“01”:“AL”,“02”:“AK”,“04”:“AZ”,“05”:“AR”,“06”:“CA”,“08”:“CO”,“09”:“CT”,
“10”:“DE”,“11”:“DC”,“12”:“FL”,“13”:“GA”,“15”:“HI”,“16”:“ID”,“17”:“IL”,
“18”:“IN”,“19”:“IA”,“20”:“KS”,“21”:“KY”,“22”:“LA”,“23”:“ME”,“24”:“MD”,
“25”:“MA”,“26”:“MI”,“27”:“MN”,“28”:“MS”,“29”:“MO”,“30”:“MT”,“31”:“NE”,
“32”:“NV”,“33”:“NH”,“34”:“NJ”,“35”:“NM”,“36”:“NY”,“37”:“NC”,“38”:“ND”,
“39”:“OH”,“40”:“OK”,“41”:“OR”,“42”:“PA”,“44”:“RI”,“45”:“SC”,“46”:“SD”,
“47”:“TN”,“48”:“TX”,“49”:“UT”,“50”:“VT”,“51”:“VA”,“53”:“WA”,“54”:“WV”,
“55”:“WI”,“56”:“WY”,
}
ABBR_TO_FIPS = {v: k for k, v in FIPS_TO_ABBR.items()}

# IRA Energy Community states (simplified — states with significant coal closure

# zones OR high fossil-fuel employment unemployment qualifying for 10% IRA adder)

IRA_ENERGY_COMMUNITY_STATES = {
“WV”,“KY”,“WY”,“MT”,“ND”,“IL”,“OH”,“PA”,“VA”,“AL”,“TN”,“CO”,“NM”,“UT”,“IN”
}

# Hardcoded NOAA 30-year normal heating degree days (base 65°F) by state abbr

# Source: NOAA 1991-2020 Climate Normals

HDD_BY_STATE = {
“AL”:2551,“AK”:10908,“AZ”:1125,“AR”:3219,“CA”:1457,“CO”:5577,“CT”:5536,
“DE”:4201,“DC”:4224,“FL”:683,“GA”:2383,“HI”:0,“ID”:5833,“IL”:5496,
“IN”:5314,“IA”:6256,“KS”:4681,“KY”:4219,“LA”:1560,“ME”:7511,“MD”:4357,
“MA”:5638,“MI”:6293,“MN”:8159,“MS”:2239,“MO”:4756,“MT”:7871,“NE”:5864,
“NV”:3432,“NH”:7188,“NJ”:4812,“NM”:3793,“NY”:5432,“NC”:3195,“ND”:9044,
“OH”:5088,“OK”:3279,“OR”:4371,“PA”:5053,“RI”:5584,“SC”:2484,“SD”:7285,
“TN”:3494,“TX”:1711,“UT”:5765,“VT”:8269,“VA”:3865,“WA”:4727,“WV”:4873,
“WI”:7188,“WY”:7397,
}

# EIA state-level renewable capacity factors (2022, weighted avg solar+wind %)

# Source: EIA Electric Power Annual Table 6.7.B

RENEWABLE_CF_BY_STATE = {
“AL”:0.15,“AK”:0.12,“AZ”:0.26,“AR”:0.17,“CA”:0.29,“CO”:0.31,“CT”:0.14,
“DE”:0.16,“DC”:0.05,“FL”:0.22,“GA”:0.19,“HI”:0.31,“ID”:0.34,“IL”:0.29,
“IN”:0.24,“IA”:0.45,“KS”:0.42,“KY”:0.13,“LA”:0.19,“ME”:0.32,“MD”:0.18,
“MA”:0.18,“MI”:0.22,“MN”:0.35,“MS”:0.16,“MO”:0.23,“MT”:0.38,“NE”:0.35,
“NV”:0.31,“NH”:0.20,“NJ”:0.18,“NM”:0.37,“NY”:0.23,“NC”:0.24,“ND”:0.44,
“OH”:0.20,“OK”:0.38,“OR”:0.38,“PA”:0.17,“RI”:0.17,“SC”:0.18,“SD”:0.45,
“TN”:0.14,“TX”:0.35,“UT”:0.28,“VT”:0.25,“VA”:0.19,“WA”:0.41,“WV”:0.12,
“WI”:0.22,“WY”:0.32,
}

# ── Census helper ─────────────────────────────────────────────────────────────

def _census_get(year: str, vars_list: list) -> pd.DataFrame:
“”“Fetch ACS 5-year variables for all states with key fallback.”””
vars_str = “,”.join([“NAME”] + vars_list)
base = f”https://api.census.gov/data/{year}/acs/acs5?get={vars_str}&for=state:*”
urls = ([base + f”&key={CENSUS_KEY}”, base] if CENSUS_KEY else [base])
last_err = “no attempt”
for url in urls:
try:
r = requests.get(url, timeout=45)
if “text/html” in r.headers.get(“Content-Type”, “”):
last_err = “HTML response (bad key) — retrying without key”
continue
r.raise_for_status()
rows = r.json()
if not rows or len(rows) < 2:
last_err = “empty response”
continue
df = pd.DataFrame(rows[1:], columns=rows[0])
df[“state_abbr”] = df[“state”].map(FIPS_TO_ABBR)
return df.dropna(subset=[“state_abbr”])
except Exception as e:
last_err = str(e)
continue
log.warning(f”Census {year} fetch failed: {last_err}”)
return pd.DataFrame()

# ── Data fetchers (all cached 24h) ────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_heating_fuel(year=“2022”) -> pd.DataFrame:
VARS = [“B25117_001E”,“B25117_003E”,“B25117_004E”,
“B25117_005E”,“B25117_006E”,“B25117_007E”,“B25117_008E”]
df = _census_get(year, VARS)
if df.empty:
return df
for v in VARS:
df[v] = pd.to_numeric(df[v], errors=“coerce”).fillna(0).clip(lower=0)
total  = df[“B25117_001E”]
fossil = df[“B25117_003E”] + df[“B25117_004E”] + df[“B25117_006E”] + df[“B25117_007E”]
df[“pct_fossil”]   = (fossil / total * 100).round(1)
df[“pct_electric”] = (df[“B25117_005E”] / total * 100).round(1)
df[“pct_gas”]      = (df[“B25117_003E”] / total * 100).round(1)
df[“pct_fuel_oil”] = (df[“B25117_006E”] / total * 100).round(1)
df[“pct_wood”]     = (df[“B25117_008E”] / total * 100).round(1)
df[“total_units”]  = total.astype(int)
return df[[“NAME”,“state”,“state_abbr”,“total_units”,
“pct_fossil”,“pct_electric”,“pct_gas”,“pct_fuel_oil”,“pct_wood”]]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_income() -> pd.DataFrame:
“”“ACS B19013 — median household income.”””
df = _census_get(“2022”, [“B19013_001E”])
if df.empty:
return df
df[“median_hh_income”] = pd.to_numeric(df[“B19013_001E”], errors=“coerce”)
return df[[“state_abbr”,“median_hh_income”]]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_housing_vintage() -> pd.DataFrame:
“”“ACS B25034 — year structure built. Pre-1980 = poor envelope.”””
VARS = [f”B25034_{str(i).zfill(3)}E” for i in range(1, 11)]
# 001=total, 002=2010+, 003=2000-09, 004=1990-99, 005=1980-89,
# 006=1970-79, 007=1960-69, 008=1950-59, 009=1940-49, 010=pre1939
df = _census_get(“2022”, VARS)
if df.empty:
return df
for v in VARS:
df[v] = pd.to_numeric(df[v], errors=“coerce”).fillna(0).clip(lower=0)
total    = df[“B25034_001E”]
pre1980  = (df[“B25034_006E”] + df[“B25034_007E”] + df[“B25034_008E”] +
df[“B25034_009E”] + df[“B25034_010E”])
df[“pct_pre1980”] = (pre1980 / total * 100).round(1)
return df[[“state_abbr”,“pct_pre1980”]]

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_eia_rates() -> pd.DataFrame:
“”“EIA API v2 — residential retail electricity rates ¢/kWh.”””
url = (
“https://api.eia.gov/v2/electricity/retail-sales/data/”
“?frequency=annual&data[0]=price&facets[sectorid][]=RES”
“&sort[0][column]=period&sort[0][direction]=desc&length=60”
+ (f”&api_key={EIA_KEY}” if EIA_KEY else “”)
)
try:
r = requests.get(url, timeout=45)
r.raise_for_status()
data = r.json().get(“response”, {}).get(“data”, [])
rows, seen = [], set()
for rec in data:
s = rec.get(“stateid”, “”)
if s and s not in seen and rec.get(“price”) is not None:
seen.add(s)
rows.append({“state_abbr”: s, “rate_cents_kwh”: float(rec[“price”])})
return pd.DataFrame(rows)
except Exception as e:
st.warning(f”EIA rates unavailable: {e}”)
return pd.DataFrame()

@st.cache_data(ttl=86400*7, show_spinner=False)
def build_master() -> pd.DataFrame:
“”“Join all sources into one state-level dataframe.”””
with st.spinner(“Loading heating fuel data (ACS B25117)…”):
df = fetch_heating_fuel(“2022”)
if df.empty:
return pd.DataFrame()

```
with st.spinner("Loading income data (ACS B19013)..."):
    inc = fetch_income()
if not inc.empty:
    df = df.merge(inc, on="state_abbr", how="left")
else:
    df["median_hh_income"] = np.nan

with st.spinner("Loading housing vintage (ACS B25034)..."):
    vin = fetch_housing_vintage()
if not vin.empty:
    df = df.merge(vin, on="state_abbr", how="left")
else:
    df["pct_pre1980"] = np.nan

with st.spinner("Loading electricity rates (EIA)..."):
    eia = fetch_eia_rates()
if not eia.empty:
    df = df.merge(eia, on="state_abbr", how="left")
else:
    df["rate_cents_kwh"] = np.nan

# Heating degree days + renewable CF from static tables
df["hdd"]          = df["state_abbr"].map(HDD_BY_STATE)
df["renewable_cf"] = df["state_abbr"].map(RENEWABLE_CF_BY_STATE)
df["ira_energy_community"] = df["state_abbr"].isin(IRA_ENERGY_COMMUNITY_STATES)

# ── Derived metrics ───────────────────────────────────────────────────────

# Energy cost burden: estimated annual heating cost as % of income
# Proxy: fossil% × HDD × rate × conversion factor / median income
# (assumes avg 750 therms/yr baseline scaled by HDD and fossil share)
df["annual_energy_cost_est"] = (
    df["pct_fossil"] / 100 * df["hdd"] / 5000 * 750 * 1.05  # gas therms equiv
    + df["rate_cents_kwh"] / 100 * 10000  # baseline kWh × rate
).round(0)
df["energy_burden_pct"] = (
    df["annual_energy_cost_est"] / df["median_hh_income"] * 100
).round(2)

# Heat pump ROI score: high fossil + high rate + moderate HDD = best economics
rate_norm = ((df["rate_cents_kwh"] - 8) / 17).clip(0, 1)
hdd_norm  = (df["hdd"] / 10000).clip(0, 1)
df["hp_roi_score"] = (
    0.4 * df["pct_fossil"] / 100 +
    0.35 * rate_norm +
    0.25 * hdd_norm
).multiply(100).round(1)

# Electrification difficulty: high pre-1980 housing + high HDD = harder
pre_norm = (df["pct_pre1980"] / 100).fillna(0.5)
df["retrofit_difficulty"] = (
    0.5 * pre_norm + 0.5 * hdd_norm
).multiply(100).round(1)

# Renewable opportunity: high renewable CF + high fossil share = best clean case
df["renewable_opportunity"] = (
    0.6 * df["renewable_cf"] + 0.4 * df["pct_fossil"] / 100
).multiply(100).round(1)

# Priority score (original)
rate_min, rate_max = 8.0, 25.0
df["rate_norm_col"] = ((df["rate_cents_kwh"] - rate_min) / (rate_max - rate_min)).clip(0, 1)
df["priority_score"] = (0.6 * df["pct_fossil"] / 100 + 0.4 * df["rate_norm_col"]).multiply(100).round(1)

# Bivariate class
df["fossil_q"] = pd.qcut(df["pct_fossil"].rank(method="first"), 3, labels=["Low","Mid","High"])
if df["rate_cents_kwh"].notna().any():
    df["rate_q"] = pd.qcut(df["rate_cents_kwh"].rank(method="first", na_option="bottom"),
                            3, labels=["Low","Mid","High"])
else:
    df["rate_q"] = "N/A"
df["bivariate_class"] = df["fossil_q"].astype(str) + " fossil / " + df["rate_q"].astype(str) + " rate"

return df
```

# ── Color helpers ─────────────────────────────────────────────────────────────

BIVARIATE_COLORS = {
“Low fossil / Low rate”:   “#e8f4f8”,
“Low fossil / Mid rate”:   “#b3cde0”,
“Low fossil / High rate”:  “#6497b1”,
“Mid fossil / Low rate”:   “#f7e4b7”,
“Mid fossil / Mid rate”:   “#d4a843”,
“Mid fossil / High rate”:  “#b5720a”,
“High fossil / Low rate”:  “#f0c4b4”,
“High fossil / Mid rate”:  “#e07b52”,
“High fossil / High rate”: “#c0392b”,
}

LAYER_CONFIG = {
“🗺️ Bivariate (Readiness × Rate)”: None,
“🔥 Fossil Heat %”:          (“pct_fossil”,          “% Fossil Heat”,           “YlOrRd”),
“⚡ Electricity Rate ¢/kWh”: (“rate_cents_kwh”,       “¢/kWh”,                   “OrRd”),
“💰 Energy Cost Burden %”:   (“energy_burden_pct”,    “Est. Energy Burden %”,    “RdPu”),
“🏠 Pre-1980 Housing %”:     (“pct_pre1980”,          “% Units Built Pre-1980”,  “YlOrBr”),
“♻️ Renewable Opportunity”:  (“renewable_opportunity”,“Renewable Opportunity”,   “Greens”),
“🔧 Retrofit Difficulty”:    (“retrofit_difficulty”,  “Retrofit Difficulty”,     “Oranges”),
“🎯 Heat Pump ROI Score”:    (“hp_roi_score”,         “Heat Pump ROI Score”,     “Blues”),
“🏆 Priority Score”:         (“priority_score”,       “Priority Score”,          “Reds”),
“🌡️ Heating Degree Days”:   (“hdd”,                  “HDD (base 65°F)”,         “PuBu”),
“📈 Median HH Income”:       (“median_hh_income”,     “Median HH Income ($)”,    “BuGn”),
}

def make_choropleth(df: pd.DataFrame, layer: str) -> go.Figure:
if layer == “🗺️ Bivariate (Readiness × Rate)”:
dfc = df.copy()
dfc[”_color”] = dfc[“bivariate_class”].map(BIVARIATE_COLORS).fillna(”#888888”)

```
    def fmt(col, suffix="", decimals=1):
        return dfc[col].apply(
            lambda v: f"{v:.{decimals}f}{suffix}" if pd.notna(v) else "N/A"
        )

    dfc["_hover"] = (
        "<b>" + dfc["NAME"] + "</b><br>" +
        "Fossil heat: "      + fmt("pct_fossil", "%") + "<br>" +
        "Electric heat: "    + fmt("pct_electric", "%") + "<br>" +
        "Elec. rate: "       + fmt("rate_cents_kwh", "¢") + "<br>" +
        "Energy burden: "    + fmt("energy_burden_pct", "%") + "<br>" +
        "Median income: $"   + dfc["median_hh_income"].apply(
                                 lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A") + "<br>" +
        "Pre-1980 housing: " + fmt("pct_pre1980", "%") + "<br>" +
        "HDD: "              + fmt("hdd", "", 0) + "<br>" +
        "Renewable CF: "     + fmt("renewable_cf", "", 2) + "<br>" +
        "HP ROI score: "     + fmt("hp_roi_score") + "<br>" +
        "IRA energy comm: "  + dfc["ira_energy_community"].map({True:"✓ Yes", False:"No"}) + "<br>" +
        "Class: "            + dfc["bivariate_class"]
    )
    n = len(dfc)
    fig = go.Figure(go.Choropleth(
        locations=dfc["state_abbr"],
        z=list(range(n)),
        locationmode="USA-states",
        colorscale=[[i / max(n - 1, 1), c] for i, c in enumerate(dfc["_color"].tolist())],
        showscale=False,
        hovertemplate=dfc["_hover"] + "<extra></extra>",
        marker_line_color="#0d0f1a",
        marker_line_width=0.8,
    ))
else:
    col, label, scale = LAYER_CONFIG[layer]
    hover_extra = {
        "pct_fossil": True,
        "rate_cents_kwh": ":.2f",
        "energy_burden_pct": ":.2f",
        "priority_score": True,
    }
    hover_extra.pop(col, None)
    fig = px.choropleth(
        df,
        locations="state_abbr",
        locationmode="USA-states",
        color=col,
        color_continuous_scale=scale,
        scope="usa",
        hover_name="NAME",
        hover_data={"state_abbr": False, col: True, **hover_extra},
        labels={
            col: label,
            "pct_fossil": "Fossil %",
            "rate_cents_kwh": "Rate ¢/kWh",
            "energy_burden_pct": "Burden %",
            "priority_score": "Priority",
        },
    )

fig.update_layout(
    geo=dict(scope="usa", showlakes=False,
             bgcolor="#0d0f1a", landcolor="#1a1d2e", subunitcolor="#2a2d3e"),
    paper_bgcolor="#0d0f1a",
    plot_bgcolor="#0d0f1a",
    font_color="#e8eaf0",
    margin=dict(l=0, r=0, t=0, b=0),
    height=520,
)
return fig
```

# ══════════════════════════════════════════════════════════════════════════════

# MAIN APP

# ══════════════════════════════════════════════════════════════════════════════

st.title(“⚡ Electrification Readiness Map”)
st.caption(
“ACS B25117 · B19013 · B25034 · EIA Retail Rates · NOAA HDD · “
“DOE IRA Energy Communities · State-level analysis”
)

df = build_master()

if df.empty:
st.error(“No data loaded — Census API unavailable. Check secrets or try again.”)
st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
st.header(“Map Controls”)
layer = st.radio(“Color by”, list(LAYER_CONFIG.keys()), index=0)

```
st.divider()
st.markdown("**Filter**")
show_ira_only = st.checkbox("IRA Energy Communities only", value=False)
min_fossil = st.slider("Min fossil heat %", 0, 80, 0, 5)

st.divider()
st.markdown("**Data sources**")
st.markdown(
    "- [ACS 5-Year 2022](https://data.census.gov) B25117, B19013, B25034\n"
    "- [EIA Retail Sales API](https://www.eia.gov/opendata/)\n"
    "- [NOAA 30-yr Climate Normals](https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals)\n"
    "- [DOE IRA Energy Communities](https://energycommunities.gov/)\n"
    "- [EIA Electric Power Annual](https://www.eia.gov/electricity/annual/)"
)
st.divider()
st.markdown("**Methodology**")
st.markdown(
    "- **Fossil heat share**: gas + LP + fuel oil + coal as % of occupied units\n"
    "- **Energy burden**: est. annual energy cost / median HH income\n"
    "- **HP ROI score**: 40% fossil share + 35% elec rate + 25% HDD\n"
    "- **Retrofit difficulty**: 50% pre-1980 housing + 50% normalized HDD\n"
    "- **Renewable opportunity**: 60% state renewable CF + 40% fossil share\n"
    "- **Priority score**: 60% fossil share + 40% normalized rate"
)
st.divider()
st.markdown("Built by [IQSpatial](https://github.com/rmkenv)")
```

# Apply filters

dff = df.copy()
if show_ira_only:
dff = dff[dff[“ira_energy_community”] == True]
dff = dff[dff[“pct_fossil”] >= min_fossil]

if dff.empty:
st.warning(“No states match current filters.”)
st.stop()

# ── Map ───────────────────────────────────────────────────────────────────────

fig = make_choropleth(dff, layer)
st.plotly_chart(fig, use_container_width=True)

# Bivariate legend

if layer == “🗺️ Bivariate (Readiness × Rate)”:
st.markdown(”**Legend — Fossil Heat Share (rows) × Electricity Rate (columns)**”)
order = [
(“High fossil / High rate”,“High fossil\nHigh rate”),
(“High fossil / Mid rate”, “High fossil\nMid rate”),
(“High fossil / Low rate”, “High fossil\nLow rate”),
(“Mid fossil / High rate”, “Mid fossil\nHigh rate”),
(“Mid fossil / Mid rate”,  “Mid fossil\nMid rate”),
(“Mid fossil / Low rate”,  “Mid fossil\nLow rate”),
(“Low fossil / High rate”, “Low fossil\nHigh rate”),
(“Low fossil / Mid rate”,  “Low fossil\nMid rate”),
(“Low fossil / Low rate”,  “Low fossil\nLow rate”),
]
cols = st.columns(9)
for i, (key, lbl) in enumerate(order):
c = BIVARIATE_COLORS.get(key, “#888”)
cols[i].markdown(
f’<div style="background:{c};height:28px;border-radius:3px;margin-bottom:3px"></div>’
f’<div style="font-size:8px;color:#6b7280;white-space:pre-line">{lbl}</div>’,
unsafe_allow_html=True,
)

st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric(“States shown”, len(dff))
k1.metric(“Avg fossil heat”, f”{dff[‘pct_fossil’].mean():.1f}%”)

rate_ok = “rate_cents_kwh” in dff.columns and dff[“rate_cents_kwh”].notna().any()
k2.metric(“Avg elec rate”,
f”{dff[‘rate_cents_kwh’].mean():.1f}¢” if rate_ok else “N/A”)
burden_ok = “energy_burden_pct” in dff.columns and dff[“energy_burden_pct”].notna().any()
k2.metric(“Avg energy burden”,
f”{dff[‘energy_burden_pct’].mean():.1f}%” if burden_ok else “N/A”)

inc_ok = “median_hh_income” in dff.columns and dff[“median_hh_income”].notna().any()
k3.metric(“Avg median income”,
f”${dff[‘median_hh_income’].mean()/1000:.0f}K” if inc_ok else “N/A”)
k3.metric(“Avg HDD”, f”{dff[‘hdd’].mean():.0f}”)

pre_ok = “pct_pre1980” in dff.columns and dff[“pct_pre1980”].notna().any()
k4.metric(“Avg pre-1980 housing”,
f”{dff[‘pct_pre1980’].mean():.1f}%” if pre_ok else “N/A”)
k4.metric(“IRA energy communities”,
str(dff[“ira_energy_community”].sum()))

ps_ok = “priority_score” in dff.columns and dff[“priority_score”].notna().any()
top_priority = dff.loc[dff[“priority_score”].dropna().index].nlargest(1, “priority_score”)
k5.metric(“Top priority state”,
top_priority[“state_abbr”].values[0] if ps_ok and len(top_priority) else “N/A”)
top_burden = dff.loc[dff[“energy_burden_pct”].dropna().index].nlargest(1, “energy_burden_pct”) if burden_ok else pd.DataFrame()
k5.metric(“Highest burden state”,
top_burden[“state_abbr”].values[0] if len(top_burden) else “N/A”)

k6.metric(“Total housing units”,
f”{dff[‘total_units’].sum()/1e6:.1f}M”)
k6.metric(“Fossil-heated units”,
f”{(dff[‘total_units’] * dff[‘pct_fossil’] / 100).sum()/1e6:.1f}M”)

st.divider()

# ── Analysis tabs ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
“📊 State Rankings”,
“💰 Equity & Burden”,
“🏠 Housing & Retrofit”,
“♻️ Renewable Transition”,
“🏛️ IRA / Policy”,
])

with tab1:
st.subheader(“State Rankings”)
rank_by = st.selectbox(“Rank by”, [
“priority_score”,“pct_fossil”,“energy_burden_pct”,“hp_roi_score”,
“retrofit_difficulty”,“renewable_opportunity”,“rate_cents_kwh”,
], format_func=lambda x: {
“priority_score”: “Priority Score”,
“pct_fossil”: “Fossil Heat %”,
“energy_burden_pct”: “Energy Burden %”,
“hp_roi_score”: “Heat Pump ROI Score”,
“retrofit_difficulty”: “Retrofit Difficulty”,
“renewable_opportunity”: “Renewable Opportunity”,
“rate_cents_kwh”: “Electricity Rate”,
}.get(x, x))

```
ranked = dff.nlargest(20, rank_by)[
    [c for c in ["NAME","state_abbr","pct_fossil","rate_cents_kwh",
                 "energy_burden_pct","median_hh_income","pct_pre1980",
                 "hdd","hp_roi_score","renewable_opportunity",
                 "retrofit_difficulty","priority_score",
                 "ira_energy_community"] if c in dff.columns]
].rename(columns={
    "NAME":"State","state_abbr":"Abbr","pct_fossil":"Fossil %",
    "rate_cents_kwh":"Rate ¢","energy_burden_pct":"Burden %",
    "median_hh_income":"Med Income","pct_pre1980":"Pre-1980 %",
    "hdd":"HDD","hp_roi_score":"HP ROI","renewable_opportunity":"Renew Opp",
    "retrofit_difficulty":"Retrofit Diff","priority_score":"Priority",
    "ira_energy_community":"IRA Comm",
}).reset_index(drop=True)
st.dataframe(ranked, width='stretch', height=500)
```

with tab2:
st.subheader(“Equity & Energy Burden”)
c1, c2 = st.columns(2)
with c1:
if burden_ok and inc_ok:
fig_scatter = px.scatter(
dff.dropna(subset=[“energy_burden_pct”,“median_hh_income”]),
x=“median_hh_income”, y=“energy_burden_pct”,
size=“total_units”, color=“pct_fossil”,
hover_name=“NAME”, text=“state_abbr”,
color_continuous_scale=“YlOrRd”,
labels={
“median_hh_income”:“Median HH Income ($)”,
“energy_burden_pct”:“Est. Energy Burden %”,
“pct_fossil”:“Fossil %”,
},
title=“Income vs Energy Burden (bubble = housing units)”,
)
fig_scatter.update_traces(textposition=“top center”, textfont_size=8)
fig_scatter.update_layout(
paper_bgcolor=”#0d0f1a”, plot_bgcolor=”#111320”,
font_color=”#e8eaf0”, height=380,
)
st.plotly_chart(fig_scatter, use_container_width=True)
else:
st.info(“Income or burden data unavailable.”)
with c2:
if burden_ok:
top_burden_df = dff.nlargest(15, “energy_burden_pct”)[
[“NAME”,“energy_burden_pct”,“median_hh_income”,“pct_fossil”,“rate_cents_kwh”]
].rename(columns={
“NAME”:“State”,“energy_burden_pct”:“Burden %”,
“median_hh_income”:“Income”,“pct_fossil”:“Fossil %”,
“rate_cents_kwh”:“Rate ¢”,
})
fig_bar = px.bar(
top_burden_df, x=“Burden %”, y=“State”,
orientation=“h”, color=“Burden %”,
color_continuous_scale=“RdPu”,
title=“Top 15 States by Energy Burden”,
)
fig_bar.update_layout(
paper_bgcolor=”#0d0f1a”, plot_bgcolor=”#111320”,
font_color=”#e8eaf0”, height=380,
yaxis={“categoryorder”:“total ascending”},
)
st.plotly_chart(fig_bar, use_container_width=True)
else:
st.info(“Burden data requires EIA API key.”)

with tab3:
st.subheader(“Housing Vintage & Retrofit Challenge”)
c1, c2 = st.columns(2)
with c1:
if pre_ok:
fig_vintage = px.scatter(
dff.dropna(subset=[“pct_pre1980”,“hdd”]),
x=“hdd”, y=“pct_pre1980”,
size=“total_units”, color=“retrofit_difficulty”,
hover_name=“NAME”, text=“state_abbr”,
color_continuous_scale=“Oranges”,
labels={
“hdd”:“Heating Degree Days”,
“pct_pre1980”:”% Units Built Pre-1980”,
“retrofit_difficulty”:“Retrofit Difficulty”,
},
title=“HDD vs Pre-1980 Housing (retrofit challenge)”,
)
fig_vintage.update_traces(textposition=“top center”, textfont_size=8)
fig_vintage.update_layout(
paper_bgcolor=”#0d0f1a”, plot_bgcolor=”#111320”,
font_color=”#e8eaf0”, height=380,
)
st.plotly_chart(fig_vintage, use_container_width=True)
else:
st.info(“Housing vintage data unavailable.”)
with c2:
if pre_ok:
fig_diff = px.bar(
dff.nlargest(15,“retrofit_difficulty”)[[“NAME”,“retrofit_difficulty”,“pct_pre1980”,“hdd”]],
x=“retrofit_difficulty”, y=“NAME”, orientation=“h”,
color=“retrofit_difficulty”, color_continuous_scale=“Oranges”,
labels={“retrofit_difficulty”:“Retrofit Difficulty”,“NAME”:“State”},
title=“Top 15 States by Retrofit Difficulty”,
)
fig_diff.update_layout(
paper_bgcolor=”#0d0f1a”, plot_bgcolor=”#111320”,
font_color=”#e8eaf0”, height=380,
yaxis={“categoryorder”:“total ascending”},
)
st.plotly_chart(fig_diff, use_container_width=True)
else:
st.info(“Housing vintage data unavailable.”)

with tab4:
st.subheader(“Renewable Transition Opportunity”)
c1, c2 = st.columns(2)
with c1:
fig_renew = px.scatter(
dff.dropna(subset=[“renewable_cf”,“pct_fossil”]),
x=“renewable_cf”, y=“pct_fossil”,
size=“total_units”, color=“renewable_opportunity”,
hover_name=“NAME”, text=“state_abbr”,
color_continuous_scale=“Greens”,
labels={
“renewable_cf”:“State Renewable Capacity Factor”,
“pct_fossil”:“Fossil Heat Share %”,
“renewable_opportunity”:“Opportunity Score”,
},
title=“Renewable CF vs Fossil Heat (top-right = best transition case)”,
)
fig_renew.update_traces(textposition=“top center”, textfont_size=8)
fig_renew.update_layout(
paper_bgcolor=”#0d0f1a”, plot_bgcolor=”#111320”,
font_color=”#e8eaf0”, height=380,
)
st.plotly_chart(fig_renew, use_container_width=True)
with c2:
fig_roi = px.scatter(
dff.dropna(subset=[“hp_roi_score”,“rate_cents_kwh”]),
x=“hdd”, y=“rate_cents_kwh”,
size=“pct_fossil”, color=“hp_roi_score”,
hover_name=“NAME”, text=“state_abbr”,
color_continuous_scale=“Blues”,
labels={
“hdd”:“Heating Degree Days”,
“rate_cents_kwh”:“Electricity Rate ¢/kWh”,
“hp_roi_score”:“HP ROI Score”,
“pct_fossil”:“Fossil %”,
},
title=“HDD vs Rate — Heat Pump ROI (bubble = fossil share)”,
)
fig_roi.update_traces(textposition=“top center”, textfont_size=8)
fig_roi.update_layout(
paper_bgcolor=”#0d0f1a”, plot_bgcolor=”#111320”,
font_color=”#e8eaf0”, height=380,
)
st.plotly_chart(fig_roi, use_container_width=True)

with tab5:
st.subheader(“IRA Energy Communities & Policy Signals”)
ira_df = dff[dff[“ira_energy_community”] == True].copy()
non_ira = dff[dff[“ira_energy_community”] == False].copy()

```
c1, c2, c3 = st.columns(3)
c1.metric("IRA Energy Community states", len(ira_df))
c2.metric("Avg fossil heat (IRA states)",
          f"{ira_df['pct_fossil'].mean():.1f}%" if len(ira_df) else "N/A")
c3.metric("Fossil-heated units in IRA states",
          f"{(ira_df['total_units'] * ira_df['pct_fossil'] / 100).sum()/1e6:.1f}M"
          if len(ira_df) else "N/A")

st.markdown(
    "States marked as **IRA Energy Communities** qualify for an additional **10% investment "
    "tax credit adder** on clean energy projects under the Inflation Reduction Act. "
    "These are areas with coal mine/plant closures or high fossil-fuel employment."
)

if len(ira_df):
    fig_ira = px.bar(
        ira_df.sort_values("priority_score", ascending=False)[
            ["NAME","priority_score","pct_fossil","rate_cents_kwh","energy_burden_pct"]
        ].rename(columns={
            "NAME":"State","priority_score":"Priority",
            "pct_fossil":"Fossil %","rate_cents_kwh":"Rate ¢",
            "energy_burden_pct":"Burden %",
        }),
        x="State", y="Priority",
        color="Fossil %", color_continuous_scale="Reds",
        title="IRA Energy Community States — Priority Score",
    )
    fig_ira.update_layout(
        paper_bgcolor="#0d0f1a", plot_bgcolor="#111320",
        font_color="#e8eaf0", height=360,
    )
    st.plotly_chart(fig_ira, use_container_width=True)

st.divider()
st.markdown("**All states — full data table**")
full_cols = {
    "NAME":"State","state_abbr":"Abbr","pct_fossil":"Fossil %",
    "pct_electric":"Electric %","rate_cents_kwh":"Rate ¢/kWh",
    "energy_burden_pct":"Burden %","median_hh_income":"Med Income",
    "pct_pre1980":"Pre-1980 %","hdd":"HDD","renewable_cf":"Renew CF",
    "hp_roi_score":"HP ROI","retrofit_difficulty":"Retrofit Diff",
    "renewable_opportunity":"Renew Opp","priority_score":"Priority",
    "ira_energy_community":"IRA Comm","total_units":"Total Units",
}
show = dff[[c for c in full_cols if c in dff.columns]].rename(columns=full_cols)
show = show.sort_values("Priority", ascending=False).reset_index(drop=True)
st.dataframe(show, width='stretch', height=420)
```