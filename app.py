"""
Electrification Readiness Map
ACS B25117 Heating Fuel × EIA Electricity Rates → bivariate state choropleth
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electrification Readiness Map",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #13162a;
        border: 1px solid #1e2340;
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0;
    }
    h1 { letter-spacing: -0.03em; }
    .stMetric label { font-size: 0.75rem; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ACS_YEAR = "2022"
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", st.secrets.get("CENSUS_API_KEY", ""))
EIA_API_KEY    = os.environ.get("EIA_API_KEY",    st.secrets.get("EIA_API_KEY", ""))

B25117_VARS = [
    "B25117_001E",  # total
    "B25117_003E",  # utility gas
    "B25117_004E",  # bottled/LP gas
    "B25117_005E",  # electricity
    "B25117_006E",  # fuel oil/kerosene
    "B25117_007E",  # coal
    "B25117_008E",  # wood
]

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
ABBR_TO_NAME = {v: k for k, v in {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
}.items()}

# ── Data fetching (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_acs() -> pd.DataFrame:
    vars_str = ",".join(["NAME"] + B25117_VARS)
    url = (
        f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
        f"?get={vars_str}&for=state:*"
        + (f"&key={CENSUS_API_KEY}" if CENSUS_API_KEY else "")
    )
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        rows = r.json()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        for v in B25117_VARS:
            df[v] = pd.to_numeric(df[v], errors="coerce").fillna(0).clip(lower=0)
        df["state_abbr"] = df["state"].map(FIPS_TO_ABBR)
        df = df.dropna(subset=["state_abbr"])

        total = df["B25117_001E"]
        fossil = df["B25117_003E"] + df["B25117_004E"] + df["B25117_006E"] + df["B25117_007E"]
        df["pct_fossil"]   = (fossil   / total * 100).round(1)
        df["pct_electric"] = (df["B25117_005E"] / total * 100).round(1)
        df["pct_gas"]      = (df["B25117_003E"] / total * 100).round(1)
        df["pct_fuel_oil"] = (df["B25117_006E"] / total * 100).round(1)
        df["pct_wood"]     = (df["B25117_008E"] / total * 100).round(1)
        df["total_units"]  = total.astype(int)
        return df[["NAME","state","state_abbr","total_units",
                   "pct_fossil","pct_electric","pct_gas","pct_fuel_oil","pct_wood"]]
    except Exception as e:
        st.error(f"Census API error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_eia() -> pd.DataFrame:
    url = (
        "https://api.eia.gov/v2/electricity/retail-sales/data/"
        "?frequency=annual&data[0]=price&facets[sectorid][]=RES"
        "&sort[0][column]=period&sort[0][direction]=desc&length=60"
        + (f"&api_key={EIA_API_KEY}" if EIA_API_KEY else "")
    )
    try:
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        rows = []
        seen = set()
        for rec in data:
            s = rec.get("stateid", "")
            if s and s not in seen and rec.get("price") is not None:
                seen.add(s)
                rows.append({"state_abbr": s, "rate_cents_kwh": float(rec["price"])})
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"EIA API error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400*7, show_spinner=False)
def build_dataframe() -> pd.DataFrame:
    with st.spinner("Fetching Census ACS B25117..."):
        acs = fetch_acs()
    with st.spinner("Fetching EIA electricity rates..."):
        eia = fetch_eia()

    if acs.empty:
        return pd.DataFrame()

    df = acs.copy()
    if not eia.empty:
        df = df.merge(eia, on="state_abbr", how="left")
    else:
        df["rate_cents_kwh"] = np.nan

    # Priority score: 60% fossil share + 40% normalized rate
    rate_min, rate_max = 8.0, 25.0
    df["rate_norm"] = ((df["rate_cents_kwh"] - rate_min) / (rate_max - rate_min) * 100).clip(0, 100)
    df["priority_score"] = (0.6 * df["pct_fossil"] + 0.4 * df["rate_norm"]).round(1)

    # Bivariate class (3×3)
    df["fossil_q"] = pd.qcut(df["pct_fossil"].rank(method="first"), 3, labels=["Low","Mid","High"])
    df["rate_q"]   = pd.qcut(df["rate_cents_kwh"].rank(method="first", na_option="bottom"),
                              3, labels=["Low","Mid","High"])
    df["bivariate_class"] = df["fossil_q"].astype(str) + " fossil / " + df["rate_q"].astype(str) + " rate"
    return df


# ── Bivariate color matrix ────────────────────────────────────────────────────
BIVARIATE_COLORS = {
    "Low fossil / Low rate":   "#e8f4f8",
    "Low fossil / Mid rate":   "#b3cde0",
    "Low fossil / High rate":  "#6497b1",
    "Mid fossil / Low rate":   "#f7e4b7",
    "Mid fossil / Mid rate":   "#d4a843",
    "Mid fossil / High rate":  "#b5720a",
    "High fossil / Low rate":  "#f0c4b4",
    "High fossil / Mid rate":  "#e07b52",
    "High fossil / High rate": "#c0392b",
}


# ── Main app ──────────────────────────────────────────────────────────────────
st.title("⚡ Electrification Readiness Map")
st.caption("ACS B25117 Heating Fuel · EIA Retail Electricity Rates · State-level choropleth")

df = build_dataframe()

if df.empty:
    st.error("No data loaded. Check API keys in Streamlit secrets.")
    st.stop()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Map Controls")
    layer = st.radio(
        "Color by",
        ["Bivariate (Readiness × Rate)", "Fossil Heat %",
         "Electricity Rate (¢/kWh)", "Priority Score"],
        index=0,
    )
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- [ACS 5-Year 2022, Table B25117](https://data.census.gov)")
    st.markdown("- [EIA Retail Electricity Sales](https://www.eia.gov/opendata/)")
    st.divider()
    st.markdown("**Methodology**")
    st.markdown(
        "Fossil heat share = % of occupied units using natural gas, "
        "bottled/LP gas, fuel oil, or coal as primary heat source. "
        "Priority score = 0.6 × fossil% + 0.4 × normalized rate."
    )
    

# ── Build choropleth ──────────────────────────────────────────────────────────
def make_choropleth(df, layer):
    if layer == "Bivariate (Readiness × Rate)":
        df = df.copy()
        df["_color"] = df["bivariate_class"].map(BIVARIATE_COLORS).fillna("#cccccc")
        df["_hover"] = (
            "<b>" + df["NAME"] + "</b><br>" +
            "Fossil heat: " + df["pct_fossil"].astype(str) + "%<br>" +
            "Electric heat: " + df["pct_electric"].astype(str) + "%<br>" +
            "Gas: " + df["pct_gas"].astype(str) + "%<br>" +
            "Fuel oil: " + df["pct_fuel_oil"].astype(str) + "%<br>" +
            "Elec. rate: " + df["rate_cents_kwh"].round(2).astype(str) + "¢/kWh<br>" +
            "Priority score: " + df["priority_score"].astype(str) + "<br>" +
            "Class: " + df["bivariate_class"]
        )
        fig = go.Figure(go.Choropleth(
            locations=df["state_abbr"],
            z=list(range(len(df))),
            locationmode="USA-states",
            colorscale=[[i/(len(df)-1), c] for i, c in enumerate(df["_color"].tolist())],
            showscale=False,
            hovertemplate=df["_hover"] + "<extra></extra>",
            marker_line_color="#0d0f1a",
            marker_line_width=0.8,
        ))
    else:
        col_map = {
            "Fossil Heat %":              ("pct_fossil",       "% Fossil Heat",    "YlOrRd"),
            "Electricity Rate (¢/kWh)":  ("rate_cents_kwh",   "¢/kWh",            "OrRd"),
            "Priority Score":            ("priority_score",    "Priority Score",   "RdPu"),
        }
        col, label, scale = col_map[layer]
        fig = px.choropleth(
            df,
            locations="state_abbr",
            locationmode="USA-states",
            color=col,
            color_continuous_scale=scale,
            scope="usa",
            hover_name="NAME",
            hover_data={
                "state_abbr": False,
                col: True,
                "pct_fossil": True,
                "pct_electric": True,
                "rate_cents_kwh": ":.2f",
                "priority_score": True,
            },
            labels={
                col: label,
                "pct_fossil": "Fossil %",
                "pct_electric": "Electric %",
                "rate_cents_kwh": "Rate ¢/kWh",
                "priority_score": "Priority",
            },
        )

    fig.update_layout(
        geo=dict(
            scope="usa",
            showlakes=False,
            bgcolor="#0d0f1a",
            landcolor="#1a1d2e",
            subunitcolor="#2a2d3e",
        ),
        paper_bgcolor="#0d0f1a",
        plot_bgcolor="#0d0f1a",
        font_color="#e8eaf0",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig


fig = make_choropleth(df, layer)
st.plotly_chart(fig, use_container_width=True)

# ── Bivariate legend ──────────────────────────────────────────────────────────
if layer == "Bivariate (Readiness × Rate)":
    st.markdown("**Legend — Fossil Heat Share × Electricity Rate**")
    cols = st.columns(9)
    order = [
        "High fossil / Low rate", "High fossil / Mid rate", "High fossil / High rate",
        "Mid fossil / Low rate",  "Mid fossil / Mid rate",  "Mid fossil / High rate",
        "Low fossil / Low rate",  "Low fossil / Mid rate",  "Low fossil / High rate",
    ]
    labels = [
        "High fossil<br>Low rate", "High fossil<br>Mid rate", "High fossil<br>High rate",
        "Mid fossil<br>Low rate",  "Mid fossil<br>Mid rate",  "Mid fossil<br>High rate",
        "Low fossil<br>Low rate",  "Low fossil<br>Mid rate",  "Low fossil<br>High rate",
    ]
    for i, (key, lbl) in enumerate(zip(order, labels)):
        color = BIVARIATE_COLORS.get(key, "#ccc")
        cols[i].markdown(
            f'<div style="background:{color};height:32px;border-radius:4px;'
            f'margin-bottom:4px"></div>'
            f'<div style="font-size:9px;color:#6b7280;line-height:1.3">{lbl}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Summary stats ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Fossil Heat Share", f"{df['pct_fossil'].mean():.1f}%")
col2.metric("Avg Electricity Rate",  f"{df['rate_cents_kwh'].mean():.1f}¢/kWh"
            if "rate_cents_kwh" in df.columns else "N/A")
col3.metric("Highest Priority State",
            df.loc[df["priority_score"].idxmax(), "state_abbr"]
            if "priority_score" in df.columns else "N/A")
col4.metric("Total Housing Units",
            f"{df['total_units'].sum()/1e6:.1f}M")

st.divider()

# ── Data table ────────────────────────────────────────────────────────────────
with st.expander("📊 Full state data table"):
    display_cols = {
        "NAME": "State",
        "pct_fossil": "Fossil Heat %",
        "pct_electric": "Electric Heat %",
        "pct_gas": "Gas %",
        "pct_fuel_oil": "Fuel Oil %",
        "pct_wood": "Wood %",
        "rate_cents_kwh": "Rate ¢/kWh",
        "priority_score": "Priority Score",
        "total_units": "Total Units",
    }
    show = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
    show = show.sort_values("Priority Score", ascending=False).reset_index(drop=True)
    st.dataframe(show, use_container_width=True, height=400)

# ── Top 10 highest priority ───────────────────────────────────────────────────
with st.expander("🔴 Top 10 highest priority states"):
    top10 = df.nlargest(10, "priority_score")[
        ["NAME", "pct_fossil", "rate_cents_kwh", "priority_score", "bivariate_class"]
    ].rename(columns={
        "NAME": "State", "pct_fossil": "Fossil %",
        "rate_cents_kwh": "Rate ¢/kWh", "priority_score": "Priority",
        "bivariate_class": "Class",
    }).reset_index(drop=True)
    st.dataframe(top10, use_container_width=True)
