import streamlit as st
import requests
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")

BASE_YEAR = 1990
FINAL_YEAR = 2020

# =========================
# INDICATORS
# =========================

INDICATORS = {
    "Population": ("SP.POP.TOTL", "MPax", 1e6),
    "GDP per capita": ("NY.GDP.PCAP.PP.KD", "k$/pax", 1e3),
    "Energy intensity": ("EG.GDP.PUSE.KO.PP.KD", "Wh/$", 1),
    "CO2 intensity": ("EN.GHG.CO2.RT.GDP.PP.KD", "kgCO2/$", 1),
    "CO2 emissions": ("EN.GHG.ALL.LU.MT.CE.AR5", "GtCO2/y", 1e9)
}

ENTITIES = {
    "World": "WLD",
    "China": "CHN",
    "India": "IND",
    "USA": "USA",
    "European Union": "EUU",
    "Russian Federation": "RUS"
}

COLORS = {
    "World": "black",
    "China": "#d62728",
    "India": "#ff7f0e",
    "USA": "#1f77b4",
    "European Union": "#2ca02c",
    "Russian Federation": "#9467bd"
}

# =========================
# FETCH
# =========================

def fetch(entity, code, years):
    url = f"https://api.worldbank.org/v2/country/{entity}/indicator/{code}?date={years[0]}:{years[1]}&format=json&per_page=1000"
    r = requests.get(url).json()
    if len(r) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(r[1])[["date", "value"]]
    df.columns = ["year", "value"]
    df["year"] = df["year"].astype(int)
    return df.sort_values("year")


def build_df(code, entities, years):
    df_all = pd.DataFrame()
    for e in entities:
        df = fetch(ENTITIES[e], code, years)
        if df.empty:
            continue

        df = df.rename(columns={"value": e})

        if df_all.empty:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="year", how="outer")

    return df_all


# =========================
# DERIVED
# =========================

def derive(df_pop, df_gdppc, df_energy_int):

    df_gdp = df_pop.copy()
    for c in df_pop.columns:
        if c == "year": continue
        df_gdp[c] = df_pop[c] * df_gdppc[c]

    df_energy = df_gdp.copy()
    for c in df_gdp.columns:
        if c == "year": continue
        df_energy[c] = df_gdp[c] * df_energy_int[c]

    return df_gdp, df_energy


# =========================
# NORMALIZATION
# =========================

def normalize(df):
    df_n = df.copy()
    for c in df.columns:
        if c == "year": continue
        base = df[df.year == BASE_YEAR][c]
        if not base.empty and base.values[0] != 0:
            df_n[c] = df[c] / base.values[0]
    return df_n


def growth(df):
    df_g = df.copy()
    for c in df.columns:
        if c == "year": continue
        df_g[c] = df[c].pct_change() * 100
    return df_g


# =========================
# LEGEND FINAL VALUES (2020)
# =========================

def build_labels(df, unit, factor):

    labels = {}

    for c in df.columns:
        if c == "year": continue

        val = df[df.year <= FINAL_YEAR][c].dropna()
        if len(val) > 0:
            v = val.iloc[-1] / factor

            if v > 100:
                labels[c] = f"{c} ({v:.0f} {unit})"
            else:
                labels[c] = f"{c} ({v:.2g} {unit})"
        else:
            labels[c] = c

    return labels


# =========================
# PLOT
# =========================

def plot(df, title, unit="", factor=1, note=None):

    if note:
        st.warning(note)

    labels = build_labels(df, unit, factor)

    df_n = normalize(df)
    df_g = growth(df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    for c in df.columns:
        if c == "year": continue

        lw = 4 if c == "World" else 2

        fig.add_trace(
            go.Scatter(
                x=df_n.year,
                y=df_n[c],
                name=labels[c],
                line=dict(color=COLORS[c], width=lw)
            ),
            row=1, col=1
        )

    for c in df.columns:
        if c == "year": continue

        lw = 2 if c == "World" else 1

        fig.add_trace(
            go.Scatter(
                x=df_g.year,
                y=df_g[c],
                showlegend=False,
                line=dict(color=COLORS[c], width=lw)
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=title,
        height=750,
        legend=dict(orientation="h", y=1.1)
    )

    fig.update_yaxes(title_text="Normalized (1990 = 1)", row=1, col=1)
    fig.update_yaxes(title_text="Growth rate [%/year]", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


# =========================
# UI
# =========================

st.title("🌍 Kaya Identity — Chapter Reproduction (Final Calibration)")

entities = st.multiselect(
    "Entities",
    list(ENTITIES.keys()),
    default=list(ENTITIES.keys())
)

years = st.slider("Years", 1990, 2022, (1990, 2020))


# =========================
# RUN
# =========================

if st.button("Generate Figures"):

    # base
    df_pop = build_df(INDICATORS["Population"][0], entities, years)
    df_gdppc = build_df(INDICATORS["GDP per capita"][0], entities, years)
    df_energy_int = build_df(INDICATORS["Energy intensity"][0], entities, years)
    df_co2_int = build_df(INDICATORS["CO2 intensity"][0], entities, years)
    df_co2 = build_df(INDICATORS["CO2 emissions"][0], entities, years)

    # derived
    df_gdp, df_energy = derive(df_pop, df_gdppc, df_energy_int)

    st.plotly_chart(plot(df_pop, "Fig I.7.1 — Population dynamics", "MPax", 1e6), use_container_width=True)

