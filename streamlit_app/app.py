import streamlit as st
import requests
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# CONFIG
# =========================

BASE_YEAR = 1990

INDICATORS = {
    "Population": "SP.POP.TOTL",
    "GDP per capita": "NY.GDP.PCAP.PP.KD",
    "Energy intensity": "EG.GDP.PUSE.KO.PP.KD",
    "CO2 intensity": "EN.GHG.CO2.RT.GDP.PP.KD",
    "CO2 emissions": "EN.GHG.ALL.LU.MT.CE.AR5"
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

LINE_WIDTH_TOP = 3
LINE_WIDTH_BOTTOM = 1.5

# =========================
# DATA
# =========================

def fetch_indicator(entity, code, years):
    url = f"https://api.worldbank.org/v2/country/{entity}/indicator/{code}?date={years[0]}:{years[1]}&format=json&per_page=1000"
    data = requests.get(url).json()

    if len(data) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(data[1])[["date", "value"]]
    df.columns = ["year", "value"]
    df["year"] = df["year"].astype(int)
    return df.sort_values("year")


def build_df(var_code, entities, years):

    df_all = pd.DataFrame()

    for name in entities:
        df = fetch_indicator(ENTITIES[name], var_code, years)
        if df.empty:
            continue

        df = df.rename(columns={"value": name})

        if df_all.empty:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="year", how="outer")

    return df_all


# =========================
# DERIVED VARIABLES
# =========================

def compute_derived(df_pop, df_gdppc, df_energy_int):

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
# TRANSFORMATIONS
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
# LEGEND WITH 1990 VALUES
# =========================

def build_labels(df):

    labels = {}

    for c in df.columns:
        if c == "year": continue

        base = df[df.year == BASE_YEAR][c]
        if not base.empty:
            val = base.values[0]
            labels[c] = f"{c} ({val:.2e})"
        else:
            labels[c] = c

    return labels


# =========================
# PLOT FUNCTION
# =========================

def plot_figure(df, title):

    labels = build_labels(df)

    df_n = normalize(df)
    df_g = growth(df)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08
    )

    for c in df.columns:
        if c == "year": continue

        fig.add_trace(
            go.Scatter(
                x=df_n["year"],
                y=df_n[c],
                name=labels[c],
                line=dict(color=COLORS[c], width=LINE_WIDTH_TOP)
            ),
            row=1, col=1
        )

    for c in df.columns:
        if c == "year": continue

        fig.add_trace(
            go.Scatter(
                x=df_g["year"],
                y=df_g[c],
                showlegend=False,
                line=dict(color=COLORS[c], width=LINE_WIDTH_BOTTOM)
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=title,
        height=750,
        legend=dict(orientation="h", x=0, y=1.08)
    )

    fig.update_yaxes(title="Normalized (1990 = 1)", row=1, col=1)
    fig.update_yaxes(title="Growth rate [%/year]", row=2, col=1)
    fig.update_xaxes(title="Year", row=2, col=1)

    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


# =========================
# UI
# =========================

st.title("🌍 Kaya Identity — Chapter Figures (High Fidelity)")

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

    df_pop = build_df(INDICATORS["Population"], entities, years)
    df_gdppc = build_df(INDICATORS["GDP per capita"], entities, years)
    df_energy_int = build_df(INDICATORS["Energy intensity"], entities, years)
    df_co2_int = build_df(INDICATORS["CO2 intensity"], entities, years)
    df_co2 = build_df(INDICATORS["CO2 emissions"], entities, years)

    df_gdp, df_energy = compute_derived(df_pop, df_gdppc, df_energy_int)

    st.plotly_chart(plot_figure(df_pop, "Fig I.7.1 — Population dynamics"), use_container_width=True)
    st.plotly_chart(plot_figure(df_gdppc, "Fig I.7.4 — GDP per capita dynamics"), use_container_width=True)
    st.plotly_chart(plot_figure(df_gdp, "Fig I.7.5 — GDP dynamics"), use_container_width=True)
    st.plotly_chart(plot_figure(df_energy_int, "Fig I.7.6 — Energy intensity dynamics"), use_container_width=True)
    st.plotly_chart(plot_figure(df_energy, "Fig I.7.7 — Energy consumption dynamics"), use_container_width=True)
    st.plotly_chart(plot_figure(df_co2_int, "Fig I.7.8 — Emission intensity dynamics"), use_container_width=True)
    st.plotly_chart(plot_figure(df_co2, "Fig I.7.9 — CO₂ emissions dynamics"), use_container_width=True)
