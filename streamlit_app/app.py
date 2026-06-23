import streamlit as st
import requests
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# CONFIG
# =========================

INDICATORS = {
    "Population": "SP.POP.TOTL",                 # A
    "GDP per capita": "NY.GDP.PCAP.PP.KD",       # B
    "Energy intensity": "EG.GDP.PUSE.KO.PP.KD",  # C
    "CO2 intensity": "EN.GHG.CO2.RT.GDP.PP.KD",  # D
    "CO2 emissions": "EN.GHG.ALL.LU.MT.CE.AR5"
}

ENTITIES = {
    "China": "CHN",
    "India": "IND",
    "USA": "USA",
    "EU": "EUU",
    "World": "WLD",
    "Russia": "RUS"
}

BASE_YEAR = 1990

# =========================
# DATA FETCH
# =========================

def fetch_indicator(country, code, years):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{code}?date={years[0]}:{years[1]}&format=json&per_page=1000"
    data = requests.get(url).json()

    if len(data) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(data[1])[["date", "value"]]
    df = df.rename(columns={"date": "year"})
    df["year"] = df["year"].astype(int)

    return df.sort_values("year")


def build_dataset(entity, years):
    df_all = pd.DataFrame()

    for name, code in INDICATORS.items():
        df = fetch_indicator(entity, code, years)
        if df.empty:
            continue

        df = df.rename(columns={"value": name})

        if df_all.empty:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="year", how="outer")

    if df_all.empty:
        return df_all

    # =========================
    # DERIVED VARIABLES
    # =========================

    df_all["GDP"] = df_all["Population"] * df_all["GDP per capita"]       # A*B
    df_all["Energy"] = df_all["GDP"] * df_all["Energy intensity"]         # A*B*C

    return df_all


# =========================
# TRANSFORMATIONS
# =========================

def normalize(df):
    df_n = df.copy()
    for col in df.columns:
        if col == "year":
            continue
        base = df[df["year"] == BASE_YEAR][col]
        if not base.empty and base.values[0] != 0:
            df_n[col] = df[col] / base.values[0]
    return df_n


def growth(df):
    df_g = df.copy()
    for col in df.columns:
        if col == "year":
            continue
        df_g[col] = df[col].pct_change() * 100
    return df_g


# =========================
# GENERIC FIGURE BUILDER
# =========================

def plot_figure(df, variables, title):

    df_norm = normalize(df)
    df_growth = growth(df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # top
    for v in variables:
        fig.add_trace(go.Scatter(x=df_norm["year"], y=df_norm[v], name=v), row=1, col=1)

    # bottom
    for v in variables:
        fig.add_trace(go.Scatter(x=df_growth["year"], y=df_growth[v], showlegend=False), row=2, col=1)

    fig.update_layout(height=700, title=title)

    return fig


# =========================
# FIGURE DEFINITIONS (CHAPTER MAPPING)
# =========================

FIGURES = {
    "Fig I.7.1 – Population": ["Population"],
    "Fig I.7.4 – GDP per capita": ["GDP per capita"],
    "Fig I.7.5 – GDP": ["GDP"],
    "Fig I.7.6 – Energy intensity": ["Energy intensity"],
    "Fig I.7.7 – Energy consumption": ["Energy"],
    "Fig I.7.8 – CO2 intensity": ["CO2 intensity"],
    "Fig I.7.9 – CO2 emissions": ["CO2 emissions"],
}

# Combined Kaya (Fig I.7.10)
KAYA_COMBINED = [
    "Population",
    "GDP per capita",
    "Energy intensity",
    "CO2 intensity",
    "Energy"
]

# =========================
# UI
# =========================

st.title("🌍 Kaya Explorer – Full Chapter Reproduction")

entities = st.multiselect(
    "Select entities", list(ENTITIES.keys()),
    default=["World", "China", "USA", "EU"]
)

years = st.slider("Years", 1990, 2022, (1990, 2020))

# =========================
# RUN
# =========================

if st.button("Run full chapter reproduction"):

    for entity_name in entities:

        st.header(f"=== {entity_name} ===")

        df = build_dataset(ENTITIES[entity_name], years)

        if df.empty:
            st.warning(f"No data for {entity_name}")
            continue

        # ---- SINGLE FIGURES ----
        for fig_name, variables in FIGURES.items():

            if all(v in df.columns for v in variables):
                fig = plot_figure(df, variables, f"{fig_name} – {entity_name}")
                st.plotly_chart(fig, use_container_width=True)

        # ---- COMBINED KAYA ----
        st.subheader("Fig I.7.10 – Combined Kaya dynamics")

        if all(v in df.columns for v in KAYA_COMBINED):
            fig = plot_figure(df, KAYA_COMBINED, f"Kaya Combined – {entity_name}")
            st.plotly_chart(fig, use_container_width=True)

        # ---- APPROX FIG I.7.11 ----
        st.subheader("Fig I.7.11 – CO2 dynamic focus")

        if "CO2 emissions" in df.columns:
            fig = plot_figure(df, ["CO2 emissions"], f"CO2 dynamics – {entity_name}")
            st.plotly_chart(fig, use_container_width=True)
