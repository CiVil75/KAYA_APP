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
    "Population": "SP.POP.TOTL",                      # A
    "GDP per capita": "NY.GDP.PCAP.PP.KD",            # B
    "Energy intensity": "EG.GDP.PUSE.KO.PP.KD",       # C
    "CO2 intensity": "EN.GHG.CO2.RT.GDP.PP.KD",       # D proxy
}

ENTITIES = {
    "China": "CHN",
    "India": "IND",
    "USA": "USA",
    "EU": "EUU",
    "World": "WLD"
}

BASE_YEAR = 1990

# =========================
# DATA FUNCTION
# =========================

def fetch_wdi(country, indicator, start, end):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=1000"
    r = requests.get(url).json()

    if len(r) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(r[1])
    df = df[["date", "value"]].rename(columns={"date": "year"})
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")

    return df


def build_kaya_dataframe(entity, years):
    data = {}

    for name, code in INDICATORS.items():
        df = fetch_wdi(entity, code, years[0], years[1])
        if df.empty:
            return pd.DataFrame()

        data[name] = df["value"].values
        years_series = df["year"]

    df_all = pd.DataFrame(data)
    df_all["year"] = years_series.values

    # =========================
    # ENERGY CONSUMPTION (A × B × C)
    # =========================

    df_all["Energy consumption"] = (
        df_all["Population"] *
        df_all["GDP per capita"] *
        df_all["Energy intensity"]
    )

    return df_all


# =========================
# TRANSFORMATIONS
# =========================

def normalize(df):
    df = df.copy()

    for col in df.columns:
        if col == "year":
            continue

        base = df[df["year"] == BASE_YEAR][col]
        if base.empty or base.values[0] == 0:
            continue

        df[col] = df[col] / base.values[0]

    return df


def growth(df):
    df = df.copy()

    for col in df.columns:
        if col == "year":
            continue

        df[col] = df[col].pct_change() * 100

    return df


# =========================
# PLOT FUNCTION
# =========================

def plot_combined_kaya(df, entity_name):

    df_norm = normalize(df)
    df_growth = growth(df)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{entity_name} – Normalized Kaya factors (base 1990)",
            f"{entity_name} – Growth rates (%/year)"
        )
    )

    variables = [
        "Population",
        "GDP per capita",
        "Energy intensity",
        "CO2 intensity",
        "Energy consumption"
    ]

    colors = {
        "Population": "gold",
        "GDP per capita": "cyan",
        "Energy intensity": "green",
        "CO2 intensity": "red",
        "Energy consumption": "purple",
    }

    # TOP PANEL
    for var in variables:
        fig.add_trace(
            go.Scatter(
                x=df_norm["year"],
                y=df_norm[var],
                name=var,
                line=dict(color=colors[var])
            ),
            row=1, col=1
        )

    # BOTTOM PANEL
    for var in variables:
        fig.add_trace(
            go.Scatter(
                x=df_growth["year"],
                y=df_growth[var],
                showlegend=False,
                line=dict(color=colors[var])
            ),
            row=2, col=1
        )

    fig.update_yaxes(title_text="Normalized (1990=1)", row=1, col=1)
    fig.update_yaxes(title_text="Growth rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    fig.update_layout(height=750)

    return fig


# =========================
# UI
# =========================

st.title("🌍 Kaya Explorer – Full Chapter Reproduction")

st.markdown("""
This version reproduces **Figure I.7.10-style combined Kaya dynamics**:

- Population (A)
- GDP per capita (B)
- Energy intensity (C)
- Carbon intensity (D)
- ✅ Energy consumption (A × B × C)

All plots:
- Top → normalized to 1990
- Bottom → growth rates
""")

entities = st.multiselect(
    "Select entities",
    list(ENTITIES.keys()),
    default=["World", "China", "USA", "EU"]
)

years = st.slider("Years", 1990, 2022, (1990, 2020))

# =========================
# EXECUTION
# =========================

if st.button("Run Kaya Analysis"):

    for entity in entities:

        st.header(entity)

        df = build_kaya_dataframe(ENTITIES[entity], years)

        if df.empty:
            st.warning(f"No data for {entity}")
            continue

        fig = plot_combined_kaya(df, entity)
        st.plotly_chart(fig, use_container_width=True)
