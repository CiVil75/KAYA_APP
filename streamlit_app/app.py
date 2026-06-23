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
    "EU": "EUU",
    "Russia": "RUS"
}

BASE_YEAR = 1990

# consistent colors per entity
ENTITY_COLORS = {
    "World": "black",
    "China": "red",
    "India": "orange",
    "USA": "blue",
    "EU": "green",
    "Russia": "purple"
}

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


def build_variable_dataframe(var_name, code, entities, years):

    df_all = pd.DataFrame()

    for ent_name in entities:
        df = fetch_indicator(ENTITIES[ent_name], code, years)
        if df.empty:
            continue

        df = df.rename(columns={"value": ent_name})

        if df_all.empty:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="year", how="outer")

    return df_all


# =========================
# TRANSFORMS
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
# GENERIC FIGURE (BOOK STYLE)
# =========================

def plot_book_style(df, title):

    df_n = normalize(df)
    df_g = growth(df)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08
    )

    entities = [c for c in df.columns if c != "year"]

    # TOP PANEL
    for e in entities:
        fig.add_trace(
            go.Scatter(
                x=df_n["year"],
                y=df_n[e],
                name=e,
                line=dict(color=ENTITY_COLORS.get(e, "gray"), width=2)
            ),
            row=1, col=1
        )

    # BOTTOM PANEL
    for e in entities:
        fig.add_trace(
            go.Scatter(
                x=df_g["year"],
                y=df_g[e],
                showlegend=False,
                line=dict(color=ENTITY_COLORS.get(e, "gray"), dash="solid")
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=750,
        title=title,
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.01
        )
    )

    fig.update_yaxes(title_text="Normalized (1990 = 1)", row=1, col=1)
    fig.update_yaxes(title_text="Growth rate (%/year)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    # grid like book
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


# =========================
# DERIVED VARIABLES
# =========================

def build_derived(df_pop, df_gdppc, df_energy_intensity, df_co2_intensity):

    df = df_pop.copy()

    # GDP = A * B
    for e in df.columns:
        if e == "year":
            continue
        df[e] = df_pop[e] * df_gdppc[e]

    df_gdp = df.copy()

    # Energy = A * B * C
    for e in df.columns:
        if e == "year":
            continue
        df[e] = df_gdp[e] * df_energy_intensity[e]

    df_energy = df.copy()

    return df_gdp, df_energy


# =========================
# UI
# =========================

st.title("🌍 Kaya Identity — Chapter Figures Reproduction")

entities = st.multiselect(
    "Select entities",
    list(ENTITIES.keys()),
    default=["World", "China", "USA", "EU"]
)

years = st.slider("Years", 1990, 2022, (1990, 2020))

# =========================
# EXECUTION
# =========================

if st.button("Generate Chapter Figures"):

    # ---- BASE VARIABLES ----
    df_pop = build_variable_dataframe("Population", INDICATORS["Population"], entities, years)
    df_gdppc = build_variable_dataframe("GDP per capita", INDICATORS["GDP per capita"], entities, years)
    df_energy_int = build_variable_dataframe("Energy intensity", INDICATORS["Energy intensity"], entities, years)
    df_co2_int = build_variable_dataframe("CO2 intensity", INDICATORS["CO2 intensity"], entities, years)
    df_co2 = build_variable_dataframe("CO2 emissions", INDICATORS["CO2 emissions"], entities, years)

    # ---- DERIVED ----
    df_gdp, df_energy = build_derived(df_pop, df_gdppc, df_energy_int, df_co2_int)

    # =========================
    # FIGURES
    # =========================

    st.header("Fig I.7.1 — Population dynamics")
    st.plotly_chart(plot_book_style(df_pop, "Population dynamics"), use_container_width=True)

    st.header("Fig I.7.4 — GDP per capita dynamics")
    st.plotly_chart(plot_book_style(df_gdppc, "GDP per capita dynamics"), use_container_width=True)

    st.header("Fig I.7.5 — GDP dynamics")
    st.plotly_chart(plot_book_style(df_gdp, "GDP dynamics"), use_container_width=True)

    st.header("Fig I.7.6 — Energy intensity dynamics")
    st.plotly_chart(plot_book_style(df_energy_int, "Energy intensity dynamics"), use_container_width=True)

    st.header("Fig I.7.7 — Energy consumption dynamics")
    st.plotly_chart(plot_book_style(df_energy, "Energy consumption dynamics"), use_container_width=True)

    st.header("Fig I.7.8 — Emission intensity dynamics")
    st.plotly_chart(plot_book_style(df_co2_int, "Emission intensity dynamics"), use_container_width=True)

    st.header("Fig I.7.9 — CO2 emissions dynamics")
    st.plotly_chart(plot_book_style(df_co2, "CO2 emissions dynamics"), use_container_width=True)

    # ---- COMBINED KAYA ----
    st.header("Fig I.7.10 — Combined Kaya dynamics")

    df_kaya = df_pop.copy()

    for e in entities:
        df_kaya[e] = df_energy[e]  # energy as main comparable

    fig = plot_book_style(df_kaya, "Combined Kaya dynamics (Energy proxy)")
    st.plotly_chart(fig, use_container_width=True)
