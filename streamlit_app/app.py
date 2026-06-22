import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# =========================
# CONFIG
# =========================

INDICATORS = {
    "Population": "SP.POP.TOTL",
    "GDP": "NY.GDP.MKTP.CD",
    "CO2": "EN.ATM.CO2E.KT",
    "Energy": "EG.USE.PCAP.KG.OE"
}

COUNTRIES = {
    "USA": "USA",
    "China": "CHN",
    "India": "IND",
    "Italy": "ITA",
    "Germany": "DEU",
    "France": "FRA",
    "UK": "GBR"
}

# =========================
# DATA LOADING
# =========================

@st.cache_data
def fetch_indicator(country, indicator, start, end):
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&date={start}:{end}&per_page=1000"
    response = requests.get(url).json()

    if len(response) < 2:
        return pd.DataFrame()

    data = response[1]
    df = pd.DataFrame(data)

    df = df[["date", "value"]]
    df.columns = ["year", indicator]
    df["year"] = df["year"].astype(int)

    return df


@st.cache_data
def get_data(country, year_range):
    start, end = year_range

    dfs = []

    for name, code in INDICATORS.items():
        df = fetch_indicator(country, code, start, end)
        if df.empty:
            continue

        df.rename(columns={code: name}, inplace=True)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="year", how="outer")

    df = df.sort_values("year")
    df = df.dropna()

    return df


# =========================
# KAYA MODEL
# =========================

def compute_kaya(df):

    df["GDP_pc"] = df["GDP"] / df["Population"]

    # Energy è per capita → ricostruiamo totale
    df["Energy_total"] = df["Energy"] * df["Population"]

    df["Energy_intensity"] = df["Energy_total"] / df["GDP"]
    df["Carbon_intensity"] = df["CO2"] / df["Energy_total"]

    df["CO2_reconstructed"] = (
        df["Population"]
        * df["GDP_pc"]
        * df["Energy_intensity"]
        * df["Carbon_intensity"]
    )

    return df


# =========================
# PLOT FUNCTIONS
# =========================

def plot_variable(df, var):
    return px.line(df, x="year", y=var, title=var)


def plot_normalized(df, var):
    base_year = df["year"].min()
    base_val = df[df["year"] == base_year][var].values[0]

    df["normalized"] = df[var] / base_val

    return px.line(
        df,
        x="year",
        y="normalized",
        title=f"{var} (normalized to {base_year})"
    )


# =========================
# UI
# =========================

st.title("🌍 Kaya Identity Explorer")

col1, col2 = st.columns(2)

with col1:
    country_name = st.selectbox("Select Country", list(COUNTRIES.keys()))

with col2:
    years = st.slider("Years", 1990, 2022, (1990, 2020))

country_code = COUNTRIES[country_name]

# =========================
# DATA PROCESS
# =========================

df = get_data(country_code, years)

if df.empty:
    st.error("No data available")
    st.stop()

df = compute_kaya(df)

# =========================
# OUTPUT
# =========================

st.subheader("📊 Kaya Factors")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_variable(df, "Population"), use_container_width=True)
    st.plotly_chart(plot_variable(df, "GDP_pc"), use_container_width=True)

with col2:
    st.plotly_chart(plot_variable(df, "Energy_intensity"), use_container_width=True)
    st.plotly_chart(plot_variable(df, "Carbon_intensity"), use_container_width=True)

# =========================
# NORMALIZED VIEW
# =========================

st.subheader("📈 Normalized Trends")

variable = st.selectbox(
    "Select variable",
    ["Population", "GDP_pc", "Energy_intensity", "Carbon_intensity"]
)

st.plotly_chart(plot_normalized(df, variable), use_container_width=True)

# =========================
# KAYA CHECK
# =========================

st.subheader("🧮 Kaya Identity Validation")

fig = px.line(
    df,
    x="year",
    y=["CO2", "CO2_reconstructed"],
    title="Actual vs Reconstructed CO₂"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# RAW DATA
# =========================

with st.expander("🔍 Show raw data"):
    st.dataframe(df)
