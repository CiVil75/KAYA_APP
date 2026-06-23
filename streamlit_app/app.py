import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(layout="wide")

# =========================
# CONFIG
# =========================

# Focused list of WDI indicators (human name -> code)
INDICATORS = {
    "GHG_total_AR5 (Mt CO2e)": "EN.GHG.ALL.LU.MT.CE.AR5",
    "GHG_per_capita_AR5 (t CO2e per person)": "EN.GHG.ALL.PC.CE.AR5",
    "CO2_to_GDP (kg CO2 per 2017 PPP $)": "EN.GHG.CO2.RT.GDP.PP.KD",
    "Population": "SP.POP.TOTL",
    "GDP_PPP (current international $)": "NY.GDP.MKTP.PP.KD",
    "Energy_use_per_GDP_PPP": "EG.GDP.PUSE.KO.PP.KD",
    "GDP_per_capita_PPP": "NY.GDP.PCAP.PP.KD",
    "Primary_energy_supply_PPP": "EG.EGY.PRIM.PP.KD",
    "Energy_use_comm_per_GDP_PPP": "EG.USE.COMM.GD.PP.KD",
    "Renewable_final_energy_pct": "EG.FEC.RNEW.ZS",
    "Commercial_energy_share_coal_pct": "EG.USE.COMM.CL.ZS",
}

# Top ~60 countries by population (display name -> ISO3), plus region codes World and European Union
ENTITIES = {
    "China": "CHN",
    "India": "IND",
    "United States": "USA",
    "Indonesia": "IDN",
    "Pakistan": "PAK",
    "Brazil": "BRA",
    "Nigeria": "NGA",
    "Bangladesh": "BGD",
    "Russia": "RUS",
    "Mexico": "MEX",
    "Japan": "JPN",
    "Ethiopia": "ETH",
    "Philippines": "PHL",
    "Egypt": "EGY",
    "Vietnam": "VNM",
    "DR Congo": "COD",
    "Turkey": "TUR",
    "Iran": "IRN",
    "Germany": "DEU",
    "Thailand": "THA",
    "United Kingdom": "GBR",
    "France": "FRA",
    "Italy": "ITA",
    "South Africa": "ZAF",
    "Tanzania": "TZA",
    "Myanmar": "MMR",
    "South Korea": "KOR",
    "Colombia": "COL",
    "Kenya": "KEN",
    "Spain": "ESP",
    "Argentina": "ARG",
    "Algeria": "DZA",
    "Sudan": "SDN",
    "Ukraine": "UKR",
    "Uganda": "UGA",
    "Iraq": "IRQ",
    "Poland": "POL",
    "Canada": "CAN",
    "Morocco": "MAR",
    "Saudi Arabia": "SAU",
    "Uzbekistan": "UZB",
    "Peru": "PER",
    "Angola": "AGO",
    "Malaysia": "MYS",
    "Mozambique": "MOZ",
    "Ghana": "GHA",
    "Yemen": "YEM",
    "Nepal": "NPL",
    "Venezuela": "VEN",
    "Madagascar": "MDG",
    "Cameroon": "CMR",
    "Côte d'Ivoire": "CIV",
    "North Korea": "PRK",
    "Australia": "AUS",
    "Taiwan": "TWN",
    "Syria": "SYR",
    "Romania": "ROU",
    # Region / aggregate codes supported by WDI
    "World": "WLD",
    "European Union": "EUU",
}

# =========================
# HELPERS
# =========================

def _requests_session_with_retries(total_retries: int = 3, backoff_factor: float = 0.5):
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@st.cache_data
def fetch_indicator(country, indicator, start, end):
    base = "https://api.worldbank.org/v2"
    url = f"{base}/country/{country}/indicator/{indicator}?format=json&date={start}:{end}&per_page=1000"

    session = _requests_session_with_retries()

    try:
        resp = session.get(url, timeout=10)
    except requests.RequestException:
        return pd.DataFrame()

    if resp.status_code != 200:
        return pd.DataFrame()

    try:
        data = resp.json()
    except ValueError:
        return pd.DataFrame()

    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return pd.DataFrame()

    records = data[1]
    df = pd.DataFrame(records)

    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    df = df[["date", "value"]].copy()
    df.columns = ["year", indicator]

    try:
        df["year"] = df["year"].astype(int)
    except Exception:
        df = df[df["year"].str.isdigit()]
        df["year"] = df["year"].astype(int)

    return df


@st.cache_data
def get_data(country, year_range, indicators_map):
    """Return merged DataFrame of the focused indicators for the given country/region and year range."""
    start, end = year_range

    dfs = []
    for name, code in indicators_map.items():
        if not code or not isinstance(code, str):
            continue
        df_ind = fetch_indicator(country, code, start, end)
        if df_ind.empty:
            continue
        df_ind.rename(columns={code: name}, inplace=True)
        dfs.append(df_ind)

    if not dfs:
        return pd.DataFrame()

    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="year", how="outer")

    df = df.sort_values("year").reset_index(drop=True)
    return df


def make_entity_series(entity_name, df):
    """Return a tidy DataFrame with columns: year, indicator, value, entity"""
    rows = []
    if df.empty:
        return pd.DataFrame()
    for _, r in df.iterrows():
        year = int(r["year"])
        for col in df.columns:
            if col == "year":
                continue
            val = r[col]
            try:
                val = float(val)
            except Exception:
                continue
            rows.append({"year": year, "indicator": col, "value": val, "entity": entity_name})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# =========================
# UI
# =========================

st.title("🌍 Kaya Explorer — focused indicators, default entities")

st.markdown(
    "This view automatically fetches the focused WDI indicators and compares a default set of entities.\n\n"
    "You can override selection (up to 6 entities) from the provided list including two region codes (World, European Union)."
)

col1, col2 = st.columns(2)
with col1:
    selected_entities = st.multiselect(
        "Select up to 6 entities (countries or region codes)", list(ENTITIES.keys()),
        default=["United States", "European Union", "World", "China", "India", "Russia"]
    )
with col2:
    years = st.slider("Years", 1960, 2022, (1990, 2020))

# Enforce max 6 entities
if len(selected_entities) > 6:
    st.error("Please select at most 6 entities.")
    st.stop()

# Always fetch all indicators (no per-indicator selection)
ind_map = INDICATORS.copy()

if st.button("Load and plot all indicators"):
    if not selected_entities:
        st.error("Choose at least one entity to visualize.")
        st.stop()

    start_year, end_year = years
    entities_dfs = []  # list of tuples (label, df)

    with st.spinner("Fetching data for selected entities..."):
        for label in selected_entities:
            code = ENTITIES.get(label, label)
            df = get_data(code, (start_year, end_year), ind_map)
            if df.empty:
                st.warning(f"No data for {label} ({code}) in the selected range.")
                continue
            df["entity"] = label
            entities_dfs.append((label, df))

    if not entities_dfs:
        st.error("No data available to plot after fetching. Try expanding the year range or selecting different entities.")
        st.stop()

    # Build a tidy dataframe with all entity series
    series_list = []
    for label, df in entities_dfs:
        s = make_entity_series(label, df)
        if not s.empty:
            series_list.append(s)

    if not series_list:
        st.error("No numeric series available to plot.")
        st.stop()

    tidy = pd.concat(series_list, ignore_index=True, sort=False)

    st.subheader("Plots for all indicators")
    for indicator in ind_map.keys():
        sub = tidy[tidy["indicator"] == indicator]
        if sub.empty:
            st.info(f"Indicator '{indicator}' has no data for the selected entities/years.")
            continue
        fig = px.line(sub, x="year", y="value", color="entity", markers=True, title=indicator)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Show combined tidy data (first 500 rows)"):
        st.dataframe(tidy.head(500))

else:
    st.info("Click 'Load and plot all indicators' to fetch the focused WDI indicators for the chosen entities.")
