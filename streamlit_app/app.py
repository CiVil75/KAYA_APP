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

# Focused list of WDI indicators (human name -> code) per user request
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

# Sample set of countries. Keep short to avoid UI overwhelm — users can add codes via the extra input.
COUNTRIES = {
    "USA": "USA",
    "China": "CHN",
    "India": "IND",
    "Italy": "ITA",
    "Germany": "DEU",
    "France": "FRA",
    "United Kingdom": "GBR",
    "Brazil": "BRA",
    "South Africa": "ZAF",
    "Indonesia": "IDN",
}

# Conversion constants (to kg oil equivalent)
TOE_TO_KG_OIL_EQ = 1000.0            # 1 toe = 1000 kg oil-equivalent
KTOE_TO_KG_OIL_EQ = 1_000_000.0      # 1 ktoe = 1,000 toe = 1,000,000 kg
MWH_TO_KG_OIL_EQ = 85.98             # 1 MWh ≈ 0.08598 toe -> *1000 = 85.98 kg
KWH_TO_KG_OIL_EQ = MWH_TO_KG_OIL_EQ / 1000.0
GJ_TO_KG_OIL_EQ = 23.9               # 1 GJ ≈ 0.0239 toe -> *1000 = 23.9 kg

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


# =========================
# DATA LOADING
# =========================

@st.cache_data
def fetch_indicator(country, indicator, start, end):
    """Fetch an indicator from the World Bank API and return a DataFrame with columns [year, <indicator>]."""
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
def get_indicator_years(country, code, start, end):
    df = fetch_indicator(country, code, start, end)
    if df.empty:
        return []
    years = sorted(df["year"].astype(int).unique().tolist())
    return years


def _detect_energy_unit(code: str, name: str):
    code_l = code.lower() if isinstance(code, str) else ""
    name_l = name.lower() if isinstance(name, str) else ""

    if "pcap" in code_l or "per capita" in name_l:
        if "kg" in code_l or "kg" in name_l or "kg oil" in name_l:
            return "kg_per_capita"
        if "toe" in code_l or "toe" in name_l:
            return "toe_per_capita"
        if "mwh" in code_l or "mwh" in name_l:
            return "mwh_per_capita"

    if "ktoe" in code_l or "kt.oe" in code_l or "ktoe" in name_l or "kto" in name_l:
        return "ktoe"
    if "toe" in code_l and "ktoe" not in code_l:
        return "toe"

    if "mwh" in code_l or "mwh" in name_l:
        return "mwh"
    if "kwh" in code_l or "kwh" in name_l:
        return "kwh"
    if "gwh" in code_l or "gwh" in name_l:
        return "gwh"

    if "gj" in code_l or "gigajoule" in name_l:
        return "gj"

    return None


@st.cache_data
def get_data(country, year_range, indicators_map):
    """Return merged DataFrame of selected indicators for the given country and year range.

    indicators_map: dict mapping human name -> code
    """
    start, end = year_range

    dfs = []

    # fetch each selected indicator
    for name, code in indicators_map.items():
        if not code or not isinstance(code, str):
            continue
        df_ind = fetch_indicator(country, code, start, end)
        if df_ind.empty:
            continue
        # rename column to human-readable name
        df_ind.rename(columns={code: name}, inplace=True)
        dfs.append(df_ind)

    if not dfs:
        return pd.DataFrame()

    # merge on year
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="year", how="outer")

    df = df.sort_values("year").reset_index(drop=True)

    # energy normalization and CO2 fallbacks are preserved from original app, but simplified here
    # (we don't aggressively reconstruct CO2 from many energy units in this focused version)

    return df


# =========================
# PLOT HELPERS
# =========================


def plot_variable_single_country(df, var):
    return px.line(df, x="year", y=var, title=var)


def plot_variable_multi_country(df, var):
    fig = px.line(df, x="year", y=var, color="country", markers=True, title=f"{var} — comparison")
    return fig


# =========================
# UI
# =========================

st.title("🌍 Kaya Identity Explorer — Focused WDI Variables & Multi-country Comparison")

with st.expander("Description"):
    st.markdown(
        """
        This variant of the Kaya Explorer limits the WDI indicators to a focused set you provided and
        allows comparing a single variable's trend across up to 6 countries.

        Enter additional country ISO3 codes or WDI indicator codes (CODE or CODE:Name) using the text inputs below.
        """
    )

col1, col2 = st.columns(2)
with col1:
    selected_countries = st.multiselect("Select up to 6 countries (by name)", list(COUNTRIES.keys()), default=["USA", "China"])
with col2:
    years = st.slider("Years", 1960, 2022, (1990, 2020))

# Allow adding extra countries by ISO3 code
extra_countries_raw = st.text_input("Extra country ISO3 codes (comma-separated, e.g. TUR, MEX)")
if extra_countries_raw.strip():
    parts = [p.strip().upper() for p in extra_countries_raw.split(",") if p.strip()]
    for p in parts:
        # if not present, add using ISO3 as both key and code
        if p not in COUNTRIES:
            COUNTRIES[p] = p
            if p not in selected_countries:
                selected_countries.append(p)
    st.success("Added extra countries to the selection box (by ISO3).")

# Indicator selection (limited to the focused set)
indicator_names = list(INDICATORS.keys())
selected_ind = st.multiselect("Indicators to fetch (choose 1 to compare across countries)", indicator_names, default=[indicator_names[0]])

# Allow adding extra indicators by code
extra_ind_raw = st.text_input("Extra WDI indicator codes (comma-separated). Optionally use CODE:Name to provide a display name.")
if st.button("Add indicators") and extra_ind_raw.strip():
    parts = [p.strip() for p in extra_ind_raw.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            code, name = p.split(":", 1)
            INDICATORS[name.strip()] = code.strip()
            if name.strip() not in selected_ind:
                selected_ind.append(name.strip())
        else:
            code = p
            INDICATORS[code] = code
            if code not in selected_ind:
                selected_ind.append(code)
    st.success("Added extra indicators to the selection. Click 'Load data' to fetch.")

# Enforce max 6 countries
if len(selected_countries) > 6:
    st.error("Please select at most 6 countries.")
    st.stop()

# Ensure exactly one indicator is chosen for cross-country comparison
if len(selected_ind) == 0:
    st.warning("Select at least one indicator to fetch.")

# Button to load the selected data (avoids heavy startup network activity)
if st.button("Load data"):
    if len(selected_countries) == 0:
        st.error("Choose at least one country.")
        st.stop()
    if len(selected_ind) == 0:
        st.error("Choose at least one indicator.")
        st.stop()

    with st.spinner("Fetching indicators from World Bank for selected countries..."):
        start_year, end_year = years
        to_fetch = {name: INDICATORS[name] for name in selected_ind}

        country_dfs = []
        for cname in selected_countries:
            ccode = COUNTRIES.get(cname, cname)  # if user added ISO3 directly, key==value
            df = get_data(ccode, (start_year, end_year), to_fetch)
            if df.empty:
                st.warning(f"No data for {cname} in the selected range.")
                continue
            df["country"] = cname
            country_dfs.append(df)

        if not country_dfs:
            st.error("No data available for the selected countries and years. Try expanding the year range or check indicators.")
            st.stop()

        # If multiple countries, build comparison for the first selected indicator (or let user pick)
        if len(selected_ind) == 1:
            var = selected_ind[0]
        else:
            var = st.selectbox("Select variable to compare across countries", selected_ind)

        # Concatenate and prepare data for plotting
        combined = pd.concat(country_dfs, ignore_index=True, sort=False)
        # Keep only year, var, country
        if var not in combined.columns:
            st.error(f"The selected variable '{var}' is not available in the fetched data.")
        else:
            plot_df = combined[["year", var, "country"]].copy()
            plot_df[var] = pd.to_numeric(plot_df[var], errors="coerce")
            plot_df = plot_df.dropna(subset=[var])

            if plot_df.empty:
                st.error("No numeric values available for the chosen variable across the selected countries/years.")
            else:
                st.subheader("Comparison plot")
                st.plotly_chart(plot_variable_multi_country(plot_df, var), use_container_width=True)

        # Show raw combined data
        with st.expander("🔍 Show raw combined data"):
            st.dataframe(combined)

else:
    st.info("Click 'Load data' to fetch the chosen indicators for the selected countries and year range.")

# End of file
