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
    """Fetch an indicator from the World Bank API and return a DataFrame with columns [year, <indicator>].

    This function uses HTTPS, a requests session with retries, timeout and robust JSON checks.
    """
    base = "https://api.worldbank.org/v2"
    url = f"{base}/country/{country}/indicator/{indicator}?format=json&date={start}:{end}&per_page=1000"

    session = _requests_session_with_retries()

    try:
        resp = session.get(url, timeout=10)
    except requests.RequestException as e:
        # Network error (DNS, timeout, connection error, etc.)
        st.warning(f"Network error when requesting World Bank API for {indicator}: {e}")
        return pd.DataFrame()

    if resp.status_code != 200:
        st.warning(f"World Bank API returned status {resp.status_code} for URL: {url}")
        return pd.DataFrame()

    try:
        data = resp.json()
    except ValueError:
        st.warning(f"World Bank API returned non-JSON response for URL: {url}")
        return pd.DataFrame()

    # Expecting a list: [metadata, records]
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return pd.DataFrame()

    records = data[1]
    df = pd.DataFrame(records)

    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    df = df[["date", "value"]].copy()
    df.columns = ["year", indicator]

    # convert year to int where possible
    try:
        df["year"] = df["year"].astype(int)
    except Exception:
        df = df[df["year"].str.isdigit()]
        df["year"] = df["year"].astype(int)

    return df


@st.cache_data
def get_data(country, year_range):
    start, end = year_range

    dfs = []

    for name, code in INDICATORS.items():
        df_ind = fetch_indicator(country, code, start, end)
        if df_ind.empty:
            # keep track via warning but continue — we'll decide later which years are usable
            st.info(f"Indicator {name} ({code}) has no data for {country} in {start}:{end} or request failed.")
            continue

        # rename column from code to human-readable name
        df_ind.rename(columns={code: name}, inplace=True)
        dfs.append(df_ind)

    if not dfs:
        return pd.DataFrame()

    # merge on year
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="year", how="outer")

    df = df.sort_values("year")

    # Required indicators for Kaya calculation
    required = ["Population", "GDP", "Energy", "CO2"]

    # Check which required columns are present
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        st.warning(
            f"Impossibile calcolare la Kaya: mancano i seguenti indicatori per il paese/periodo selezionato: {', '.join(missing_cols)}. "
            "Prova ad ampliare l'intervallo di anni o verifica gli indicatori disponibili."
        )
        return pd.DataFrame()

    # Keep only rows with all required values present
    df = df.dropna(subset=required)

    return df


@st.cache_data
def get_indicator_years(country, code, start, end):
    """Return sorted list of years available for an indicator (in the given date window)."""
    df = fetch_indicator(country, code, start, end)
    if df.empty:
        return []
    years = sorted(df["year"].astype(int).unique().tolist())
    return years


# =========================
# KAYA MODEL
# =========================

def compute_kaya(df):
    df = df.copy()

    # GDP per capita
    df["GDP_pc"] = df["GDP"] / df["Population"]

    # Energy is per capita -> reconstruct total
    df["Energy_total"] = df["Energy"] * df["Population"]

    # Intensities
    df["Energy_intensity"] = df["Energy_total"] / df["GDP"]
    df["Carbon_intensity"] = df["CO2"] / df["Energy_total"]

    # Reconstructed CO2 via Kaya identity
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
    base_vals = df[df["year"] == base_year][var].values
    if len(base_vals) == 0 or pd.isna(base_vals[0]) or base_vals[0] == 0:
        st.warning(f"Impossible to normalize: base value for {var} in year {base_year} is missing or zero.")
        return px.line(df, x="year", y=var, title=f"{var} (raw, base {base_year} not available)")

    base_val = base_vals[0]
    d = df.copy()
    d["normalized"] = d[var] / base_val

    return px.line(
        d,
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

# Show availability of indicators for the selected country and range
start_year, end_year = years
availability = []
for name, code in INDICATORS.items():
    yrs = get_indicator_years(country_code, code, start_year, end_year)
    if yrs:
        yrs_str = f"{min(yrs)}-{max(yrs)} ({len(yrs)} years)"
    else:
        yrs_str = "No data"
    availability.append({"Indicator": name, "Code": code, "Available": yrs_str})
avail_df = pd.DataFrame(availability)

with st.expander("📚 Indicator availability for selected country & range"):
    st.table(avail_df)

# =========================
# DATA PROCESS
# =========================

df = get_data(country_code, years)

if df.empty:
    st.error("No data available for the selected country and years. Try expanding the year range or check indicators.")
    st.stop()

# compute kaya components
try:
    df = compute_kaya(df)
except Exception as e:
    st.error(f"Error computing Kaya identity: {e}")
    st.stop()

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
