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
    NOTE: This function does NOT call Streamlit UI functions (st.*) because it is cached.
    """
    base = "https://api.worldbank.org/v2"
    url = f"{base}/country/{country}/indicator/{indicator}?format=json&date={start}:{end}&per_page=1000"

    session = _requests_session_with_retries()

    try:
        resp = session.get(url, timeout=10)
    except requests.RequestException:
        # Network error -> return empty DataFrame
        return pd.DataFrame()

    if resp.status_code != 200:
        return pd.DataFrame()

    try:
        data = resp.json()
    except ValueError:
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
    """Return merged DataFrame of indicators for the given country and year range.

    This cached function does not call st.*; it returns an empty DataFrame if data
    are insufficient. Any UI messages must be shown outside this function.
    The function will attempt a fallback for CO2 using EN.ATM.CO2E.PC (per-capita) if
    the primary CO2 indicator EN.ATM.CO2E.KT is not available.
    """
    start, end = year_range

    dfs = []

    for name, code in INDICATORS.items():
        df_ind = fetch_indicator(country, code, start, end)
        if df_ind.empty:
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

    # Attempt CO2 fallback if primary CO2 indicator missing
    if "CO2" not in df.columns:
        # try per-capita indicator EN.ATM.CO2E.PC
        co2_pc_df = fetch_indicator(country, "EN.ATM.CO2E.PC", start, end)
        if not co2_pc_df.empty and "Population" in df.columns:
            co2_pc_df.rename(columns={"EN.ATM.CO2E.PC": "CO2_pc"}, inplace=True)
            df = df.merge(co2_pc_df, on="year", how="left")
            # compute CO2 in kilotons: (metric tons per person * population) / 1000
            df["CO2"] = (
                pd.to_numeric(df["CO2_pc"], errors="coerce") * pd.to_numeric(df["Population"], errors="coerce")
            ) / 1000.0
            df["CO2_from_fallback"] = df["CO2"].notna()

    # Required indicators for Kaya calculation
    required = ["Population", "GDP", "Energy", "CO2"]

    # If any required column is missing entirely, return empty; UI will handle messaging
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
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

# Availability: compute on demand inside the expander to avoid startup network loops
with st.expander("📚 Indicator availability for selected country & range"):
    if st.button("Compute availability"):
        start_year, end_year = years
        availability = []
        missing_indicators = []
        with st.spinner("Checking indicator availability..."):
            for name, code in INDICATORS.items():
                # For CO2, also check fallback per-capita indicator and report both
                if name == "CO2":
                    yrs_primary = get_indicator_years(country_code, code, start_year, end_year)
                    yrs_fallback = get_indicator_years(country_code, "EN.ATM.CO2E.PC", start_year, end_year)

                    if yrs_primary:
                        primary_str = f"KT: {min(yrs_primary)}-{max(yrs_primary)} ({len(yrs_primary)} years)"
                    else:
                        primary_str = "No data"

                    if yrs_fallback:
                        fallback_str = f"PC (fallback): {min(yrs_fallback)}-{max(yrs_fallback)} ({len(yrs_fallback)} years)"
                    else:
                        fallback_str = "No data"

                    combined = primary_str
                    if yrs_fallback and not yrs_primary:
                        # primary missing but fallback exists: mark as missing indicator (but available via fallback)
                        combined = f"Fallback used -> {fallback_str}"
                        missing_indicators.append(name)
                    elif yrs_fallback and yrs_primary:
                        combined = f"{primary_str}; fallback available -> {fallback_str}"

                    availability.append({"Indicator": name, "Code": code, "Available": combined, "Fallback": fallback_str if yrs_fallback else ""})
                else:
                    yrs = get_indicator_years(country_code, code, start_year, end_year)
                    if yrs:
                        yrs_str = f"{min(yrs)}-{max(yrs)} ({len(yrs)} years)"
                    else:
                        yrs_str = "No data"
                        missing_indicators.append(name)
                    availability.append({"Indicator": name, "Code": code, "Available": yrs_str, "Fallback": ""})
        avail_df = pd.DataFrame(availability)
        st.table(avail_df)

        if missing_indicators:
            st.warning(
                f"Non tutti gli indicatori sono disponibili per il paese/intervallo selezionato: {', '.join(missing_indicators)}. "
                "La validazione della Kaya richiede tutti gli indicatori; prova ad ampliare l'intervallo di anni o a selezionare un altro paese."
            )

# =========================
# DATA PROCESS
# =========================

df = get_data(country_code, years)

if df.empty:
    st.error("No data available for the selected country and years. Try expanding the year range or check indicators.")
    st.stop()

# Notify user if CO2 was obtained via fallback
if "CO2_from_fallback" in df.columns and df["CO2_from_fallback"].any():
    st.info("CO2 values were reconstructed from EN.ATM.CO2E.PC (per-capita) because the primary CO2 indicator was not available for some years.")

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
