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

# Expanded list of relevant WDI indicators (human name -> code)
INDICATORS = {
    "Population": "SP.POP.TOTL",
    "GDP": "NY.GDP.MKTP.CD",
    "GDP_per_capita": "NY.GDP.PCAP.CD",
    "CO2 (kt)": "EN.ATM.CO2E.KT",
    "CO2 per capita": "EN.ATM.CO2E.PC",
    "CO2 from solid fuel (% of total)": "EN.ATM.CO2E.SF.ZS",
    "CO2 from liquid fuel (% of total)": "EN.ATM.CO2E.LF.ZS",
    "CO2 from gaseous fuel (% of total)": "EN.ATM.CO2E.GF.ZS",
    "Transport CO2 (% of fuel)": "EN.CO2.TRAN.ZS",
    "Electricity/heat CO2 (% of fuel)": "EN.CO2.ETOT.ZS",
    "Industry CO2 (% of fuel)": "EN.CO2.MANF.ZS",
    "CO2 per unit of GDP (kg per 2015 PPP $)": "EN.ATM.CO2E.KD.GD",
    "CO2 per unit energy (kg per kg oil eq)": "EN.ATM.CO2E.EG.ZS",
    "Energy per capita (kg oil eq)": "EG.USE.PCAP.KG.OE",
    "Energy use (total, ktoe)": "EG.USE.COMM.KT.OE"
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
    It must not call Streamlit UI functions because it's cached.
    """
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
    """Return sorted list of years available for an indicator (in the given date window)."""
    df = fetch_indicator(country, code, start, end)
    if df.empty:
        return []
    years = sorted(df["year"].astype(int).unique().tolist())
    return years


@st.cache_data
def get_data(country, year_range, indicators_map):
    """Return merged DataFrame of selected indicators for the given country and year range.

    indicators_map: dict mapping human name -> code
    """
    start, end = year_range

    dfs = []

    for name, code in indicators_map.items():
        # skip empty codes
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

    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="year", how="outer")

    df = df.sort_values("year")

    # If CO2 missing, try fallback per-capita conversion
    if "CO2 (kt)" not in df.columns:
        co2_pc_df = fetch_indicator(country, "EN.ATM.CO2E.PC", start, end)
        if not co2_pc_df.empty and "Population" in df.columns:
            co2_pc_df.rename(columns={"EN.ATM.CO2E.PC": "CO2_pc"}, inplace=True)
            df = df.merge(co2_pc_df, on="year", how="left")
            df["CO2 (kt)"] = (
                pd.to_numeric(df["CO2_pc"], errors="coerce") * pd.to_numeric(df["Population"], errors="coerce")
            ) / 1000.0
            df["CO2_from_fallback"] = df["CO2 (kt)"].notna()

    return df


# =========================
# KAYA MODEL
# =========================

def compute_kaya(df):
    df = df.copy()

    # Expect columns: Population, GDP, Energy (per capita) or Energy_total
    if "GDP" not in df.columns or "Population" not in df.columns:
        raise ValueError("Missing Population or GDP for Kaya computation")

    # GDP per capita
    df["GDP_pc"] = df["GDP"] / df["Population"]

    # If Energy provided as per-capita (EG.USE.PCAP.KG.OE), reconstruct total
    if "EG.USE.PCAP.KG.OE" in df.columns:
        df.rename(columns={"EG.USE.PCAP.KG.OE": "Energy_per_capita"}, inplace=True)
        df["Energy_total"] = df["Energy_per_capita"] * df["Population"]
    elif "Energy per capita (kg oil eq)" in df.columns:
        df["Energy_total"] = df["Energy per capita (kg oil eq)"] * df["Population"]
    elif "Energy" in df.columns:
        df["Energy_total"] = df["Energy"] * df["Population"]
    else:
        # try any energy-like column
        energy_cols = [c for c in df.columns if "Energy" in c or c.startswith("EG.")]
        if energy_cols:
            df["Energy_total"] = pd.to_numeric(df[energy_cols[0]], errors="coerce") * df["Population"]
        else:
            raise ValueError("Missing Energy indicator for Kaya computation")

    # Intensities
    df["Energy_intensity"] = df["Energy_total"] / df["GDP"]

    # Use CO2 column if present (try both names)
    co2_col = None
    if "CO2 (kt)" in df.columns:
        co2_col = "CO2 (kt)"
    elif "CO2" in df.columns:
        co2_col = "CO2"

    if co2_col is None:
        raise ValueError("Missing CO2 indicator for Kaya computation")

    df["Carbon_intensity"] = pd.to_numeric(df[co2_col], errors="coerce") / df["Energy_total"]

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
    years = st.slider("Years", 1960, 2022, (1990, 2020))

country_code = COUNTRIES[country_name]

st.markdown("""Select which indicators to load. By default a curated set of climate & energy indicators
from the World Bank WDI is selected. You can also paste additional WDI indicator codes (comma-separated)
into the text box and click 'Add indicators'.""")

# Multiselect for indicators (human names)
indicator_names = list(INDICATORS.keys())
selected = st.multiselect("Indicators to fetch", indicator_names, default=indicator_names)

# Text input for additional raw indicator codes (comma-separated). Format: CODE or CODE:Name
extra_raw = st.text_input("Extra WDI codes (comma-separated). Optionally use CODE:Name to provide a display name.")
if st.button("Add indicators") and extra_raw.strip():
    parts = [p.strip() for p in extra_raw.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            code, name = p.split(":", 1)
            INDICATORS[name.strip()] = code.strip()
            selected.append(name.strip())
        else:
            # use code as name if no name provided
            code = p
            INDICATORS[code] = code
            selected.append(code)
    st.success("Added extra indicators to the selection. Click 'Load data' to fetch.")

# Button to load the selected data (avoids heavy startup network activity)
if st.button("Load data"):
    with st.spinner("Fetching indicators from World Bank..."):
        to_fetch = {name: INDICATORS[name] for name in selected}
        df = get_data(country_code, years, to_fetch)

        if df.empty:
            st.error("No data available for the selected country and years. Try expanding the year range or check indicators.")
            st.stop()

        # Notify user if CO2 fallback used
        if "CO2_from_fallback" in df.columns and df["CO2_from_fallback"].any():
            st.info("CO2 values were reconstructed from per-capita indicator because the primary CO2 indicator was not available for some years.")

        # Compute KAYA if possible, otherwise show partial results
        try:
            df_kaya = compute_kaya(df)
            kaya_ok = True
        except Exception as e:
            st.warning(f"Kaya computation not possible: {e}. Showing available variables instead.")
            kaya_ok = False

        # Show raw data
        with st.expander("🔍 Show raw data"):
            st.dataframe(df)

        # Show available plots (for selected variables)
        st.subheader("📊 Available Variables")
        # choose numeric columns to plot
        numeric_cols = [c for c in df.columns if c != "year" and pd.api.types.is_numeric_dtype(df[c])]
        for col in ["Population", "GDP", "GDP_pc", "Energy_total", "Energy_per_capita", "CO2 (kt)"]:
            if col in df.columns or col in numeric_cols:
                try:
                    st.plotly_chart(plot_variable(df, col if col in df.columns else col), use_container_width=True)
                except Exception:
                    pass

        # If Kaya computed, show Kaya outputs
        if kaya_ok:
            st.subheader("📊 Kaya Factors")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_variable(df_kaya, "Population"), use_container_width=True)
                st.plotly_chart(plot_variable(df_kaya, "GDP_pc"), use_container_width=True)
            with col2:
                st.plotly_chart(plot_variable(df_kaya, "Energy_intensity"), use_container_width=True)
                st.plotly_chart(plot_variable(df_kaya, "Carbon_intensity"), use_container_width=True)

            st.subheader("📈 Normalized Trends")
            variable = st.selectbox(
                "Select variable",
                ["Population", "GDP_pc", "Energy_intensity", "Carbon_intensity"]
            )
            st.plotly_chart(plot_normalized(df_kaya, variable), use_container_width=True)

            st.subheader("🧮 Kaya Identity Validation")
            fig = px.line(
                df_kaya,
                x="year",
                y=["CO2 (kt)", "CO2_reconstructed"],
                title="Actual vs Reconstructed CO₂"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Load data' to fetch selected indicators from the World Bank for the chosen country and year range.")

# End of file
