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

# Top ~60 countries by population (display name -> ISO3). This list is a convenience subset.
TOP60 = {
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
}

# Predefined groups (each group maps to a list of ISO3 codes drawn from TOP60 when appropriate)
GROUPS = {
    "World (top60 subset)": list(TOP60.values()),
    "European Union (subset)": ["DEU", "FRA", "ITA", "ESP", "POL"],
    "Africa (subset)": [c for c in TOP60.values() if c in {"NGA", "EGY", "ZAF", "DZA", "SDN", "MAR", "MOZ", "CMR", "CIV", "UGA", "KEN", "TZA", "MDG"}],
    "Asia (subset)": [c for c in TOP60.values() if c in {"CHN", "IND", "IDN", "PAK", "BGD", "VNM", "MMR", "THA", "KOR", "IRN", "SAU", "TWN", "SYR", "UZB"}],
    "Latin America (subset)": [c for c in TOP60.values() if c in {"BRA", "MEX", "COL", "ARG", "PER", "VEN"}],
    "Oceania (subset)": ["AUS"],
}

# Conversion constants (kept for potential group-weighting using Population)

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
    """Return merged DataFrame of selected indicators for the given country and year range."""
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


def aggregate_group_members(dfs, group_name):
    """Aggregate a list of country DataFrames (each with year + indicators) into a single group DataFrame.

    Heuristic:
    - For per-capita or percent indicators (detected by keywords), compute population-weighted average when Population is available.
    - Otherwise sum numeric indicators across members.
    """
    if not dfs:
        return pd.DataFrame()

    # concat with country label if present
    all_df = pd.concat(dfs, ignore_index=True, sort=False)
    # ensure year is int
    all_df["year"] = all_df["year"].astype(int)

    numeric_cols = [c for c in all_df.columns if c != "year" and c != "country"]

    result_rows = []
    for year, group in all_df.groupby("year"):
        row = {"year": int(year)}
        # population sum for weighting
        pop_sum = None
        if "Population" in numeric_cols:
            pop_sum = pd.to_numeric(group["Population"], errors="coerce").sum()
        for col in numeric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            if vals.dropna().empty:
                row[col] = None
                continue
            # detect per-capita / percent indicators by name
            lname = col.lower()
            if ("per" in lname) or ("pc" in lname) or ("pct" in lname) or ("per_capita" in lname) or ("%" in lname) or ("_pc" in lname):
                # try population-weighted average if population exists
                if "Population" in group.columns and pop_sum and pop_sum > 0:
                    weights = pd.to_numeric(group["Population"], errors="coerce")
                    valid = (~vals.isna()) & (~weights.isna())
                    if valid.any():
                        weighted = (vals[valid] * weights[valid]).sum() / weights[valid].sum()
                        row[col] = float(weighted)
                        continue
                # fallback to simple mean
                row[col] = float(vals.mean())
            else:
                # sum totals
                row[col] = float(vals.sum())
        result_rows.append(row)

    if not result_rows:
        return pd.DataFrame()
    out = pd.DataFrame(result_rows).sort_values("year")
    return out


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

st.title("🌍 Kaya Explorer — focused indicators, 4-country selection + groups")

st.markdown("This view automatically fetches the full focused set of indicators and graphs them for the selected countries and groups.")

col1, col2 = st.columns(2)
with col1:
    selected_countries = st.multiselect("Select up to 4 countries (from top ~60 by population)", list(TOP60.keys()), default=["United States", "China"])
with col2:
    years = st.slider("Years", 1960, 2022, (1990, 2020))

group_options = list(GROUPS.keys())
col3, col4 = st.columns(2)
with col3:
    group_a = st.selectbox("Group A (optional)", ["None"] + group_options, index=0)
with col4:
    group_b = st.selectbox("Group B (optional)", ["None"] + group_options, index=0)

# Enforce max 4 countries
if len(selected_countries) > 4:
    st.error("Please select at most 4 countries.")
    st.stop()

# Always fetch all indicators (no per-indicator selection)
ind_map = INDICATORS.copy()

if st.button("Load and plot all indicators"):
    if not selected_countries and group_a == "None" and group_b == "None":
        st.error("Choose at least one country or group to visualize.")
        st.stop()

    start_year, end_year = years
    entities_dfs = []  # list of tuples (label, df)

    with st.spinner("Fetching data for selected countries..."):
        # fetch for each selected country
        for cname in selected_countries:
            ccode = TOP60.get(cname, cname)
            df = get_data(ccode, (start_year, end_year), ind_map)
            if df.empty:
                st.warning(f"No data for {cname} in the selected range.")
                continue
            df["country"] = cname
            entities_dfs.append((cname, df))

        # handle groups by aggregating their members
        for g_label in (group_a, group_b):
            if not g_label or g_label == "None":
                continue
            members = GROUPS.get(g_label, [])
            member_dfs = []
            for iso3 in members:
                dfm = get_data(iso3, (start_year, end_year), ind_map)
                if dfm.empty:
                    continue
                dfm["country"] = iso3
                member_dfs.append(dfm)
            if not member_dfs:
                st.warning(f"No member data available for group {g_label} in the selected range.")
                continue
            agg = aggregate_group_members(member_dfs, g_label)
            if agg.empty:
                st.warning(f"Aggregation produced no data for {g_label}.")
                continue
            entities_dfs.append((g_label, agg))

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
    # For each indicator, plot comparison across entities
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
    st.info("Click 'Load and plot all indicators' to fetch the focused WDI indicators for the chosen countries and groups.")
