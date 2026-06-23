import streamlit as st
import requests
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import itertools

st.set_page_config(layout="wide")

BASE_YEAR = 1990
FINAL_YEAR = 2022 

# =========================
# INDICATORS
# =========================

INDICATORS = {
    "Population": ("SP.POP.TOTL", "MPax", 1e6),
    "GDP per capita": ("NY.GDP.PCAP.PP.CD", "k$/pax", 1e3),
    "Energy intensity": ("EG.GDP.PUSE.KO.PP", "Wh/$", 1),
    "CO2 emissions": ("EN.GHG.ALL.MT.CE.AR5", "MtCO2/y", 1e9)
}

# NOTE: ENTITIES will be loaded dynamically from World Bank API below.
ENTITIES = {
    "World": "WLD",
    "China": "CHN",
    "India": "IND",
    "United States": "USA",
    "European Union": "EUU",
    "Russian Federation": "RUS"
}

# default COLORS for common countries; additional colors will be generated dynamically
COLORS = {
    "World": "red",
    "China": "blue",
    "India": "magenta",
    "United States": "green",
    "European Union": "cyan",
    "Russian Federation": "yellow"
}

# =========================
# FETCH / COUNTRIES LIST
# =========================

@st.cache_data(ttl=24*3600)
def fetch_countries(include_aggregates=True):
    """
    Fetch the list of countries and aggregates from World Bank API.
    Returns a dict mapping display name -> ISO3 / aggregate code.
    If include_aggregates is False, aggregate entries (region == 'Aggregates') are filtered out.
    """
    url = "https://api.worldbank.org/v2/country?per_page=1000&format=json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # fallback to the minimal ENTITIES defined above
        return ENTITIES.copy()

    if not isinstance(data, list) or len(data) < 2:
        return ENTITIES.copy()

    countries = {}
    for item in data[1]:
        # item contains: id (ISO3 or aggregate code), name, region, incomeLevel, etc.
        cid = item.get("id")
        name = item.get("name")
        region = item.get("region", {}).get("value", "")

        if not cid or not name:
            continue

        if region == "Aggregates" and not include_aggregates:
            continue

        # Some names contain commas or parentheses; keep the original WB name for clarity
        countries[name] = cid

    # If for some reason the list is empty, use the hard-coded ENTITIES
    if not countries:
        return ENTITIES.copy()

    return countries


# =========================
# FETCH INDICATOR DATA
# =========================

def fetch(entity, code, years):
    url = f"https://api.worldbank.org/v2/country/{entity}/indicator/{code}?date={years[0]}:{years[1]}&format=json&per_page=1000"
    try:
        r = requests.get(url)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return pd.DataFrame()

    if len(payload) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(payload[1])[ ["date", "value"] ]
    df.columns = ["year", "value"]
    df["year"] = df["year"].astype(int)
    return df.sort_values("year")


def build_df(code, entities, years):
    df_all = pd.DataFrame()
    for e in entities:
        # entities here are the display names; ENTITIES_MAP maps name -> iso3
        iso = ENTITIES.get(e) if isinstance(ENTITIES, dict) else None
        if iso is None:
            # try to use entities mapping passed externally
            iso = entities.get(e) if isinstance(entities, dict) else None

        # If iso still None, assume e is an ISO code already
        if iso is None:
            iso = e

        df = fetch(iso, code, years)
        if df.empty:
            continue

        df = df.rename(columns={"value": e})

        if df_all.empty:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, on="year", how="outer")

    return df_all


# =========================
# DERIVED
# =========================

def derive(df_pop, df_gdppc, df_energy_int, df_CO2):

    df_gdp = df_pop.copy()
    for c in df_pop.columns:
        if c == "year": continue
        df_gdp[c] = df_pop[c] * df_gdppc[c]

    df_en_int = df_energy_int.copy()
    for c in df_energy_int.columns:
        if c == "year": continue
        df_en_int[c] = 1 / df_energy_int[c] * 11630 
    
    df_energy = df_gdp.copy()
    for c in df_gdp.columns:
        if c == "year": continue
        df_energy[c] = df_gdp[c] * df_en_int[c] / 11630 / 1e9

    df_CO2_intensity = df_CO2.copy()
    for c in df_gdp.columns:
        if c == "year": continue
        df_CO2_intensity[c] = df_CO2[c] / df_energy[c]

    
    return df_gdp, df_en_int, df_energy, df_CO2_intensity


# =========================
# NORMALIZATION
# =========================

def normalize(df, years):
    df_n = df.copy()
    for c in df.columns:
        if c == "year": continue
        base = df[df.year == years[0]][c]
        if not base.empty and base.values[0] != 0:
            df_n[c] = df[c] / base.values[0]
    return df_n


def growth(df):
    df_g = df.copy()
    for c in df.columns:
        if c == "year": continue
        df_g[c] = df[c].pct_change() * 100
    return df_g


# =========================
# LEGEND FINAL VALUES
# =========================

def build_labels(df, unit, factor, years):

    labels = {}

    for c in df.columns:
        if c == "year": continue

        val = df[df.year <= years[1]][c].dropna()
        if len(val) > 0:
            v = val.iloc[-1] / factor

            if v > 100:
                labels[c] = f"{c} ({v:.0f} {unit})"
            else:
                labels[c] = f"{c} ({v:.2g} {unit})"
        else:
            labels[c] = c

    return labels


# =========================
# PLOT
# =========================

def plot(df, title, unit="", factor=1, note=None):

    if note:
        st.warning(note)

    labels = build_labels(df, unit, factor, years)

    df_n = normalize(df, years)
    df_g = growth(df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    for c in df.columns:
        if c == "year": continue

        lw = 4 if c == "World" else 2

        fig.add_trace(
            go.Scatter(
                x=df_n.year,
                y=df_n[c],
                name=labels[c],
                line=dict(color=COLORS.get(c, "black"), width=lw)
            ),
            row=1, col=1
        )

    for c in df.columns:
        if c == "year": continue

        lw = 2 if c == "World" else 1

        fig.add_trace(
            go.Scatter(
                x=df_g.year,
                y=df_g[c],
                showlegend=False,
                line=dict(color=COLORS.get(c, "black"), width=lw)
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=title,
        height=750,
        legend=dict(orientation="h", y=1.1)
    )

    fig.update_yaxes(title_text="Normalized (1990 = 1)", row=1, col=1)
    fig.update_yaxes(title_text="Growth rate [%/year]", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)

    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig


# =========================
# TEMPERATURE METRICS FROM CO2
# =========================

def compute_temperature_metrics(df_co2, years, co2_to_gtoe_factor=7.82, temp_divisor=120.0):
    """
    Compute for each column in df_co2 (except 'year'):
      - integrated_gtoe: trapezoidal integral of (CO2 [MtCO2/y] / co2_to_gtoe_factor) over the selected years -> total Gtoe
      - temp_increase: integrated_gtoe / temp_divisor  (estimated temperature increase over period)
      - final_derivative: last derivative (°C/year) of the cumulative temperature time series
    Returns a DataFrame indexed by country/entity with those three values.
    """
    results = []

    for c in df_co2.columns:
        if c == "year":
            continue

        s = df_co2[["year", c]].dropna()
        if s.empty:
            results.append({
                "entity": c,
                "integrated_gtoe": np.nan,
                "temp_increase_C": np.nan,
                "final_derivative_C_per_year": np.nan
            })
            continue

        # mask to selected years (ensure contiguous years used)
        mask = (s["year"] >= years[0]) & (s["year"] <= years[1])
        ssel = s[mask].sort_values("year")
        ya = ssel["year"].values
        co2_mt = ssel[c].values  # MtCO2 / year

        if len(ya) < 2 or np.all(np.isnan(co2_mt)):
            results.append({
                "entity": c,
                "integrated_gtoe": np.nan,
                "temp_increase_C": np.nan,
                "final_derivative_C_per_year": np.nan
            })
            continue

        # convert to Gtoe/y
        gtoe_per_year = co2_mt / co2_to_gtoe_factor

        # integrate over years (Gtoe) using trapezoidal rule
        integrated_gtoe = np.trapz(gtoe_per_year, ya)

        # temperature increase estimate
        temp_increase = integrated_gtoe / temp_divisor

        # build cumulative temperature time series (cum integral up to each year)
        cum_integral = np.array([np.trapz(gtoe_per_year[:i+1], ya[:i+1]) for i in range(len(ya))])
        temp_ts = cum_integral / temp_divisor  # °C time series

        # compute derivative (°C / year)
        deriv = np.gradient(temp_ts, ya)
        final_derivative = deriv[-1]

        results.append({
            "entity": c,
            "integrated_gtoe": integrated_gtoe,
            "temp_increase_C": temp_increase,
            "final_derivative_C_per_year": final_derivative
        })

    df_res = pd.DataFrame(results).set_index("entity")
    return df_res


# =========================
# UI
# =========================

st.title("🌍 Kaya Identity - Data elaboration from World Development Indicators Dataset - World Bank Group")

# allow user to include aggregates (regions, income groups) from WB
include_aggregates = st.checkbox("Include aggregates and income/institution groups (regions, income groups)", value=True)

# fetch the available entities (countries + optional aggregates)
entities_map = fetch_countries(include_aggregates=include_aggregates)

# sort for presentation
sorted_names = sorted(list(entities_map.keys()))

# provide a sensible default selection if available
default_selection = [n for n in ["World", "China", "India", "United States", "European Union", "Russian Federation"] if n in entities_map]

entities = st.multiselect(
    "Entities",
    sorted_names,
    default=default_selection
)

# slider for years (keeps previous defaults)
years = st.slider("Years", 1990, 2022, (1990, 2020))

# Build a mapping used by build_df: name -> iso
ENTITIES = {name: entities_map[name] for name in entities_map}

# Build a color map for all entries (so plotting functions can access COLORS[c])
palette = px.colors.qualitative.Plotly
# If more colors needed, extend by cycling through palette and adding variations
colors_cycle = itertools.cycle(palette)
COLORS = {}
for name in sorted_names:
    # keep previous explicit colors for familiar names if present
    if name in COLORS:
        continue
    COLORS[name] = next(colors_cycle)

# Ensure the earlier well-known entries keep their chosen colors (override if present)
COLORS.update({
    "World": "red",
    "China": "blue",
    "India": "magenta",
    "United States": "green",
    "European Union": "cyan",
    "Russian Federation": "yellow"
})


# =========================
# RUN
# =========================

if st.button("Generate Figures"):

    # if no entities selected, warn and stop
    if not entities:
        st.warning("Please select at least one entity to fetch data.")
    else:
        # base
        df_pop = build_df(INDICATORS["Population"][0], entities_map if False else entities, years)
        df_gdppc = build_df(INDICATORS["GDP per capita"][0], entities_map if False else entities, years)
        df_energy_int = build_df(INDICATORS["Energy intensity"][0], entities_map if False else entities, years)
        df_CO2 = build_df(INDICATORS["CO2 emissions"][0], entities_map if False else entities, years)

        # derived
        df_gdp, df_en_int, df_energy, df_CO2_intensity = derive(df_pop, df_gdppc, df_energy_int, df_CO2)

        # --- compute temperature metrics from CO2 ---
        # Using the requested factors: divide CO2 by 7.82 to get Gtoe/y, then integrated / 120 => temperature increase [°C]
        temp_metrics = compute_temperature_metrics(df_CO2, years, co2_to_gtoe_factor=7.82, temp_divisor=120.0)

        # Format and show table
        if not temp_metrics.empty:
            display_df = temp_metrics.copy()
            # nice rounding
            display_df["integrated_gtoe"] = display_df["integrated_gtoe"].map(lambda x: np.round(x, 3) if pd.notna(x) else x)
            display_df["temp_increase_C"] = display_df["temp_increase_C"].map(lambda x: np.round(x, 4) if pd.notna(x) else x)
            display_df["final_derivative_C_per_year"] = display_df["final_derivative_C_per_year"].map(lambda x: np.round(x, 6) if pd.notna(x) else x)

            st.subheader("CO2-based temperature metrics (selected entities)")
            st.write("Method: CO2 [MtCO2/y] -> divide by 7.82 -> Gtoe/y; integrate over period -> total Gtoe; ΔT = total_Gtoe / 120. Final derivative is last-year slope of ΔT curve (°C/year).")
            st.table(display_df)

        # plots
        st.plotly_chart(plot(df_pop, "A — Population", "MPax", 1e6), use_container_width=True)
        st.plotly_chart(plot(df_gdppc, "B — GDP (PPP) per capita", "k$/pax", 1e3), use_container_width=True)
        st.plotly_chart(plot(df_gdp, "Ec — GDP dynamics", "G$/y", 1e9), use_container_width=True)
        st.plotly_chart(plot(df_en_int, "C — Energy intensity", "Wh/$", 1), use_container_width=True)
        st.plotly_chart(plot(df_energy, "En — Energy consumption", "Mtoe/y", 1), use_container_width=True)
        st.plotly_chart(plot(df_CO2_intensity, "D — Emission intensity", "gCO2/kWh", 1e-3), use_container_width=True)
        st.plotly_chart(plot(df_CO2, "Em — CO₂ emissions", "MtCO2/y", 1), use_container_width=True)
