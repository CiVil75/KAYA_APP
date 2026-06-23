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

def compute_temperature_metrics(df_co2, years, mt_to_gt_factor=1000.0, gt_to_ppm_factor=7.82, ppm_to_temp_factor=120.0):
    """
    Compute for each column in df_co2 (except 'year') the following chain:
      - cumuklative_GtCO2: integrated_MtCO2 / mt_to_gt_factor -> total GtCO2
      - resident_ppm: integrated_GtCO2 / gt_to_ppm_factor -> ppm equivalent
      - temp_increase_C: resident_ppm / ppm_to_temp_factor -> estimated temperature increase [°C]
      - final_derivative_C_per_year: last derivative (°C/year) of the cumulative temperature time series

    Returns a DataFrame indexed by country/entity with those values.
    """
    results = []

    for c in df_co2.columns:
        if c == "year":
            continue

        s = df_co2[["year", c]].dropna()
        if s.empty:
            results.append({
                "entity": c,
                "integrated_GtCO2": np.nan,
                "resident_ppm": np.nan,
                "temp_increase_C": np.nan,
                "final_derivative_C_per_year": np.nan
            })
            continue

        # mask to selected years
        mask = (s["year"] >= years[0]) & (s["year"] <= years[1])
        ssel = s[mask].sort_values("year")
        ya = ssel["year"].values
        co2_mt = ssel[c].values  # MtCO2 / year

        if len(ya) < 2 or np.all(np.isnan(co2_mt)):
            results.append({
                "entity": c,
                "integrated_GtCO2": np.nan,
                "resident_ppm": np.nan,
                "temp_increase_C": np.nan,
                "final_derivative_C_per_year": np.nan
            })
            continue

        # mask out NaNs
        mask_vals = ~np.isnan(co2_mt) & ~np.isnan(ya)
        x = ya[mask_vals]
        y_mt = co2_mt[mask_vals]

        if len(x) < 2:
            integrated_Gt = np.nan
            resident_ppm = np.nan
            temp_increase = np.nan
            final_derivative = np.nan
        else:
            # integrate CO2 (MtCO2) over time -> total MtCO2
            integrated_Mt = np.sum((y_mt[:-1] + y_mt[1:]) * (x[1:] - x[:-1]) / 2.0)

            # convert to GtCO2
            integrated_Gt = integrated_Mt / mt_to_gt_factor

            # resident ppm equivalent
            resident_ppm = integrated_Gt / gt_to_ppm_factor

            # temperature increase estimate
            temp_increase = resident_ppm / ppm_to_temp_factor

            # build cumulative integral (MtCO2 up to each available year)
            cum_Mt = np.zeros(len(x))
            for i in range(1, len(x)):
                cum_Mt[i] = cum_Mt[i-1] + (y_mt[i-1] + y_mt[i]) * (x[i] - x[i-1]) / 2.0

            cum_Gt = cum_Mt / mt_to_gt_factor
            cum_ppm = cum_Gt / gt_to_ppm_factor
            temp_ts = cum_ppm / ppm_to_temp_factor  # °C time series

            # compute derivative (°C / year)
            deriv = np.gradient(temp_ts, x)
            final_derivative = deriv[-1]

        results.append({
            "entity": c,
            "integrated_GtCO2": integrated_Gt,
            "resident_ppm": resident_ppm,
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
        # base (for plotting) - use only user-selected entities
        df_pop = build_df(INDICATORS["Population"][0], entities, years)
        df_gdppc = build_df(INDICATORS["GDP per capita"][0], entities, years)
        df_energy_int = build_df(INDICATORS["Energy intensity"][0], entities, years)
        df_CO2 = build_df(INDICATORS["CO2 emissions"][0], entities, years)

        # derived
        df_gdp, df_en_int, df_energy, df_CO2_intensity = derive(df_pop, df_gdppc, df_energy_int, df_CO2)

        # Determine display name for World (if present in entities_map)
        world_name = None
        for n, iso in entities_map.items():
            if iso == "WLD" or n.lower() == "world":
                world_name = n
                break

        # --- compute temperature metrics from CO2 ---
        # Ensure World is included in the metrics table even if not selected
        entities_for_metrics = list(entities)
        if world_name and world_name not in entities_for_metrics:
            entities_for_metrics.append(world_name)

        df_CO2_metrics = build_df(INDICATORS["CO2 emissions"][0], entities_for_metrics, years)

        temp_metrics = compute_temperature_metrics(df_CO2_metrics, years, mt_to_gt_factor=1000.0, gt_to_ppm_factor=7.82, ppm_to_temp_factor=120.0)

        # plots (keep these before the table per your request)
        st.plotly_chart(plot(df_pop, "A — Population", "MPax", 1e6), use_container_width=True)
        st.plotly_chart(plot(df_gdppc, "B — GDP (PPP) per capita", "k$/pax", 1e3), use_container_width=True)
        st.plotly_chart(plot(df_gdp, "Ec — GDP dynamics", "G$/y", 1e9), use_container_width=True)
        st.plotly_chart(plot(df_en_int, "C — Energy intensity", "Wh/$", 1), use_container_width=True)
        st.plotly_chart(plot(df_energy, "En — Energy consumption", "Mtoe/y", 1), use_container_width=True)
        st.plotly_chart(plot(df_CO2_intensity, "D — Emission intensity", "gCO2/kWh", 1e-3), use_container_width=True)
        st.plotly_chart(plot(df_CO2, "Em — CO₂ emissions", "MtCO2/y", 1), use_container_width=True)

        # Format and show table AFTER plots
        if not temp_metrics.empty:
            display_df = temp_metrics.copy()

            # Compute percentages relative to World values (if World exists)
            if world_name and world_name in display_df.index:
                world_vals = display_df.loc[world_name]
                pct_cols = {}
                for col in ["integrated_GtCO2", "resident_ppm", "temp_increase_C", "final_derivative_C_per_year"]:
                    w = world_vals.get(col, np.nan)
                    if pd.notna(w) and w != 0:
                        display_df[f"pct_of_world_{col}"] = display_df[col] / w * 100
                    else:
                        display_df[f"pct_of_world_{col}"] = np.nan
            else:
                # no world values available
                for col in ["integrated_GtCO2", "resident_ppm", "temp_increase_C", "final_derivative_C_per_year"]:
                    display_df[f"pct_of_world_{col}"] = np.nan

            # rounding and ordering
            display_df["integrated_GtCO2"] = display_df["integrated_GtCO2"].map(lambda x: np.round(x, 4) if pd.notna(x) else x)
            display_df["resident_ppm"] = display_df["resident_ppm"].map(lambda x: np.round(x, 4) if pd.notna(x) else x)
            display_df["temp_increase_C"] = display_df["temp_increase_C"].map(lambda x: np.round(x, 4) if pd.notna(x) else x)
            display_df["final_derivative_C_per_year"] = display_df["final_derivative_C_per_year"].map(lambda x: np.round(x, 6) if pd.notna(x) else x)

            for col in [f"pct_of_world_{c}" for c in ["integrated_MtCO2", "integrated_GtCO2", "resident_ppm", "temp_increase_C", "final_derivative_C_per_year"]]:
                display_df[col] = display_df[col].map(lambda x: np.round(x, 2) if pd.notna(x) else x)

            # ensure World row is present and show world first
            if world_name and world_name in display_df.index:
                # reorder so World appears first, then selected entities (unique)
                ordered = [world_name] + [e for e in entities_for_metrics if e != world_name and e in display_df.index]
                display_df = display_df.reindex(ordered)

            st.subheader("CO2-based temperature metrics (selected entities)")
            st.write("Method: integrate CO2 [MtCO2/y] -> total MtCO2; convert to GtCO2 (/1000); convert to atmospheric ppm (divide by 7.82); ΔT = ppm / 120. Percent columns show each value as % of the World value.")

            # show a subset of columns in a sensible order
            cols_order = [
                "integrated_GtCO2", "resident_ppm", "temp_increase_C", "final_derivative_C_per_year",
                "pct_of_world_final_derivative_C_per_year"
            ]

            # some columns may be missing if metrics couldn't be computed; keep only existing
            cols_order = [c for c in cols_order if c in display_df.columns]

            st.table(display_df[cols_order])

        else:
            st.warning("No temperature metrics computed — check data availability for the selected entities and years.")
