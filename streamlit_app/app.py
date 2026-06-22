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
    "CO2 per unit energy (kg per kg oil eq)": "EN.ATM.CO2E.EG.ZS",
    "CO2 from solid fuel (% of total)": "EN.ATM.CO2E.SF.ZS",
    "CO2 from liquid fuel (% of total)": "EN.ATM.CO2E.LF.ZS",
    "CO2 from gaseous fuel (% of total)": "EN.ATM.CO2E.GF.ZS",
    "Transport CO2 (% of fuel)": "EN.CO2.TRAN.ZS",
    "Electricity/heat CO2 (% of fuel)": "EN.CO2.ETOT.ZS",
    "Industry CO2 (% of fuel)": "EN.CO2.MANF.ZS",
    "CO2 per unit of GDP (kg per 2015 PPP $)": "EN.ATM.CO2E.KD.GD",
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

# Conversion constants (to kg oil equivalent)
# Source / rationale summarized in the UI documentation expander
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


def _detect_energy_unit(code: str, name: str):
    """Heuristic detection of energy units from WDI code or human name.

    Returns one of: 'kg_per_capita', 'ktoe', 'toe', 'mwh', 'kwh', 'gwh', 'gj', 'toe_per_capita', 'mwh_per_capita', or None
    """
    code_l = code.lower() if isinstance(code, str) else ""
    name_l = name.lower() if isinstance(name, str) else ""

    # per-capita in kg oil-equivalent
    if "pcap" in code_l or "per capita" in name_l:
        if "kg" in code_l or "kg" in name_l or "kg oil" in name_l:
            return "kg_per_capita"
        # sometimes per-capita is given in toe per capita
        if "toe" in code_l or "toe" in name_l:
            return "toe_per_capita"
        if "mwh" in code_l or "mwh" in name_l:
            return "mwh_per_capita"

    # total ktoe / toe
    if "ktoe" in code_l or "kt.oe" in code_l or "ktoe" in name_l or "kto" in name_l:
        return "ktoe"
    if "toe" in code_l and "ktoe" not in code_l:
        return "toe"

    # electricity or energy in MWh/kWh/GWh
    if "mwh" in code_l or "mwh" in name_l:
        return "mwh"
    if "kwh" in code_l or "kwh" in name_l:
        return "kwh"
    if "gwh" in code_l or "gwh" in name_l:
        return "gwh"

    # gigajoule
    if "gj" in code_l or "gigajoule" in name_l:
        return "gj"

    return None


@st.cache_data
def get_data(country, year_range, indicators_map):
    """Return merged DataFrame of selected indicators for the given country and year range.

    indicators_map: dict mapping human name -> code
    The function will try multiple fallbacks to reconstruct absolute CO2 (in kt):
      1. Use EN.ATM.CO2E.KT directly if available
      2. Use EN.ATM.CO2E.PC (per-capita) and Population to compute kt
      3. Use EN.ATM.CO2E.EG.ZS (kg CO2 per kg oil-eq) and energy indicators (in various units) to compute kt
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

    # ---- Energy unit normalization ----
    # We try to produce these standardized columns if possible:
    # - energy_total_kg_oil_eq  (total energy in kg oil-equivalent)
    # - energy_per_capita_kg_oil_eq (per-person in kg oil-equivalent)

    energy_total = None
    energy_per_capita = None

    # First, look for explicit per-capita in kg oil eq
    for name, code in indicators_map.items():
        unit = _detect_energy_unit(code, name)
        if unit == "kg_per_capita" and name in df.columns:
            energy_per_capita = pd.to_numeric(df[name], errors="coerce")
            df["energy_per_capita_kg_oil_eq"] = energy_per_capita
            # total energy
            if "Population" in df.columns:
                df["energy_total_kg_oil_eq"] = energy_per_capita * pd.to_numeric(df["Population"], errors="coerce")
                energy_total = df["energy_total_kg_oil_eq"]
            break

    # If not found, look for total ktoe / toe / MWh / GWh / GJ / kWh
    if energy_total is None:
        for name, code in indicators_map.items():
            unit = _detect_energy_unit(code, name)
            if unit and name in df.columns:
                vals = pd.to_numeric(df[name], errors="coerce")
                if unit == "ktoe":
                    df["energy_total_kg_oil_eq"] = vals * KTOE_TO_KG_OIL_EQ
                    energy_total = df["energy_total_kg_oil_eq"]
                    break
                if unit == "toe":
                    df["energy_total_kg_oil_eq"] = vals * TOE_TO_KG_OIL_EQ
                    energy_total = df["energy_total_kg_oil_eq"]
                    break
                if unit == "mwh":
                    # treat as TOTAL MWh unless per-capita detected earlier
                    df["energy_total_kg_oil_eq"] = vals * MWH_TO_KG_OIL_EQ
                    energy_total = df["energy_total_kg_oil_eq"]
                    break
                if unit == "gwh":
                    df["energy_total_kg_oil_eq"] = vals * (MWH_TO_KG_OIL_EQ * 1000.0)
                    energy_total = df["energy_total_kg_oil_eq"]
                    break
                if unit == "kwh":
                    df["energy_total_kg_oil_eq"] = vals * KWH_TO_KG_OIL_EQ
                    energy_total = df["energy_total_kg_oil_eq"]
                    break
                if unit == "gj":
                    df["energy_total_kg_oil_eq"] = vals * GJ_TO_KG_OIL_EQ
                    energy_total = df["energy_total_kg_oil_eq"]
                    break

    # If we have per-capita in toe or MWh, convert to kg and multiply by population
    if ("energy_total_kg_oil_eq" not in df.columns) and ("Population" in df.columns):
        for name, code in indicators_map.items():
            unit = _detect_energy_unit(code, name)
            if unit and name in df.columns:
                vals = pd.to_numeric(df[name], errors="coerce")
                if unit == "toe_per_capita":
                    df["energy_per_capita_kg_oil_eq"] = vals * TOE_TO_KG_OIL_EQ
                    df["energy_total_kg_oil_eq"] = df["energy_per_capita_kg_oil_eq"] * pd.to_numeric(df["Population"], errors="coerce")
                    energy_total = df["energy_total_kg_oil_eq"]
                    break
                if unit == "mwh_per_capita":
                    df["energy_per_capita_kg_oil_eq"] = vals * MWH_TO_KG_OIL_EQ
                    df["energy_total_kg_oil_eq"] = df["energy_per_capita_kg_oil_eq"] * pd.to_numeric(df["Population"], errors="coerce")
                    energy_total = df["energy_total_kg_oil_eq"]
                    break

    # Normalize column names if we produced them
    if "energy_total_kg_oil_eq" in df.columns:
        # make numeric
        df["energy_total_kg_oil_eq"] = pd.to_numeric(df["energy_total_kg_oil_eq"], errors="coerce")
        # derive per-capita if missing
        if "energy_per_capita_kg_oil_eq" not in df.columns and "Population" in df.columns:
            df["energy_per_capita_kg_oil_eq"] = df["energy_total_kg_oil_eq"] / pd.to_numeric(df["Population"], errors="coerce")

    # ---- CO2 reconstruction fallbacks ----
    # 1) direct CO2 (kt) already present? do nothing
    if "CO2 (kt)" not in df.columns:
        # 2) try per-capita
        co2_pc_df = fetch_indicator(country, "EN.ATM.CO2E.PC", start, end)
        if not co2_pc_df.empty and "Population" in df.columns:
            co2_pc_df.rename(columns={"EN.ATM.CO2E.PC": "CO2_per_capita"}, inplace=True)
            df = df.merge(co2_pc_df, on="year", how="left")
            df["CO2 (kt)"] = (pd.to_numeric(df["CO2_per_capita"], errors="coerce") * pd.to_numeric(df["Population"], errors="coerce")) / 1000.0
            df["CO2_from_per_capita"] = df["CO2 (kt)"].notna()

    # 3) if still absent, try CO2 per unit energy (kg CO2 per kg oil-eq) and energy_total_kg_oil_eq
    if "CO2 (kt)" not in df.columns and "CO2 per unit energy (kg per kg oil eq)" in indicators_map.keys():
        if "CO2 per unit energy (kg per kg oil eq)" in df.columns and "energy_total_kg_oil_eq" in df.columns:
            df["CO2_kg_from_energy"] = pd.to_numeric(df["CO2 per unit energy (kg per kg oil eq)"], errors="coerce") * pd.to_numeric(df["energy_total_kg_oil_eq"], errors="coerce")
            df["CO2 (kt)"] = df["CO2_kg_from_energy"] / 1_000_000.0
            df["CO2_from_per_energy"] = df["CO2 (kt)"].notna()

    # Final: return DataFrame with whatever we could compute/merge; downstream code will check columns
    return df


# =========================
# KAYA MODEL
# =========================

def compute_kaya(df):
    df = df.copy()

    if "Population" not in df.columns or "GDP" not in df.columns:
        raise ValueError("Missing Population or GDP for Kaya computation")

    df["GDP_pc"] = pd.to_numeric(df["GDP"], errors="coerce") / pd.to_numeric(df["Population"], errors="coerce")

    # Use energy_total_kg_oil_eq if available
    if "energy_total_kg_oil_eq" in df.columns:
        df["Energy_total"] = df["energy_total_kg_oil_eq"]
    else:
        # try previous fallbacks
        if "Energy" in df.columns:
            df["Energy_total"] = pd.to_numeric(df["Energy"], errors="coerce") * pd.to_numeric(df["Population"], errors="coerce")
        else:
            raise ValueError("Missing Energy indicator for Kaya computation")

    df["Energy_intensity"] = df["Energy_total"] / pd.to_numeric(df["GDP"], errors="coerce")

    # Use CO2 (kt) converted to kg
    if "CO2 (kt)" in df.columns:
        df_co2_kg = pd.to_numeric(df["CO2 (kt)"], errors="coerce") * 1_000_000.0
    else:
        raise ValueError("Missing CO2 indicator for Kaya computation")

    df["Carbon_intensity"] = df_co2_kg / df["Energy_total"]

    df["CO2_reconstructed"] = (
        pd.to_numeric(df["Population"], errors="coerce")
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

    return px.line(d, x="year", y="normalized", title=f"{var} (normalized to {base_year})")


# =========================
# UI
# =========================

st.title("🌍 Kaya Identity Explorer")

# Documentazione conversioni unità energetiche e fonti (mostrata all'utente)
with st.expander("ℹ️ Conversioni unità energia e fonti (come vengono calcolate le ricostruzioni CO₂)"):
    st.markdown(
        """
        Questo pannello spiega le conversioni usate dall'app per trasformare indicatori di energia
        in una unità standard (kg oil‑equivalent) e come vengono ricostruite le emissioni di CO₂
        quando l'indicatore assoluto non è disponibile.

        Coefficienti principali usati (unità: kg oil‑equivalent):
        - 1 toe = 1 000 kg oil‑equivalent (TOE_TO_KG_OIL_EQ = 1000.0)
        - 1 ktoe = 1 000 toe = 1 000 000 kg oil‑equivalent (KTOE_TO_KG_OIL_EQ = 1e6)
        - 1 MWh ≈ 0.08598 toe → 85.98 kg oil‑eq per MWh (MWH_TO_KG_OIL_EQ = 85.98)
        - 1 kWh = 0.001 MWh → ≈ 0.08598 kg oil‑eq per kWh (KWH_TO_KG_OIL_EQ ≈ 0.08598)
        - 1 GJ ≈ 0.0239 toe → ≈ 23.9 kg oil‑eq per GJ (GJ_TO_KG_OIL_EQ ≈ 23.9)

        Come vengono ricostruite le emissioni CO₂ (ordine dei fallback):
        1) EN.ATM.CO2E.KT (CO₂ totale in kiloton) → usato direttamente se disponibile.
        2) EN.ATM.CO2E.PC (t CO₂ per persona) → CO₂_tot_kt = (CO2_per_capita [t/person] * Population) / 1000.
        3) EN.ATM.CO2E.EG.ZS (kg CO₂ per kg oil‑eq) + indicatore energia → 
           CO2_kg_total = CO2_per_energy (kg/kg_oil_eq) * energy_total_kg_oil_eq;
           CO2_kt = CO2_kg_total / 1e6.

        Note e avvertenze:
        - Le equivalenze toe↔MWh↔GJ si basano su valori standard IEA/World Bank (arrotondati per praticità).
        - La conversione da "per unità di energia" richiede che l'indicatore energia sia
          espresso in unità identificabili (ktoe, toe, MWh, kWh, GWh, GJ o per-capita). Il codice
          tenta di rilevare l'unità dalla sigla/codice dell'indicatore; se l'unità non è chiara
          i risultati vanno verificati manualmente usando i metadata WDI.
        - Le ricostruzioni sono stime: consigliabile verificare qualitativamente i valori (es. confrontando
          con altre fonti) prima di usarli per analisi decisionali.

        Fonti e riferimenti:
        - World Bank — World Development Indicators (WDI): https://databank.worldbank.org/source/world-development-indicators
        - Note energetiche/IEA: conversione toe ↔ MWh ↔ GJ (IEA / definizioni energetiche standard)
        - Documentazione WDI per singoli indicatori: https://datahelpdesk.worldbank.org/knowledgebase/articles/906519
        """
    )

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
            code = p
            INDICATORS[code] = code
            selected.append(code)
    st.success("Added extra indicators to the selection. Click 'Load data' to fetch.")

# Availability: compute on demand inside the expander
with st.expander("📚 Indicator availability for selected country & range"):
    if st.button("Compute availability"):
        start_year, end_year = years
        availability = []
        missing_indicators = []
        with st.spinner("Checking indicator availability..."):
            for name, code in INDICATORS.items():
                yrs = get_indicator_years(country_code, code, start_year, end_year)
                if yrs:
                    yrs_str = f"{min(yrs)}-{max(yrs)} ({len(yrs)} years)"
                else:
                    yrs_str = "No data"
                    missing_indicators.append(name)
                availability.append({"Indicator": name, "Code": code, "Available": yrs_str})
        avail_df = pd.DataFrame(availability)
        st.table(avail_df)
        if missing_indicators:
            st.warning(
                f"Non tutti gli indicatori sono disponibili per il paese/intervallo selezionato: {', '.join(missing_indicators)}. "
                "La validazione della Kaya richiede alcuni indicatori; prova ad ampliare l'intervallo di anni o a selezionare un altro paese."
            )

# Button to load the selected data (avoids heavy startup network activity)
if st.button("Load data"):
    with st.spinner("Fetching indicators from World Bank..."):
        to_fetch = {name: INDICATORS[name] for name in selected}
        df = get_data(country_code, years, to_fetch)

        if df.empty:
            st.error("No data available for the selected country and years. Try expanding the year range or check indicators.")
            st.stop()

        # Notify user if CO2 fallback used
        if "CO2_from_per_capita" in df.columns and df["CO2_from_per_capita"].any():
            st.info("CO2 values were reconstructed from per-capita indicator (EN.ATM.CO2E.PC).")
        if "CO2_from_per_energy" in df.columns and df["CO2_from_per_energy"].any():
            st.info("CO2 values were reconstructed from CO2 per unit energy (EN.ATM.CO2E.EG.ZS) and energy indicators.")

        # Attempt Kaya
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
        numeric_cols = [c for c in df.columns if c != "year" and pd.api.types.is_numeric_dtype(df[c])]
        for col in ["Population", "GDP", "GDP_pc", "energy_total_kg_oil_eq", "energy_per_capita_kg_oil_eq", "CO2 (kt)"]:
            if col in df.columns or col in numeric_cols:
                try:
                    st.plotly_chart(plot_variable(df, col if col in df.columns else col), use_container_width=True)
                except Exception:
                    pass

        # If Kaya computed, show Kaya outputs
        if kaya_ok:
            st.subheader("📊 Kaya Factors")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_variable(df_kaya, "Population"), use_container_width=True)
                st.plotly_chart(plot_variable(df_kaya, "GDP_pc"), use_container_width=True)
            with c2:
                st.plotly_chart(plot_variable(df_kaya, "Energy_intensity"), use_container_width=True)
                st.plotly_chart(plot_variable(df_kaya, "Carbon_intensity"), use_container_width=True)

            st.subheader("📈 Normalized Trends")
            variable = st.selectbox("Select variable", ["Population", "GDP_pc", "Energy_intensity", "Carbon_intensity"])
            st.plotly_chart(plot_normalized(df_kaya, variable), use_container_width=True)

            st.subheader("🧮 Kaya Identity Validation")
            fig = px.line(df_kaya, x="year", y=["CO2 (kt)", "CO2_reconstructed"], title="Actual vs Reconstructed CO₂")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Load data' to fetch selected indicators from the World Bank for the chosen country and year range.")

# End of file
