"""Data loaders and the prosperity/GDP merge pipeline.

Both Streamlit pages (and the landing app) import everything they need
from this module. Centralizing the loaders here means the data prep
happens in exactly one place and is cached across page navigations,
so switching pages does not re-read the CSVs.

All loaders use relative paths so the same code runs locally and on
Deepnote. Missing files raise FileNotFoundError and broken invariants
raise AssertionError — failures are loud rather than papered over.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Resolve the project root from this file's location so callers can import
# us regardless of where Streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Metric names where a HIGHER value means a BETTER outcome. Only GDP works
# that way — every prosperity score is a *ranking* (1 = best, 167 = worst),
# so for prosperity metrics a LOWER value is better. Both pages use this set
# to decide whether "Top N" means nlargest (GDP) or nsmallest (rankings),
# and to invert color/size scales on maps so bright/big always means "better".
HIGHER_IS_BETTER = {"gdp_2023"}

# The 12 prosperity dimensions used as predictors in the OLS model.
# Stored as a tuple so it can be safely shared across pages without mutation.
PROSPERITY_SCORE_COLS = (
    "safety_and_security",
    "personal_freedom",
    "governance",
    "social_capital",
    "investment_environment",
    "enterprise_conditions",
    "infrastructure_and_market_access",
    "economic_quality",
    "living_conditions",
    "health",
    "education",
    "natural_environment",
)

# Maps prosperity-dataset country names to the World Bank GDP dataset's
# spelling. Without these corrections the inner-merge silently drops ~14
# countries whose names disagree across the two sources.
MANUAL_CORRECTIONS: dict[str, str] = {
    "Hong Kong": "Hong Kong SAR, China",
    "South Korea": "Korea, Rep.",
    "Slovakia": "Slovak Republic",
    "Vietnam": "Viet Nam",
    "Russia": "Russian Federation",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Turkey": "Turkiye",
    "Laos": "Lao PDR",
    "Egypt": "Egypt, Arab Rep.",
    "Iran": "Iran, Islamic Rep.",
    "Venezuela": "Venezuela, RB",
    "Congo": "Congo, Rep.",
    "Syria": "Syrian Arab Republic",
    "Yemen": "Yemen, Rep.",
}


@st.cache_data
def load_global_prosperity() -> pd.DataFrame:
    """Load the prosperity dataset and coerce score columns to int64.

    Latin-1 encoding is required because the source file uses non-UTF-8
    characters in some country names (e.g. Côte d'Ivoire).
    """
    df = pd.read_csv(PROJECT_ROOT / "global_prosperity.csv", encoding="ISO-8859-1")
    # [DA1] clean/manipulate data — coerce score columns from float to int64
    # since they're rankings (whole numbers), not continuous measurements.
    for col in PROSPERITY_SCORE_COLS:
        # [DA9] add column / column calculation — overwrite with cast values.
        df[col] = df[col].astype(np.int64)
    return df


@st.cache_data
def load_gdp() -> pd.DataFrame:
    """Load the World Bank GDP table and keep only the 2023 column."""
    df = pd.read_csv(PROJECT_ROOT / "gdp_data.csv")
    # [DA7] select columns — keep only what we need for the merge + 2023 GDP.
    return df[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "2023"]]


@st.cache_data
def load_gdp_long() -> pd.DataFrame:
    """Load the GDP table reshaped to long format (one row per country-year).

    Used by the Country Explorer page's "GDP over time" line chart. The raw
    CSV is wide (years 1960..2023 are columns); melting makes the time series
    plottable directly.
    """
    df = pd.read_csv(PROJECT_ROOT / "gdp_data.csv")
    year_cols = [str(y) for y in range(1960, 2024)]
    # [DA1] clean/manipulate data — wide-to-long reshape so each (country, year)
    # is a single row with one numeric value.
    long = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp",
    )
    # [DA9] column calculation — convert year strings to integers for plotting.
    long["year"] = long["year"].astype(int)
    long = long.dropna(subset=["gdp"])
    return long


@st.cache_data
def load_country_centroids() -> pd.DataFrame:
    """Load ISO-3 → (latitude, longitude) centroids used by the lat/long map.

    Source: Google Public Data Explorer "countries.csv" (public domain).
    File lives at data/country_centroids.csv with columns:
        Country Code, latitude, longitude, name
    """
    df = pd.read_csv(PROJECT_ROOT / "data" / "country_centroids.csv")
    assert {"Country Code", "latitude", "longitude"}.issubset(df.columns), (
        "country_centroids.csv missing required columns"
    )
    return df


@st.cache_data
def build_merged() -> pd.DataFrame:
    """Build the country-level analytical table by joining prosperity and GDP.

    This is the single source of truth that page 1 (Prosperity_GDP)
    consumes. The pipeline: apply the country name corrections,
    inner-merge on the corrected names, rename columns to friendlier
    identifiers, drop countries missing 2023 GDP, and add a binary
    regime grouping (Democracy vs. Autocracy).
    """
    global_df = load_global_prosperity().copy()
    gdp_df = load_gdp()

    # [PY3] try/except — surface a clear error if global_prosperity.csv is
    # missing the 'country' column instead of letting the lambda raise a
    # cryptic KeyError deep inside .apply(). The except re-raises with
    # context rather than swallowing the failure.
    try:
        # [PY5] dict access — MANUAL_CORRECTIONS.get(x, x) looks up the
        # corrected spelling, falling back to the original name when no
        # correction is needed.
        global_df["Countries"] = global_df["country"].apply(
            lambda x: MANUAL_CORRECTIONS.get(x, x)
        )
    except KeyError as e:
        raise ValueError(
            f"global_prosperity.csv is missing required column {e}. "
            "Check that the CSV header row is intact."
        ) from e

    # Step 2: inner-merge on the harmonized name so only countries present
    # in both datasets survive — anything that can't be matched is dropped.
    merged = gdp_df.merge(
        global_df, how="inner", left_on="Country Name", right_on="Countries"
    )

    # Step 3: rename to readable identifiers (the raw names are awkward).
    merged = merged.rename(columns={"average_score": "prosperity_score", "2023": "gdp_2023"})

    # [DA4] filter by one condition — drop countries with missing 2023 GDP
    # since they can't participate in the regression.
    merged = merged.dropna(subset=["gdp_2023"])

    # [DA7] add column / new grouping — collapse the four-level regime
    # variable into a binary category so the OLS interaction is interpretable.
    merged["Regime_Group"] = merged["political_regime"].map(
        {
            "Electoral democracy": "Democracy",
            "Liberal democracy": "Democracy",
            "Electoral autocracy": "Autocracy",
            "Closed autocracy": "Autocracy",
        }
    )

    # Loud invariants — if either fails the page will halt with a clear
    # error pointing at the problem (per the no-fallback rule).
    assert len(merged) > 0, "merged df is empty — check MANUAL_CORRECTIONS or input CSVs"
    assert "gdp_2023" in merged.columns, "gdp_2023 column missing after rename"
    # Do NOT reset_index — page 1's outlier-removal targets rows 151 and 25
    # by their post-merge index, and we want to preserve that mapping.
    return merged
