"""Data loaders and the prosperity/GDP merge pipeline.

All loaders use relative paths so the same code runs locally and on Deepnote.
Per CLAUDE.md, no fallbacks: missing files raise FileNotFoundError, broken
invariants raise AssertionError. Both make the failure obvious.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

# Lifted verbatim from Notebook 1, cell 18. Maps prosperity-dataset country
# names to the World Bank GDP dataset's spelling.
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
    df = pd.read_csv(PROJECT_ROOT / "global_prosperity.csv", encoding="ISO-8859-1")
    for col in PROSPERITY_SCORE_COLS:
        df[col] = df[col].astype(np.int64)
    return df


@st.cache_data
def load_gdp() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "gdp_data.csv")
    return df[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "2023"]]


@st.cache_data
def build_merged() -> pd.DataFrame:
    """Reproduces the merge from Notebook 1 cells 18–37."""
    global_df = load_global_prosperity().copy()
    gdp_df = load_gdp()

    global_df["Countries"] = global_df["country"].apply(
        lambda x: MANUAL_CORRECTIONS.get(x, x)
    )

    merged = gdp_df.merge(
        global_df, how="inner", left_on="Country Name", right_on="Countries"
    )
    merged = merged.rename(columns={"average_score": "prosperity_score", "2023": "gdp_2023"})
    merged = merged.dropna(subset=["gdp_2023"])

    merged["Regime_Group"] = merged["political_regime"].map(
        {
            "Electoral democracy": "Democracy",
            "Liberal democracy": "Democracy",
            "Electoral autocracy": "Autocracy",
            "Closed autocracy": "Autocracy",
        }
    )

    assert len(merged) > 0, "merged df is empty — check MANUAL_CORRECTIONS or input CSVs"
    assert "gdp_2023" in merged.columns, "gdp_2023 column missing after rename"
    # Do NOT reset_index — Notebook 1 references rows 151 and 25 as outliers
    # by their post-merge index, and we want to preserve that mapping.
    return merged


@st.cache_data
def load_bank_train() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "bank_train.csv")


@st.cache_data
def load_bank_new() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "bank_new.csv")
