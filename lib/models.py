"""Model fitting for both workstreams.

Workstream A: OLS via statsmodels (mirrors Notebook 1).
Workstream B: scikit-learn classifier on the bank deposit task.

@st.cache_resource is used because fitted estimators are not what
@st.cache_data is designed for (it serializes return values).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OLS_FORMULA = (
    "gdp_2023 ~ infrastructure_and_market_access * Regime_Group "
    "+ education + health + economic_quality + living_conditions + social_capital"
)

BANK_CATEGORICAL_COLS = (
    "job", "marital", "education", "default", "home_loan", "personal_loan", "contact",
)
BANK_NUMERIC_COLS = ("age", "balance")


def fit_prosperity_ols(df: pd.DataFrame):
    """Fit the OLS from Notebook 1. Caller is responsible for any row filtering."""
    assert len(df) > 0, "fit_prosperity_ols received an empty dataframe"
    model = smf.ols(OLS_FORMULA, data=df).fit()
    assert model.nobs > 0, "OLS fit produced zero observations"
    return model


def _encode_bank(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals; keep numerics as-is."""
    return pd.get_dummies(
        df, columns=list(BANK_CATEGORICAL_COLS), drop_first=True, dtype=float
    )


@st.cache_resource
def fit_bank_classifier(
    _df_train: pd.DataFrame,
    model_kind: Literal["logreg", "tree"],
    test_size: float,
    random_state: int = 42,
):
    """Train a deposit classifier and return (model, X_test, y_test, feature_columns)."""
    assert "deposit" in _df_train.columns, "bank_train.csv missing 'deposit' column"

    raw = _df_train.drop(columns=["customer_id"])
    y = (raw["deposit"] == "yes").astype(int)
    X = _encode_bank(raw.drop(columns=["deposit"]))

    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    if model_kind == "logreg":
        # Scale features so logreg converges (age/balance dwarf the 0/1 dummies).
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])
    elif model_kind == "tree":
        model = DecisionTreeClassifier(max_depth=8, random_state=random_state)
    else:
        raise ValueError(f"unknown model_kind: {model_kind!r}")

    model.fit(X_train, y_train)
    return model, X_test, y_test, feature_columns


def predict_bank_new(
    model,
    df_new: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score bank_new.csv using the trained model.

    Reindexes encoded features to match training columns *exactly*. Missing
    columns are filled with 0 (one-hot levels not present in new data),
    but unexpected extra columns will silently be dropped — that's a real
    problem if it happens, so we assert there are none.
    """
    assert "customer_id" in df_new.columns, "bank_new.csv missing 'customer_id'"
    encoded = _encode_bank(df_new.drop(columns=["customer_id"]))

    extra = set(encoded.columns) - set(feature_columns)
    assert not extra, f"bank_new.csv produced unexpected encoded columns: {sorted(extra)}"

    X = encoded.reindex(columns=feature_columns, fill_value=0)
    proba = model.predict_proba(X)[:, 1]
    pred = np.where(proba >= threshold, "yes", "no")

    return pd.DataFrame({
        "customer_id": df_new["customer_id"].to_numpy(),
        "predicted_deposit": pred,
        "probability_of_deposit": proba,
    })
