"""OLS regression model for the prosperity-vs-GDP analysis.

Role in the app:
    The Prosperity_GDP page calls into this module to fit the regression.
    Keeping the model code here means the page stays focused on UI/widgets,
    and the modeling choices (formula, assertions) live in one place that's
    easy to audit.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

# Workstream regression formula. The `*` between infrastructure and
# Regime_Group expands into both main effects PLUS the interaction term,
# letting the slope of GDP-on-infrastructure differ between democracies
# and autocracies — that interaction is the central question of the analysis.
OLS_FORMULA = (
    "gdp_2023 ~ infrastructure_and_market_access * Regime_Group "
    "+ education + health + economic_quality + living_conditions + social_capital"
)


def fit_prosperity_ols(df: pd.DataFrame):
    """Fit the prosperity-vs-GDP OLS regression.

    Role: workstream A's only modeling step. The page passes in a dataframe
    that has already been filtered (region, regime) and possibly had outliers
    dropped, so this function makes no assumptions about which rows are in.
    Returns a statsmodels RegressionResultsWrapper that the page renders
    via .summary() and feeds into the influence plot.
    """
    assert len(df) > 0, "fit_prosperity_ols received an empty dataframe"
    model = smf.ols(OLS_FORMULA, data=df).fit()
    assert model.nobs > 0, "OLS fit produced zero observations"
    return model
