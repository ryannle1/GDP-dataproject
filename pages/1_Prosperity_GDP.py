"""Workstream A: Prosperity scores vs. GDP (interactive port of Notebook 1)."""

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

from lib.data import PROSPERITY_SCORE_COLS, build_merged
from lib.models import fit_prosperity_ols

st.set_page_config(page_title="Prosperity ↔ GDP", layout="wide")
st.title("Prosperity scores vs. 2023 GDP")
st.caption("OLS with an infrastructure × regime interaction term.")

merged = build_merged()

with st.sidebar:
    st.header("Filters")
    regions = sorted(merged["region"].dropna().unique())
    selected_regions = st.multiselect("Region", regions, default=regions)

    regime_groups = sorted(merged["Regime_Group"].dropna().unique())
    selected_regimes = st.multiselect("Regime group", regime_groups, default=regime_groups)

    drop_outliers = st.checkbox("Drop high-influence outliers (idx 151, 25)", value=True)

    st.header("Plot")
    primary_predictor = st.selectbox(
        "X-axis predictor",
        list(PROSPERITY_SCORE_COLS),
        index=list(PROSPERITY_SCORE_COLS).index("infrastructure_and_market_access"),
    )

filtered = merged[
    merged["region"].isin(selected_regions)
    & merged["Regime_Group"].isin(selected_regimes)
]

if filtered.empty:
    st.warning("No rows match the current filters. Loosen them in the sidebar.")
    st.stop()

st.subheader(f"Merged dataset — {len(filtered)} countries")
st.dataframe(
    filtered[
        [
            "Country Name", "region", "Regime_Group", "prosperity_score",
            "gdp_2023", *PROSPERITY_SCORE_COLS,
        ]
    ],
    use_container_width=True, hide_index=True,
)

st.subheader(f"{primary_predictor} vs. GDP, by regime")
g = sns.lmplot(
    data=filtered,
    x=primary_predictor,
    y="gdp_2023",
    hue="Regime_Group",
    markers=["o", "x"],
    palette="Set2",
    height=6, aspect=1.6,
)
g.set_axis_labels(primary_predictor, "GDP 2023 (USD)")
st.pyplot(g.figure)
plt.close(g.figure)

if drop_outliers:
    fit_input = filtered.drop(index=[i for i in (151, 25) if i in filtered.index])
else:
    fit_input = filtered

if fit_input.empty:
    st.error("No rows left to fit the model after applying outlier drop. Adjust filters.")
    st.stop()

model = fit_prosperity_ols(fit_input)

with st.expander("OLS regression summary", expanded=True):
    st.text(model.summary().as_text())

with st.expander("Cook's-distance influence plot"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    ax.set_title("Influence Plot (Cook's Distance)")
    st.pyplot(fig)
    plt.close(fig)
