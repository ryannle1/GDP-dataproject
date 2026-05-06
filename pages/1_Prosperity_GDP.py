"""Workstream A: Prosperity scores vs. GDP.

Role in the app:
    Interactive presentation of the OLS regression linking 12 prosperity
    dimensions and political regime type to a country's 2023 GDP. The user
    can filter the dataset, choose which prosperity dimension to plot
    against GDP, and toggle outlier removal — every change re-renders the
    dataframe, scatterplot, model summary, and influence plot in place.

Tabs on the page (left to right):
    Data        — filtered country table + region × regime mean-GDP pivot
    World Map   — interactive choropleth, color metric chosen inline
    Regression  — scatter plot of selected predictor + OLS summary
    Insights    — correlation strength chart (Q1), over/under-performers
                  (Q2), Cook's-distance influence plot
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

from lib.data import PROSPERITY_SCORE_COLS, build_merged
from lib.models import fit_prosperity_ols

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Prosperity ↔ GDP", layout="wide")
st.title("Prosperity scores vs. 2023 GDP")
st.caption("OLS with an infrastructure × regime interaction term.")

# Build the merged country-level dataframe once. build_merged() is cached, so
# this is essentially free on every rerun once the first run has populated
# the cache.
merged = build_merged()

# ---------------------------------------------------------------------------
# Sidebar — filters and plot controls
# ---------------------------------------------------------------------------
# Each widget's return value drives a downstream filter or rendering choice.
# Streamlit reruns the script on every widget change, so the rest of the page
# always reflects the current sidebar state. Sidebar widgets are page-level,
# so the same selection flows into every tab below.
with st.sidebar:
    st.header("Filters")
    regions = sorted(merged["region"].dropna().unique())
    selected_regions = st.multiselect("Region", regions, default=regions)

    regime_groups = sorted(merged["Regime_Group"].dropna().unique())
    selected_regimes = st.multiselect("Regime group", regime_groups, default=regime_groups)

    # Rows 151 and 25 were flagged as high-influence via Cook's distance
    # in earlier exploratory analysis; dropping them more than doubles the
    # model's R². Default is on so the headline summary shows the cleaner fit.
    drop_outliers = st.checkbox("Drop high-influence outliers (idx 151, 25)", value=True)

    st.header("Plot")
    # Only the scatterplot's X-axis changes when this is updated; the OLS
    # formula stays fixed so the model summary remains comparable across
    # predictor choices.
    primary_predictor = st.selectbox(
        "X-axis predictor",
        list(PROSPERITY_SCORE_COLS),
        index=list(PROSPERITY_SCORE_COLS).index("infrastructure_and_market_access"),
    )

# ---------------------------------------------------------------------------
# Apply filters (runs before the tabs so empty-filter halts cleanly).
# ---------------------------------------------------------------------------
# [DA5] filter by 2+ conditions with AND — region in user list AND
# Regime_Group in user list. Boolean-mask the merged df by both at once.
filtered = merged[
    merged["region"].isin(selected_regions)
    & merged["Regime_Group"].isin(selected_regimes)
]

if filtered.empty:
    st.warning("No rows match the current filters. Loosen them in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Fit the OLS once before tabs — both the Regression and Insights tabs use
# `model` and `fit_input`, so computing them up front avoids duplicate work.
# ---------------------------------------------------------------------------
if drop_outliers:
    # [PY4] list comprehension — only includes outlier indices that actually
    # exist in the current filtered slice; if the user filtered them out via
    # region/regime the drop becomes a no-op rather than an error.
    outlier_idx = [i for i in (151, 25) if i in filtered.index]
    fit_input = filtered.drop(index=outlier_idx)

    # [DA8] iterate rows of a DataFrame with iterrows() — build an
    # "Excluded high-influence rows" caption listing exactly which
    # countries were dropped. Rendered in the Regression tab below.
    excluded = []
    for _, row in filtered.loc[outlier_idx].iterrows():
        excluded.append(f"{row['Country Name']} (GDP ${row['gdp_2023']:,.0f})")
    excluded_caption = (
        "Excluded high-influence rows: " + "; ".join(excluded) if excluded else ""
    )
else:
    fit_input = filtered
    excluded_caption = ""

if fit_input.empty:
    st.error("No rows left to fit the model after applying outlier drop. Adjust filters.")
    st.stop()

model = fit_prosperity_ols(fit_input)

# ---------------------------------------------------------------------------
# Tabs — purely a layout primitive. All four share the same sidebar state.
# ---------------------------------------------------------------------------
tab_data, tab_map, tab_regression, tab_insights = st.tabs(
    ["Data", "World Map", "Regression", "Insights"]
)

# ===========================================================================
# Tab: Data — filtered country table + mean-GDP pivot
# ===========================================================================
with tab_data:
    # Shows the rows the model is about to be fit on, with the score columns
    # spread out for spot-checking. hide_index=True keeps the merge-preserved
    # pandas index out of view.
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

    # [VIZ3] table visualization — a pivot table rendered as a styled dataframe.
    # [DA6] analyze with pivot tables — mean 2023 GDP by region × regime,
    # a quick "where is the money" view that complements the regression below.
    st.subheader("Mean 2023 GDP by region × regime")
    pivot = filtered.pivot_table(
        values="gdp_2023", index="region", columns="Regime_Group", aggfunc="mean"
    )
    st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)

# ===========================================================================
# Tab: World Map — interactive choropleth with inline metric selector
# ===========================================================================
with tab_map:
    # Interactive choropleth — hover any country to see its value, and the map
    # automatically reshapes when sidebar filters change. ISO-3 country codes
    # from the GDP dataset drive the country lookup.
    st.subheader("World map")
    map_metric = st.selectbox(
        "Color countries by",
        ["gdp_2023", "prosperity_score", *PROSPERITY_SCORE_COLS],
        index=0,
    )
    # Build a label map that strips underscores from column names so the hover
    # tooltip and color bar read naturally ("Regime Group" not "Regime_Group").
    map_labels = {
        col: col.replace("_", " ").title()
        for col in (*PROSPERITY_SCORE_COLS, "gdp_2023", "prosperity_score", "Regime_Group")
    }
    map_labels["gdp_2023"] = "GDP 2023 (USD)"

    fig_map = px.choropleth(
        filtered,
        locations="Country Code",
        color=map_metric,
        hover_name="Country Name",
        hover_data={"Country Code": False, "Regime_Group": True, map_metric: ":,.2f"},
        labels=map_labels,
        color_continuous_scale="Viridis",
        projection="natural earth",
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
    st.plotly_chart(fig_map, use_container_width=True)

# ===========================================================================
# Tab: Regression — scatter plot + OLS summary
# ===========================================================================
with tab_regression:
    # Seaborn's lmplot draws a separate regression line per regime group.
    # Different markers + palette make the lines easy to tell apart at a glance.
    st.subheader(f"{primary_predictor} vs. GDP, by regime")
    # Build the marker list dynamically from the regimes actually present in
    # the filtered slice — seaborn requires exactly one marker per hue level,
    # so a fixed ["o", "x"] would crash when the user filters to a single regime.
    regime_marker_map = {"Democracy": "o", "Autocracy": "x"}
    present_regimes = sorted(filtered["Regime_Group"].dropna().unique())
    plot_markers = [regime_marker_map.get(r, "o") for r in present_regimes]
    g = sns.lmplot(
        data=filtered,
        x=primary_predictor,
        y="gdp_2023",
        hue="Regime_Group",
        hue_order=present_regimes,
        markers=plot_markers,
        palette="Set2",
        height=6, aspect=1.6,
    )
    g.set_axis_labels(primary_predictor, "GDP 2023 (USD)")
    st.pyplot(g.figure)
    # Explicitly close the figure so matplotlib doesn't accumulate open figures
    # across reruns and warn about it in the terminal.
    plt.close(g.figure)

    if excluded_caption:
        st.caption(excluded_caption)

    # st.text preserves statsmodels' fixed-width formatting so the coefficient
    # table stays readable. Wrapped in an expander to keep the page tidy on
    # first load (open by default since this is the headline result).
    with st.expander("OLS regression summary", expanded=True):
        st.text(model.summary().as_text())

# ===========================================================================
# Tab: Insights — Q1 correlation + Q2 over/under-performers + Cook's plot
# ===========================================================================
with tab_insights:
    # -----------------------------------------------------------------------
    # Q1: which prosperity dimension correlates most strongly with GDP?
    # -----------------------------------------------------------------------
    # Computes Pearson r between gdp_2023 and each of the 12 dimensions, then
    # ranks them by absolute correlation so the strongest relationships sit
    # at the top (regardless of sign).
    st.subheader("Q1 — Which prosperity dimension correlates most strongly with GDP?")
    # [DA9] column calculation — Pearson r between each dimension and GDP.
    corr_rows = [
        {"Dimension": dim.replace("_", " ").title(),
         "Pearson r with GDP": fit_input[dim].corr(fit_input["gdp_2023"])}
        for dim in PROSPERITY_SCORE_COLS
    ]
    corr_df = pd.DataFrame(corr_rows)
    # [DA2] sort — order by absolute correlation so the strongest relationships
    # end up at the top of the bar chart.
    corr_df = corr_df.reindex(corr_df["Pearson r with GDP"].abs().sort_values().index)

    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    bar_colors = ["#2E86AB" if v >= 0 else "#A4303F" for v in corr_df["Pearson r with GDP"]]
    ax_corr.barh(corr_df["Dimension"], corr_df["Pearson r with GDP"], color=bar_colors)
    ax_corr.axvline(0, color="black", linewidth=0.5)
    ax_corr.set_xlabel("Pearson correlation with 2023 GDP")
    ax_corr.set_title("Strength of relationship: prosperity dimensions vs. GDP")
    ax_corr.grid(axis="x", linestyle="--", alpha=0.3)
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)
    st.caption(
        "Note: prosperity scores are *rankings* (lower = better-ranked country), "
        "so a **negative** correlation indicates that better-ranked countries "
        "tend to have higher GDP — i.e. a strong positive real-world relationship."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Q2: which countries overperform or underperform their prosperity score?
    # -----------------------------------------------------------------------
    # Uses the fitted OLS to predict each country's GDP from its prosperity
    # profile, then ranks by residual (actual − predicted). Positive residual
    # = the country produces MORE GDP than the model expects given its
    # prosperity scores; negative = LESS than expected.
    st.subheader("Q2 — Which countries overperform or underperform their prosperity score?")
    n_extreme = st.slider("How many of each", 5, 20, 10, 1, key="resid_n")

    # [DA9] column calculation — predicted GDP and residual per country.
    resid_df = fit_input.copy()
    resid_df["predicted_gdp"] = model.predict(resid_df)
    resid_df["residual"] = resid_df["gdp_2023"] - resid_df["predicted_gdp"]

    top_over = resid_df.nlargest(n_extreme, "residual")
    top_under = resid_df.nsmallest(n_extreme, "residual")
    combined = pd.concat([top_under, top_over]).sort_values("residual")

    fig_res, ax_res = plt.subplots(figsize=(10, max(6, 0.35 * len(combined))))
    res_colors = ["#2E86AB" if v >= 0 else "#A4303F" for v in combined["residual"]]
    ax_res.barh(combined["Country Name"], combined["residual"] / 1e9, color=res_colors)
    ax_res.axvline(0, color="black", linewidth=0.5)
    ax_res.set_xlabel("Residual (Billion USD): actual GDP − model-predicted GDP")
    ax_res.set_title(f"Top {n_extreme} over- (blue) and under-performers (red)")
    ax_res.grid(axis="x", linestyle="--", alpha=0.3)
    fig_res.tight_layout()
    st.pyplot(fig_res)
    plt.close(fig_res)
    st.caption(
        "Blue bars on the right = countries producing **more** GDP than their "
        "prosperity profile predicts. Red bars on the left = countries producing "
        "**less**. Large oil exporters and major industrial economies often "
        "appear on either extreme."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Cook's-distance influence plot — diagnostic that motivates dropping
    # rows 151 and 25; collapsed by default since it's a deep-dive view.
    # -----------------------------------------------------------------------
    with st.expander("Cook's-distance influence plot"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
        ax.set_title("Influence Plot (Cook's Distance)")
        st.pyplot(fig)
        plt.close(fig)
