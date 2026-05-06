"""Country Explorer — interactive views of the merged GDP/prosperity dataset.

Role in the app:
    Companion page to Prosperity_GDP. Where page 1 fits a regression across
    every country, this page lets the user zoom in on individual countries
    and rankings: top-N by any metric, GDP trajectories from 1960 to 2023,
    and a world map of countries weighted by any prosperity dimension.

Tabs on the page (left to right):
    Top Rankings    — snapshot metrics + Top-N countries by selected metric
    GDP Over Time   — line chart of selected countries' GDP trajectories
    World Map       — lat/long bubble map sized by selected metric
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from lib.data import (
    HIGHER_IS_BETTER,
    PROSPERITY_SCORE_COLS,
    build_merged,
    load_country_centroids,
    load_gdp_long,
)

# ---------------------------------------------------------------------------
# Page setup — [ST4] customized page design (wide layout, custom page title).
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Country Explorer", layout="wide")
st.title("Country Explorer")
st.caption("Top-N rankings, GDP trajectories, and a world map of any metric.")

merged = build_merged()
centroids = load_country_centroids()
gdp_long = load_gdp_long()

# Metric options shared by several tabs — GDP plus the 12 prosperity dims.
METRIC_OPTIONS = ["gdp_2023", "prosperity_score", *PROSPERITY_SCORE_COLS]


# ---------------------------------------------------------------------------
# Helper functions used by the tabs below.
# ---------------------------------------------------------------------------

def top_n_countries(df: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    """[PY1] Return the top-N rows of `df` ordered by `metric` descending.

    The default `n=10` is used by the headline snapshot at the top of
    the Top Rankings tab; the bar-chart section below calls the function
    again with the slider value to override the default.
    """
    # [DA3] find top largest values in a column.
    return df.nlargest(n, metric)


def gdp_summary(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """[PY2] Return (min, max, mean, median) of gdp_2023 — multi-value return.

    Drives the four metric cards in the snapshot row so the headline
    statistics are computed in one place.
    """
    s = df["gdp_2023"]
    return s.min(), s.max(), s.mean(), s.median()


# ---------------------------------------------------------------------------
# Sidebar — [ST1] slider, [ST2] selectbox, [ST3] multiselect. Sidebar widgets
# are page-level, so the same selections flow into whatever tab is active.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Top-N controls")
    # [ST1] slider widget.
    top_n = st.slider("Top N countries", min_value=5, max_value=30, value=10, step=1)
    # [ST2] selectbox widget — drives the bar chart and the world map.
    metric = st.selectbox("Metric", METRIC_OPTIONS, index=0)

    st.header("Time series")
    default_countries = [c for c in ("United States", "China", "India") if c in merged["Country Name"].values]
    # [ST3] multiselect widget.
    chosen_countries = st.multiselect(
        "Countries to plot over time",
        sorted(merged["Country Name"].unique()),
        default=default_countries,
    )
    start_year = st.slider("Start year", 1960, 2023, 1990, 1)
    log_scale = st.checkbox("Log-scale GDP axis", value=True)

# ---------------------------------------------------------------------------
# Tabs — purely a layout primitive. All three share the same sidebar state.
# ---------------------------------------------------------------------------
tab_topn, tab_time, tab_map = st.tabs(
    ["Top Rankings", "GDP Over Time", "World Map"]
)

# ===========================================================================
# Tab: Top Rankings — snapshot metrics + bar chart
# ===========================================================================
with tab_topn:
    # Headline snapshot — min/max/mean/median GDP across the merged set.
    gdp_min, gdp_max, gdp_mean, gdp_median = gdp_summary(merged)
    top_default = top_n_countries(merged, "gdp_2023")  # uses default n=10

    st.subheader("Snapshot — 2023 GDP across all countries")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min GDP", f"${gdp_min/1e9:,.1f}B")
    c2.metric("Max GDP", f"${gdp_max/1e12:,.2f}T")
    c3.metric("Mean GDP", f"${gdp_mean/1e9:,.1f}B")
    c4.metric("Median GDP", f"${gdp_median/1e9:,.1f}B")
    st.caption(
        f"Highest 2023 GDP: **{top_default.iloc[0]['Country Name']}** at "
        f"${top_default.iloc[0]['gdp_2023']/1e12:,.2f}T."
    )

    st.divider()

    # Top-N countries by selected metric.
    # Query: "What are the top N countries by [metric]?"
    # [VIZ1] horizontal bar chart with title, custom colors, axis labels.
    #
    # Direction-aware: for GDP we want the LARGEST values; for prosperity
    # rankings (where 1 = best, 167 = worst) we want the SMALLEST values.
    # Using the wrong direction would silently surface the worst-performing
    # countries when the user asked for the best.
    is_higher_better = metric in HIGHER_IS_BETTER
    direction_label = "highest" if is_higher_better else "best-ranked (lowest rank number)"

    st.subheader(f"Top {top_n} countries by {metric.replace('_', ' ')} — {direction_label}")

    # [DA4] filter by one condition — drop rows where the chosen metric is null.
    metric_df = merged.dropna(subset=[metric])

    if is_higher_better:
        # GDP path: helper uses nlargest, so this satisfies the [PY1] "called
        # with explicit n (no default)" pattern.
        top_view = top_n_countries(metric_df, metric, n=top_n)
    else:
        # Ranking path: nsmallest gives us the BEST-ranked countries
        # (rank 1 is the best country in the dataset).
        # [DA3] find smallest values of a column.
        top_view = metric_df.nsmallest(top_n, metric)

    # [DA2] sort so the BEST country sits at the top of the bar chart in
    # both directions: ascending=True for GDP (largest bar on top), and
    # ascending=False for rankings (lowest rank number = best, on top).
    top_view = top_view.sort_values(metric, ascending=is_higher_better)

    fig_bar, ax_bar = plt.subplots(figsize=(8, max(4, 0.3 * top_n)))
    ax_bar.barh(top_view["Country Name"], top_view[metric], color="#2E86AB")
    ax_bar.set_xlabel(metric.replace("_", " ").title())
    ax_bar.set_ylabel("Country")
    ax_bar.set_title(
        f"Top {top_n} by {metric.replace('_', ' ')} — {direction_label}"
    )
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3)
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)
    if not is_higher_better:
        st.caption(
            "Prosperity scores are *rankings* (1 = best). The chart shows the "
            "countries with the **lowest rank numbers** for this dimension — "
            "i.e. the best-performing countries."
        )

# ===========================================================================
# Tab: GDP Over Time — line chart of selected countries
# ===========================================================================
with tab_time:
    # Query: "How has GDP for [country list] evolved from [start year]?"
    # [VIZ2] line chart with custom legend, axis labels, optional log scale.
    st.subheader("GDP over time")
    if not chosen_countries:
        st.info("Pick at least one country in the sidebar to see the time series.")
    else:
        # [DA5] filter by 2+ conditions with AND — country in list AND year >= start.
        ts = gdp_long[
            gdp_long["Country Name"].isin(chosen_countries)
            & (gdp_long["year"] >= start_year)
        ]

        fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
        for country in chosen_countries:
            country_ts = ts[ts["Country Name"] == country].sort_values("year")
            ax_ts.plot(country_ts["year"], country_ts["gdp"], marker="o", markersize=3, label=country)
        if log_scale:
            ax_ts.set_yscale("log")
        ax_ts.set_xlabel("Year")
        ax_ts.set_ylabel("GDP (USD)" + (" — log scale" if log_scale else ""))
        ax_ts.set_title(f"GDP from {start_year} to 2023")
        ax_ts.legend(loc="upper left")
        ax_ts.grid(linestyle="--", alpha=0.3)
        fig_ts.tight_layout()
        st.pyplot(fig_ts)
        plt.close(fig_ts)

        # [DA8] iterate rows of a DataFrame with iterrows() — render a small
        # "Latest GDP" caption underneath the chart.
        latest = (
            ts.sort_values("year")
            .groupby("Country Name", as_index=False)
            .last()
        )
        parts = []
        for _, row in latest.iterrows():
            parts.append(f"{row['Country Name']} ({int(row['year'])}): ${row['gdp']/1e9:,.1f}B")
        st.caption("Latest GDP in series — " + " · ".join(parts))

# ===========================================================================
# Tab: World Map — lat/long bubble map
# ===========================================================================
with tab_map:
    # Query: "Where on the map are countries with high [metric]?"
    # [VIZ4 MAP] interactive map plotted from country centroid latitude and
    # longitude — bubble area scales with the user-chosen metric.
    st.subheader(f"World map — bubble size = {metric.replace('_', ' ')}")

    # Join the merged dataset to the centroid lookup so st.map gets lat/long.
    # [DA1] clean/manipulate — left-join to attach geographic coordinates.
    map_df = merged.merge(centroids, on="Country Code", how="left").dropna(
        subset=["latitude", "longitude", metric]
    )

    # st.map expects lowercase 'lat' and 'lon' columns.
    # [DA9] add columns / column calculation — rename + compute a normalized
    # bubble size so very large values don't drown out smaller ones.
    map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})
    metric_min = map_df[metric].min()
    metric_max = map_df[metric].max()

    # Direction-aware bubble sizing: bigger circle should always mean
    # "better-performing country", regardless of whether the metric is
    # higher-is-better (GDP) or lower-is-better (prosperity rankings).
    if metric in HIGHER_IS_BETTER:
        bubble_basis = map_df[metric]                    # bigger value → bigger bubble
    else:
        bubble_basis = -map_df[metric]                   # invert so rank 1 → bigger bubble

    basis_min = bubble_basis.min()
    basis_max = bubble_basis.max()
    if basis_max > basis_min:
        map_df["size"] = (
            20_000 + 480_000 * (bubble_basis - basis_min) / (basis_max - basis_min)
        )
    else:
        map_df["size"] = 100_000

    st.map(map_df, latitude="lat", longitude="lon", size="size", color="#2E86AB")
    direction_text = (
        "higher value" if metric in HIGHER_IS_BETTER
        else "better rank (lower number)"
    )
    st.caption(
        f"Bubble area scales with **{metric.replace('_', ' ')}** — bigger circle "
        f"= **{direction_text}**. Value range across {len(map_df)} countries: "
        f"{metric_min:,.0f} – {metric_max:,.0f}."
    )
