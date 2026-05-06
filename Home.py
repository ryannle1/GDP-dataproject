"""
Name:        Ryan Le, Mitchel Igolimah, Ashley Shambare
Data:        Global Prosperity Index (Legatum) + World Bank GDP (1960–2023)
URL:         https://gdp-dataproject-jmyuywwxxd6qmb3ujgktby.streamlit.app

Description:
    This program is an interactive Streamlit application that explores the
    relationship between national prosperity dimensions and economic output
    across ~150 countries.

    Page 1 — Prosperity ↔ GDP — fits an OLS regression of 2023 GDP on
    prosperity scores with an infrastructure × regime interaction term.
    Users can filter by region and political regime, choose any of the 12
    prosperity dimensions to plot against GDP, and toggle removal of two
    high-influence Cook's-distance outliers.

    Page 2 — Country Explorer — lets users zoom in on individual countries
    via three queries: top-N rankings by any metric, GDP trajectories from
    1960 to 2023, and a world map of country centroids weighted by any
    chosen metric.

    Charts include scatter plots with regression lines, a region × regime
    pivot table, a Plotly choropleth, a horizontal bar chart, a line chart,
    and a latitude/longitude bubble map.

References / tools used:
    - Anthropic Claude Code (CLI assistant) — used to scaffold the multi-page
      Streamlit layout and the lib/ data + model helpers. All analytical
      decisions (model formula, outlier choices, country-name corrections,
      chart selection) are the group's own.
    - Streamlit docs — multi-page apps, caching, AppTest.
    - statsmodels formula API — OLS with interaction terms.
    - Google Public Data Explorer — country centroid CSV (public domain).

Run locally:  streamlit run Home.py (after installing requirements.txt)
"""

import streamlit as st

from lib.data import build_merged, load_gdp_long

# ---------------------------------------------------------------------------
# Page setup — landing page Streamlit serves at "/". The two analysis pages
# live under pages/ and Streamlit auto-discovers them into the sidebar nav.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CS 602 Project", layout="wide")

st.title("Global Prosperity & Economic Outcomes")
st.caption("CS 602 — Project · Ryan Le, Mitchel Igolimah, Ashley Shambare")

# ---------------------------------------------------------------------------
# Project overview — one-paragraph intro framing the single coherent story.
# ---------------------------------------------------------------------------
st.markdown(
    """
This project explores the relationship between **national prosperity** and
**economic output** using two real-world datasets — the Legatum Global
Prosperity Index (12 dimensions per country) and the World Bank's GDP series
(1960–2023). We ask: *do better-scoring countries on prosperity dimensions
like infrastructure, education, and health actually produce higher GDP?
Does political regime change that relationship?*

Use the sidebar to navigate:

1. **Prosperity ↔ GDP** — the headline regression, with filters, a world
   choropleth, and Cook's-distance diagnostics.
2. **Country Explorer** — top-N rankings, GDP trajectories from 1960 to
   2023, and a world map of country centroids by any chosen metric.
"""
)

# ---------------------------------------------------------------------------
# Dataset summary metrics — touching the loaders here primes Streamlit's
# cache so the analysis pages render instantly when the user clicks through.
# ---------------------------------------------------------------------------
merged = build_merged()
gdp_long = load_gdp_long()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries (merged)", len(merged))
col2.metric("Regions", merged["region"].nunique())
col3.metric("Year span (GDP)", f"{gdp_long['year'].min()}–{gdp_long['year'].max()}")
col4.metric("Country-year rows", f"{len(gdp_long):,}")

st.info("Open **Prosperity_GDP** or **Country_Explorer** in the sidebar.")
