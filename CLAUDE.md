# CS 602 — Project 3

Graduate course final project (Spring 2026). The deliverable is an interactive Streamlit app that explores the relationship between national prosperity dimensions and economic output.

## Workstream — Global Prosperity & GDP

Single-story analysis using two datasets:

- `global_prosperity.csv` — Legatum Global Prosperity Index, 12 dimensions per country plus political regime.
- `gdp_data.csv` — World Bank GDP series, 1960–2023.

OLS in `statsmodels` modeling `gdp_2023 ~ infrastructure_and_market_access * Regime_Group + education + health + economic_quality + living_conditions + social_capital`. Country names are reconciled via the hand-curated `MANUAL_CORRECTIONS` dict in `lib/data.py` before an inner merge. Two high-influence rows (idx 151, 25) are dropped after a Cook's-Distance check to produce the cleaner fit (R² ≈ 0.23 vs. 0.10 with outliers in).

## Environment

Runs both locally and in Deepnote. The app uses **relative paths only** — `pd.read_csv("gdp_data.csv")`, not `/work/gdp_data.csv`. Don't reintroduce Deepnote-absolute paths.

`Init.ipynb` is the Deepnote bootstrap (installs `requirements.txt`); leave it alone unless adding pinned deps.

## Streamlit presentation app

Run locally with `streamlit run app.py` (opens at `http://localhost:8501`). Layout:

- `app.py` — landing page with the **required documentation header** (Name/Data/URL/Description), project overview, and dataset summary metrics.
- `pages/1_Prosperity_GDP.py` — region/regime filters, predictor selector, outlier toggle, regression scatter, OLS summary, region × regime pivot table, Plotly choropleth, Cook's-distance plot.
- `pages/2_Country_Explorer.py` — three queries: top-N countries by metric (bar chart), GDP-over-time line chart, lat/long world bubble map. Headline pandas summary (min/max/mean/median GDP) at the top.
- `lib/data.py` — cached CSV loaders + the `MANUAL_CORRECTIONS` dict + `build_merged()` + `load_gdp_long()` + `load_country_centroids()`. Data invariants enforced via `assert`s. **Do not call `reset_index` in `build_merged`** — page 1's outlier removal targets rows 151 and 25 by their post-merge index, and the page relies on that mapping.
- `lib/models.py` — `fit_prosperity_ols` (statsmodels). The bank classifier was removed — see "History" below.
- `data/country_centroids.csv` — 241 rows, ISO-3 → (latitude, longitude). Sourced from Google Public Data Explorer (public domain) and pre-converted to ISO-3 via pycountry; pycountry is NOT a runtime dependency.

Cache choices: `@st.cache_data` on every loader and on `build_merged`. The OLS fit is intentionally not cached — it's sub-second on ~150 rows and sidebar filters change the input rows on every interaction.

## Rubric tagging

The grader keyword-searches the code for `[PY1]`–`[PY5]`, `[ST1]`–`[ST4]`, `[VIZ1]`–`[VIZ4]`, `[DA1]`–`[DA9]`. Every tag is present at least once across `app.py`, `lib/data.py`, `pages/1_Prosperity_GDP.py`, and `pages/2_Country_Explorer.py`. When adding new code, **leave existing tag comments in place** — removing one breaks the audit.

## History

The original build had a second workstream (UK bank term-deposit classification) on its own page. It was removed before submission because it shared no narrative thread with the prosperity/GDP analysis and hurt the rubric's "tells a story" score. If you need to restore it, see the `pages/2_Bank_Deposit_Classifier.py` deletion in git history.

## Working agreements

### Update this file after every big change
After any non-trivial change — new analysis page, new model, new dataset, schema rename, restructured workflow, new dependency — update the relevant section above. The file is the project's source of truth; if it drifts, future sessions will be wrong.

What counts as "big": anything that changes inputs/outputs, columns, model formulas, file layout, or how the project is run. Cosmetic edits don't count.

### Error handling: be loud, but be friendly at the boundary.
Functionality must work, or it must surface a clear, actionable error. `try/except` is welcome and shows up in several places — it's part of demonstrating real error-handling craft. The rule is *what you do inside the `except` block*: never swallow a failure and pretend everything's fine.

**Good `try/except` patterns** (use these freely):

- **Add context and re-raise** — e.g. `lib/data.py`'s `build_merged` catches a `KeyError` from a malformed prosperity CSV and re-raises a `ValueError` that names the missing column.
- **Convert exception to user-friendly Streamlit message** — at the top of a page, wrap a fragile operation, then call `st.error("Couldn't compute X because Y. Try Z.")` followed by `st.stop()`. The user sees a clean message instead of a red traceback box.
- **Catch a *specific* exception class**, never bare `except:` or bare `except Exception:` (those hide bugs you'd want to see).
- **Log/show the original error** — when surfacing to the user, include `str(exc)` or the exception type so the underlying cause is still visible for debugging.

**Bad patterns** (still forbidden):

- `try: ... except: pass` — silently eats the error.
- `try: load_real_data() except: load_dummy_data()` — falls back to wrong data without telling anyone.
- Silent `.fillna()` / `.dropna()` without first confirming the missingness is expected.
- Defaults that mask a real failure: `df.get(col, 0)`, `pd.read_csv(..., on_bad_lines='skip')`.
- `if file_exists: real else: dummy` path-juggling.

**Other allowed and encouraged**:

- Explicit `raise ValueError("merged df empty — check country-name reconciliation")` when an invariant breaks.
- Assertions on shape/columns after merges (`assert len(df_merged) > 0`, `assert 'gdp_2023' in df_merged.columns`).
- Letting `pd.read_csv` raise `FileNotFoundError` naturally if the user hasn't wrapped it — that IS a graceful error.
- Boundary-level `st.warning(...) + st.stop()` when filters produce zero rows (already done on both pages).

The point: when something breaks, the user (and the developer reading the traceback) should know **what** broke and **why**. Never paper over a failure to keep the app limping along on bad data.
