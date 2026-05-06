# CS 602 Data Science — Project 3

Graduate course project (Spring 2026). Two unrelated workstreams share this directory.

## Workstreams

### A. Global Prosperity ↔ GDP regression (`Notebook 1.ipynb`)
OLS in `statsmodels` modeling `gdp_2023 ~ infrastructure_and_market_access * Regime_Group + education + health + economic_quality + living_conditions + social_capital`. Inputs: `global_prosperity.csv`, `gdp_data.csv`. Country names are reconciled via the hand-curated `manual_corrections` dict before an inner merge. Two high-influence rows (idx 151, 25) are dropped after a Cook's-Distance check; the refit model is `model_clean`.

### B. Bank term-deposit classification (`bank_*.csv`)
- `bank_train.csv` — labeled training data (target: `deposit` yes/no)
- `bank_new.csv` — unlabeled holdout
- `bank_new_predictions.csv` — predictions with class + probability
- `bank_submit.csv` — submission format (`customer_id, deposit`)

No notebook for this workstream yet.

## Environment

Runs both locally and in Deepnote. The notebook uses **relative paths only** — `pd.read_csv("gdp_data.csv")`, not `/work/gdp_data.csv`. Don't reintroduce Deepnote-absolute paths.

`Init.ipynb` is the Deepnote bootstrap (installs `requirements.txt`); leave it alone unless adding pinned deps.

## Streamlit presentation app

Run locally with `streamlit run app.py` (opens at `http://localhost:8501`). Layout:

- `app.py` — landing page; shows row counts and links to the two pages.
- `pages/1_Prosperity_GDP.py` — interactive port of Notebook 1: region/regime filters, predictor selector, outlier toggle, OLS summary, Cook's-distance plot.
- `pages/2_Bank_Deposit_Classifier.py` — workstream B: logistic-regression / decision-tree toggle, test-split + threshold sliders, confusion matrix, ROC, predictions table for `bank_new.csv` with a submission-format download button.
- `lib/data.py` — cached CSV loaders + the `MANUAL_CORRECTIONS` dict + `build_merged()`. Data invariants are enforced via `assert`s. **Do not call `reset_index` in `build_merged`** — Notebook 1 references rows 151 and 25 as outliers by their post-merge index, and the page relies on that mapping.
- `lib/models.py` — `fit_prosperity_ols` (statsmodels), `fit_bank_classifier` + `predict_bank_new` (sklearn). Bank features are one-hot encoded with `drop_first=True`; `predict_bank_new` asserts no surprise columns appear in `bank_new.csv` versus training.

Cache choices: `@st.cache_data` on every loader and on `build_merged`. `@st.cache_resource` only on `fit_bank_classifier` (the training set is fixed; cache key is model_kind + test_size). The OLS fit is intentionally not cached — it's sub-second on ~150 rows and sidebar filters change the input rows on every interaction.

## Working agreements

### Update this file after every big change
After any non-trivial change — new analysis cell, new model, new dataset, schema rename, restructured workflow, new notebook for workstream B — update the relevant section above. The file is the project's source of truth; if it drifts, future sessions will be wrong.

What counts as "big": anything that changes inputs/outputs, columns, model formulas, file layout, or how the project is run. Cosmetic edits don't count.

### No fallbacks. Fail loudly.
Functionality must work, or it must raise a clear error that points at the problem. Forbidden:

- `try/except` that swallows errors and continues with a default
- `if file_exists else use_other_file` path-juggling
- Silent `.fillna()` / `.dropna()` without first confirming the missingness is expected
- Default values that mask a real failure (e.g. `df.get(col, 0)`, `pd.read_csv(..., on_bad_lines='skip')`)
- Wrapping a model fit in `try/except` to "skip if it fails"

Allowed and encouraged:

- Explicit `raise ValueError("merged df empty — check country-name reconciliation")` when an invariant breaks
- Assertions on shape/columns after merges (`assert len(df_merged) > 0`, `assert 'gdp_2023' in df_merged.columns`)
- Dropping rows with a printed count of what was dropped and why (e.g. the existing `dropna(subset=['2023_gdp'])` is fine because it's intentional and visible)
- Letting `pd.read_csv` raise `FileNotFoundError` naturally — that IS the graceful error

The point: when something breaks, the traceback should name the problem. Never paper over a failure to keep the notebook running.
