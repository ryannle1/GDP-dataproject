"""CS 602 Data Science — Project 3 presentation app.

Run locally:  streamlit run app.py
"""

import streamlit as st

from lib.data import build_merged, load_bank_new, load_bank_train

st.set_page_config(page_title="CS 602 Project", layout="wide")

st.title("Data Project")
st.caption("Select a page from the sidebar.")

st.markdown(
    """
This project covers two distinct analyses on different datasets.

1. **Prosperity ↔ GDP regression** — How do prosperity scores (infrastructure, education, health, …)
   and political regime type relate to a country's 2023 GDP? An OLS model with an interaction term
   between infrastructure access and regime group, validated against high-influence outliers.
2. **Bank term-deposit classification** — Given customer attributes (age, job, balance, prior loans, …),
   predict whether a customer will subscribe to a term deposit.
"""
)

merged = build_merged()
bank_train = load_bank_train()
bank_new = load_bank_new()

col1, col2, col3 = st.columns(3)
col1.metric("Countries (merged)", len(merged))
col2.metric("Bank training rows", f"{len(bank_train):,}")
col3.metric("Bank prediction rows", f"{len(bank_new):,}")

st.info("Use the sidebar to open **Prosperity_GDP** or **Bank_Deposit_Classifier**.")
