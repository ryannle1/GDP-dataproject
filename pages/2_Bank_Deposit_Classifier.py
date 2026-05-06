"""Workstream B: Bank term-deposit classifier."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from lib.data import load_bank_new, load_bank_train
from lib.models import fit_bank_classifier, predict_bank_new

st.set_page_config(page_title="Bank Deposit Classifier", layout="wide")
st.title("Bank term-deposit classifier")
st.caption("Predict whether a customer subscribes to a term deposit (yes/no).")

bank_train = load_bank_train()
bank_new = load_bank_new()

with st.sidebar:
    st.header("Model")
    model_label = st.radio("Estimator", ["Logistic Regression", "Decision Tree"])
    model_kind = "logreg" if model_label == "Logistic Regression" else "tree"

    test_size = st.slider("Test split", 0.10, 0.40, 0.20, 0.05)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

st.subheader(f"Training data — {len(bank_train):,} rows")
left, right = st.columns([2, 1])
with left:
    st.dataframe(bank_train.head(20), use_container_width=True, hide_index=True)
with right:
    counts = bank_train["deposit"].value_counts().rename_axis("deposit").reset_index(name="count")
    st.bar_chart(counts.set_index("deposit"))

model, X_test, y_test, feature_columns = fit_bank_classifier(
    bank_train, model_kind=model_kind, test_size=test_size,
)

proba_test = model.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= threshold).astype(int)

st.subheader("Held-out evaluation")

cm = confusion_matrix(y_test, pred_test)
auc = roc_auc_score(y_test, proba_test)

c1, c2 = st.columns(2)
with c1:
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["no", "yes"]).plot(
        ax=ax_cm, cmap="Blues", colorbar=False,
    )
    ax_cm.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with c2:
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label=f"{model_label} (AUC={auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
    plt.close(fig_roc)

report = classification_report(
    y_test, pred_test, target_names=["no", "yes"], output_dict=True, zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader(f"Predictions for bank_new.csv — {len(bank_new):,} rows")
predictions = predict_bank_new(model, bank_new, feature_columns, threshold=threshold)
st.dataframe(predictions, use_container_width=True, hide_index=True)

submit_csv = predictions[["customer_id", "predicted_deposit"]].rename(
    columns={"predicted_deposit": "deposit"}
).to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download submission CSV (customer_id, deposit)",
    data=submit_csv,
    file_name="bank_submit.csv",
    mime="text/csv",
)
