from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.calibration import CalibrationDisplay
from src.evaluation import evaluate_ranking


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Bank Campaign Prioritisation",
    layout="wide",
)

st.title("Bank Campaign Prioritisation")
st.caption("Explainable customer targeting under a limited contact budget")

st.warning(
    "Historical simulation: these results do not establish additional "
    "subscriptions caused by contact or increased revenue. "
    "Probability estimates still underestimate subscriptions in the test period."
)

model_path = ROOT / "artifacts" / "campaign_models.joblib"
data_path = ROOT / "data" / "raw" / "bank-full.csv"

if not model_path.exists() or not data_path.exists():
    st.error(
        "Add data/raw/bank-full.csv and run python -m src.train first."
    )
    st.stop()


@st.cache_resource
def load_bundle(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path, sep=";")


bundle = load_bundle(str(model_path), model_path.stat().st_mtime)
data = load_data(str(data_path), data_path.stat().st_mtime)

# Reuse the same held-out test period as the notebook.
test_start = int(len(data) * 0.80)
X_test = data[bundle["features"]].iloc[test_start:].copy()
y_test = data["y"].map({"no": 0, "yes": 1}).iloc[test_start:]

available_models = {
    **bundle["models"],
    bundle["final_model_name"]: bundle["final_model"],
}

names = list(available_models)

model_name = st.sidebar.selectbox(
    "Model",
    names,
    index=names.index(bundle["final_model_name"]),
)

capacity_percent = st.sidebar.slider(
    "Contact capacity (%)",
    min_value=1,
    max_value=100,
    value=10,
)

st.sidebar.caption(
    "The calibrated random forest was chosen using development data. "
    "Comparisons here do not change that original choice."
)

model = available_models[model_name]
probabilities = model.predict_proba(X_test)[:, 1]

result = evaluate_ranking(
    y_test,
    probabilities,
    capacity=capacity_percent / 100,
)

columns = st.columns(4)
columns[0].metric("Records selected", f"{result['selected']:,}")
columns[1].metric("Historical precision", f"{result['precision']:.2%}")
columns[2].metric("Historical recall", f"{result['recall']:.2%}")
columns[3].metric("Lift over random", f"{result['lift']:.2f}×")

st.caption(
    f"Historical subscribers captured: {result['captured']:,} | "
    f"Test subscription rate: {y_test.mean():.2%} | "
    f"Brier score: {result['brier']:.4f}"
)

# Disagreement across the three distinct model families.
family_predictions = pd.DataFrame(
    {
        name: estimator.predict_proba(X_test)[:, 1]
        for name, estimator in bundle["models"].items()
    },
    index=X_test.index,
)

percentiles = family_predictions.rank(pct=True, method="average")
rank_spread = percentiles.max(axis=1) - percentiles.min(axis=1)

ranked = X_test.copy()
ranked.insert(0, "record_id", ranked.index)
ranked["predicted_probability"] = probabilities
ranked["rank_disagreement"] = rank_spread
ranked = ranked.sort_values(
    "predicted_probability",
    ascending=False,
    kind="stable",
)

ranked.insert(0, "priority_rank", np.arange(1, len(ranked) + 1))
ranked["models_disagree"] = ranked["rank_disagreement"] >= 0.20

boundary_band = max(1, int(len(ranked) * 0.01))
ranked["near_budget_cutoff"] = (
    (ranked["priority_rank"] - result["selected"]).abs() <= boundary_band
) & (result["selected"] < len(ranked))

ranked["review_flag"] = (
    ranked["models_disagree"] | ranked["near_budget_cutoff"]
)

selected = ranked.head(result["selected"])

st.subheader("Simulated contact list")
st.caption(
    "IDs refer to dataset records, not verified unique customers. "
    "Review flags are heuristics, not confidence intervals."
)
st.dataframe(selected, hide_index=True)

st.download_button(
    "Download contact list",
    data=selected.to_csv(index=False).encode("utf-8"),
    file_name="simulated_contact_list.csv",
    mime="text/csv",
)

st.subheader("How contact capacity affects capture")

capacities = np.arange(1, 101) / 100
captures = [
    evaluate_ranking(y_test, probabilities, capacity)["captured"]
    for capacity in capacities
]
random_capture = [
    int(np.floor(len(y_test) * capacity)) * y_test.mean()
    for capacity in capacities
]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(capacities * 100, captures, label=model_name)
ax.plot(
    capacities * 100,
    random_capture,
    "--",
    label="Random selection: expected",
)
ax.axvline(capacity_percent, color="grey", linestyle=":")
ax.set_xlabel("Contact capacity (%)")
ax.set_ylabel("Historical subscribers captured")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Explain a selected record")

record_id = st.selectbox(
    "Record ID",
    selected["record_id"].tolist(),
)

customer = X_test.loc[[record_id]]
original_probability = model.predict_proba(customer)[0, 1]

effects = {}
for feature in bundle["features"]:
    changed = customer.copy()
    changed[feature] = bundle["reference_values"][feature]
    changed_probability = model.predict_proba(changed)[0, 1]
    effects[feature] = (
        original_probability - changed_probability
    ) * 100

st.write(f"Predicted subscription probability: {original_probability:.2%}")
st.dataframe(customer, hide_index=True)

effects = pd.Series(effects).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(effects.index, effects.values)
ax.axvline(0, color="grey")
ax.set_xlabel("Difference from training-reference replacement (percentage points)")
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Sensitivity replaces one feature at a time with its training median "
    "or most common category. Effects do not add up to the prediction, "
    "may involve unrealistic combinations, and are not causal effects."
)

with st.expander("Probability calibration"):
    fig, ax = plt.subplots(figsize=(6, 4))
    CalibrationDisplay.from_predictions(
        y_test,
        probabilities,
        n_bins=8,
        strategy="quantile",
        name=model_name,
        ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Method and limitations"):
    st.write(
        "Original-order split: 60% training, 20% validation, 20% testing. "
        "Calibration uses the first half of validation. "
        "Inputs are age, job, marital status, education, default, balance, "
        "housing loan and personal loan. Call duration and current-campaign "
        "information are excluded."
    )
    st.write(
        "Subscription rates change substantially between periods. "
        "Logistic regression ranked better on the final test period despite "
        "random forest winning validation. Demographic inputs also require "
        "fairness assessment before any real-world use."
    )