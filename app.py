"""Streamlit front-end: evaluate saved Mobile Price classifiers on an uploaded test CSV.

The models are loaded from model/*.joblib and used for prediction only.
Nothing is ever re-fitted on the uploaded data.
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from common import (
    CLASS_LABELS, FEATURES, MODEL_DIR, MODEL_FILES, TARGET, TEST_CSV,
    evaluate_model,
)

st.set_page_config(page_title="Phone Price-Band Lab", page_icon="📱", layout="wide")


@st.cache_resource
def load_model(display_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[display_name])


@st.cache_data
def read_bundled_test_data():
    return pd.read_csv(TEST_CSV)


def validate(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"The CSV is missing required column(s): {', '.join(missing)}")
        return False
    if df[FEATURES + [TARGET]].isnull().any().any():
        st.error("The CSV contains empty cells. Please upload a clean test file.")
        return False
    unknown = sorted(set(df[TARGET].unique().tolist()) - set(CLASS_LABELS))
    if unknown:
        st.error(f"Unexpected target values {unknown}; expected 0, 1, 2 or 3.")
        return False
    return True


# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
use_sample = st.sidebar.checkbox(
    "Use the bundled test_data.csv", value=False, disabled=uploaded is not None
)

st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES))

st.sidebar.caption(
    "Expected columns: " + ", ".join(FEATURES) + f", plus the target `{TARGET}`."
)

# --------------------------------------------------------------------- main
st.title("📱 Phone Price-Band Lab")
st.write(
    "Pick a trained classifier, give it a held-out test file, and see how well it "
    "predicts a phone's **price band** (0 = Low, 1 = Medium, 2 = High, 3 = Very high) "
    "from its hardware specs. Models are pre-trained: nothing is re-fitted here."
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_sample:
    data = read_bundled_test_data()
else:
    st.info("Upload a test CSV (or tick the bundled sample) in the sidebar to begin.")
    st.stop()

if not validate(data):
    st.stop()

X, y = data[FEATURES], data[TARGET]
st.success(f"Loaded {len(data):,} rows. Class counts: "
           + ", ".join(f"{CLASS_LABELS[k]}={v}" for k, v in sorted(y.value_counts().items())))

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20))

# ---------------------------------------------------- selected model results
model = load_model(model_name)
y_pred, metrics = evaluate_model(model, X, y)

st.subheader(f"Results: {model_name}")
cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{round(value, 4):.4f}")  # same rounding as model/metrics.csv

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    labels = sorted(CLASS_LABELS)
    cm = confusion_matrix(y, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax,
        xticklabels=[CLASS_LABELS[i] for i in labels],
        yticklabels=[CLASS_LABELS[i] for i in labels],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y, y_pred, labels=labels,
        target_names=[CLASS_LABELS[i] for i in labels],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3))

# ----------------------------------------------- all models side by side
st.subheader("All five models on this test data")
rows = []
for name in MODEL_FILES:
    _, m = evaluate_model(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(
    comparison.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
)
st.caption("Green = best value in each column. Precision, recall and F1 are weighted "
           "averages; AUC is one-vs-rest weighted.")
