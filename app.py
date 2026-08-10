import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_absolute_error
)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(
    page_title="AI Future Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔮 AI Future Prediction Dashboard")
st.caption(
    "Upload a dataset, pick a target column, and the app automatically "
    "chooses a suitable machine learning model, generates predictions, "
    "and produces graphs and a downloadable PDF report."
)

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    st.markdown("---")
    st.caption(
        "No file? Try the built-in sample dataset to see how the "
        "dashboard works."
    )
    use_sample = st.checkbox("Use sample dataset (house prices)")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_sample():
    rng = np.random.default_rng(42)
    n = 300
    size = rng.integers(600, 3500, n)
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 40, n)
    location_score = rng.integers(1, 10, n)
    price = (
        size * 120
        + bedrooms * 8000
        - age * 500
        + location_score * 4000
        + rng.normal(0, 8000, n)
    )
    return pd.DataFrame(
        {
            "size_sqft": size,
            "bedrooms": bedrooms,
            "age_years": age,
            "location_score": location_score,
            "price": price.round(0),
        }
    )


df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
elif use_sample:
    df = load_sample()

if df is None:
    st.info("⬅️ Upload a dataset or tick **Use sample dataset** in the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing values", int(df.isna().sum().sum()))

# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------
st.subheader("🎯 2. Choose What to Predict")
target_col = st.selectbox("Target column (what you want to predict)", df.columns)

feature_cols = [c for c in df.columns if c != target_col]
feature_cols = st.multiselect(
    "Feature columns (used to make the prediction)",
    feature_cols,
    default=feature_cols,
)

if not feature_cols:
    st.warning("Select at least one feature column.")
    st.stop()

data = df[feature_cols + [target_col]].dropna()

if data.shape[0] < 20:
    st.warning("Not enough clean rows to train a reliable model (need at least ~20).")
    st.stop()

# Encode categorical features
X = data[feature_cols].copy()
encoders = {}
for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

y = data[target_col].copy()
is_classification = (y.dtype == object) or (y.nunique() <= 10 and y.dtype != float)

target_encoder = None
if is_classification and y.dtype == object:
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))

# ---------------------------------------------------------------------------
# Model selection (automatic)
# ---------------------------------------------------------------------------
st.subheader("🤖 3. Model Selection")
task_label = "Classification" if is_classification else "Regression"
st.write(f"Detected task type: **{task_label}** (based on the target column)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if is_classification:
    candidates = {
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }
else:
    candidates = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
        "Linear Regression": LinearRegression(),
    }

results = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if is_classification:
        score = f1_score(y_test, preds, average="weighted")
    else:
        score = r2_score(y_test, preds)
    results[name] = (model, score, preds)

best_name = max(results, key=lambda n: results[n][1])
best_model, best_score, best_preds = results[best_name]

st.success(f"Best model selected automatically: **{best_name}**")

score_cols = st.columns(len(results))
for col, (name, (model, score, preds)) in zip(score_cols, results.items()):
    label = "F1 score" if is_classification else "R² score"
    col.metric(name, f"{score:.3f}", label)

if is_classification:
    acc = accuracy_score(y_test, best_preds)
    st.write(f"Accuracy of best model on test data: **{acc:.1%}**")
else:
    mae = mean_absolute_error(y_test, best_preds)
    st.write(f"Mean Absolute Error of best model on test data: **{mae:,.2f}**")

# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
st.subheader("📊 4. Prediction Graphs")

fig1, ax1 = plt.subplots(figsize=(6, 4))
if is_classification:
    labels = (
        target_encoder.inverse_transform(sorted(set(y_test)))
        if target_encoder
        else sorted(set(y_test))
    )
    counts_actual = pd.Series(y_test).value_counts().sort_index()
    counts_pred = pd.Series(best_preds).value_counts().sort_index()
    idx = sorted(set(counts_actual.index) | set(counts_pred.index))
    width = 0.35
    x = np.arange(len(idx))
    ax1.bar(x - width / 2, [counts_actual.get(i, 0) for i in idx], width, label="Actual")
    ax1.bar(x + width / 2, [counts_pred.get(i, 0) for i in idx], width, label="Predicted")
    ax1.set_xticks(x)
    ax1.set_xticklabels(idx)
    ax1.set_title("Actual vs Predicted class counts")
    ax1.legend()
else:
    ax1.scatter(y_test, best_preds, alpha=0.6, color="#2E75B6")
    lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
    ax1.plot(lims, lims, "r--", label="Ideal fit")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted values")
    ax1.legend()

st.pyplot(fig1)

fig2 = None
if hasattr(best_model, "feature_importances_"):
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    importances.plot(kind="barh", ax=ax2, color="#1F3A5F")
    ax2.set_title("Feature importance")
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# Live prediction on new input
# ---------------------------------------------------------------------------
st.subheader("🧮 5. Try a Live Prediction")
st.write("Enter values below to get an instant prediction from the trained model.")

input_data = {}
cols = st.columns(min(3, len(feature_cols)))
for i, col_name in enumerate(feature_cols):
    with cols[i % len(cols)]:
        if col_name in encoders:
            options = list(encoders[col_name].classes_)
            val = st.selectbox(col_name, options, key=f"in_{col_name}")
            input_data[col_name] = encoders[col_name].transform([val])[0]
        else:
            default_val = float(data[col_name].mean())
            val = st.number_input(col_name, value=default_val, key=f"in_{col_name}")
            input_data[col_name] = val

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([input_data])[feature_cols]
    pred = best_model.predict(input_df)[0]
    if target_encoder is not None:
        pred = target_encoder.inverse_transform([int(pred)])[0]
    st.success(f"Predicted {target_col}: **{pred}**")

# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
st.subheader("📥 6. Download PDF Report")


def build_pdf():
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Future Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Task type: {task_label}", styles["Normal"]))
    story.append(Paragraph(f"Best model: {best_name}", styles["Normal"]))
    metric_label = "F1 score" if is_classification else "R² score"
    story.append(Paragraph(f"{metric_label}: {best_score:.3f}", styles["Normal"]))
    story.append(Spacer(1, 12))

    table_data = [["Rows", "Columns", "Target"], [str(df.shape[0]), str(df.shape[1]), target_col]]
    t = Table(table_data)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 20))

    img_buf1 = io.BytesIO()
    fig1.savefig(img_buf1, format="png", dpi=150, bbox_inches="tight")
    img_buf1.seek(0)
    story.append(Image(img_buf1, width=15 * cm, height=10 * cm))

    if fig2 is not None:
        story.append(Spacer(1, 16))
        img_buf2 = io.BytesIO()
        fig2.savefig(img_buf2, format="png", dpi=150, bbox_inches="tight")
        img_buf2.seek(0)
        story.append(Image(img_buf2, width=15 * cm, height=10 * cm))

    doc.build(story)
    buf.seek(0)
    return buf


pdf_buffer = build_pdf()
st.download_button(
    "Download PDF report",
    data=pdf_buffer,
    file_name="AI_Future_Prediction_Report.pdf",
    mime="application/pdf",
)

st.markdown("---")
st.caption("Built with Streamlit • scikit-learn • ReportLab")
