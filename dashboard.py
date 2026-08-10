import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="AI Future Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Fixed, built-in dataset — no upload, no user input required.
# This guarantees the dashboard always loads and works identically
# for anyone who opens the link.
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
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


df = load_data()
feature_cols = ["size_sqft", "bedrooms", "age_years", "location_score"]
target_col = "price"

X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔮 AI Future Prediction Dashboard")
st.caption(
    "A live machine learning dashboard: it loads housing data, trains a "
    "prediction model automatically, and shows the results below."
)

# ---------------------------------------------------------------------------
# Train models (always runs the same way — fully reliable)
# ---------------------------------------------------------------------------
candidates = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Linear Regression": LinearRegression(),
}

results = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = (model, r2_score(y_test, preds), preds)

best_name = max(results, key=lambda n: results[n][1])
best_model, best_score, best_preds = results[best_name]
mae = mean_absolute_error(y_test, best_preds)

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
st.subheader("📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", best_name)
c2.metric("R² score", f"{best_score:.3f}")
c3.metric("Avg. error", f"${mae:,.0f}")
c4.metric("Rows of data", f"{len(df)}")

# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
st.subheader("📈 Actual vs Predicted Prices")
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.scatter(y_test, best_preds, alpha=0.6, color="#2E75B6")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax1.plot(lims, lims, "r--", label="Perfect prediction")
ax1.set_xlabel("Actual Price ($)")
ax1.set_ylabel("Predicted Price ($)")
ax1.legend()
st.pyplot(fig1)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🔑 What Drives the Price Most")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    importances.plot(kind="barh", ax=ax2, color="#1F3A5F")
    st.pyplot(fig2)

with col_b:
    st.subheader("🏠 Price Distribution")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(df["price"], bins=25, color="#2E75B6")
    ax3.set_xlabel("Price ($)")
    ax3.set_ylabel("Number of houses")
    st.pyplot(fig3)

# ---------------------------------------------------------------------------
# Live "try a prediction" — the only interactive part
# ---------------------------------------------------------------------------
st.subheader("🧮 Try a Live Prediction")
st.write("Move the sliders and see the model predict a price instantly.")

c1, c2, c3, c4 = st.columns(4)
size = c1.slider("Size (sqft)", 600, 3500, 1500)
bedrooms = c2.slider("Bedrooms", 1, 5, 3)
age = c3.slider("Age (years)", 0, 40, 10)
location = c4.slider("Location score", 1, 10, 6)

input_df = pd.DataFrame(
    [[size, bedrooms, age, location]], columns=feature_cols
)
prediction = best_model.predict(input_df)[0]
st.success(f"### Predicted Price: ${prediction:,.0f}")

st.markdown("---")
st.caption("Built with Streamlit • scikit-learn")
