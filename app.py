import datetime, json, os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import FEATURES, load_data, train

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

MODEL_PATH, METRICS_PATH, DATA_PATH = "model/car_price_model.pkl", "model/metrics.json", "data/car_data.csv"
THIS_YEAR = datetime.datetime.now().year


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:  # missing file or sklearn version mismatch -> retrain
        train()
        return joblib.load(MODEL_PATH)


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        get_model()
    with open(METRICS_PATH) as f:
        return json.load(f)


def price_range(pipe, X):
    """Spread of individual tree predictions (only for forest models)."""
    model = pipe.named_steps["model"]
    if hasattr(model, "estimators_") and isinstance(model.estimators_, (list, np.ndarray)) and np.ndim(model.estimators_) == 1:
        Xt = pipe.named_steps["pre"].transform(X)
        preds = np.array([t.predict(Xt)[0] for t in model.estimators_])
        return np.percentile(preds, 10), np.percentile(preds, 90)
    return None


pipe = get_model()
df = get_data()
metrics = get_metrics()

st.title("🚗 Car Price Prediction System")
st.caption("Predict the resale value of a used car using machine learning.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Insights", "🧪 Model Performance"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        present_price = st.number_input("Showroom (new) price, in ₹ lakhs", 0.5, 100.0, 8.0, 0.5)
        year = st.slider("Year of purchase", 2000, THIS_YEAR, THIS_YEAR - 5)
        kms = st.number_input("Kilometres driven", 0, 500000, 40000, 1000)
        owner = st.selectbox("Previous owners", [0, 1, 2, 3], help="0 = first-hand")
    with c2:
        fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
        seller = st.selectbox("Seller type", ["Dealer", "Individual"])
        trans = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("Predict price", type="primary"):
        X = pd.DataFrame([{
            "Present_Price": present_price, "Kms_Driven": kms, "Owner": owner,
            "Car_Age": THIS_YEAR - year, "Fuel_Type": fuel,
            "Seller_Type": seller, "Transmission": trans,
        }])[FEATURES]
        pred = max(float(pipe.predict(X)[0]), 0.0)
        st.success(f"Estimated selling price: **₹ {pred:.2f} lakhs**  (≈ ₹ {pred * 100000:,.0f})")
        rng = price_range(pipe, X)
        if rng:
            st.info(f"Likely range: ₹ {rng[0]:.2f} – ₹ {rng[1]:.2f} lakhs")
        st.caption(f"Model: {metrics['best_model']}")

with tab2:
    st.subheader("Dataset overview")
    st.write(f"{len(df):,} rows after cleaning")
    st.dataframe(df.head(20), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Average selling price by car age**")
        st.bar_chart(df.groupby("Car_Age")["Selling_Price"].mean())
        st.markdown("**Average price by fuel type**")
        st.bar_chart(df.groupby("Fuel_Type")["Selling_Price"].mean())
    with b:
        st.markdown("**Selling price vs. kilometres driven**")
        st.scatter_chart(df, x="Kms_Driven", y="Selling_Price")
        st.markdown("**Correlation with selling price**")
        corr = df[["Selling_Price", "Present_Price", "Kms_Driven", "Owner", "Car_Age"]].corr()["Selling_Price"].drop("Selling_Price")
        st.bar_chart(corr)

with tab3:
    st.subheader("Model comparison (20% hold-out test set)")
    st.dataframe(pd.DataFrame(metrics["results"]).T, use_container_width=True)
    st.success(f"Best model: {metrics['best_model']}")
    st.caption("R² closer to 1 is better; MAE and RMSE are in ₹ lakhs (lower is better).")
