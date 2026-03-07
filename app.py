import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline (preprocessing + model)
@st.cache_resource
def load_model():
    return joblib.load("pricing_model.pkl")

model = load_model()

st.set_page_config(page_title="Dynamic Pricing Engine", layout="centered")
st.title("🛒 Dynamic Pricing Engine")
st.markdown("Enter product details to predict the optimal price.")

# Input fields in the order the model expects
col1, col2 = st.columns(2)

with col1:
    competitor_price = st.number_input(
        "Competitor Pricing ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01
    )
    demand_forecast = st.number_input(
        "Demand Forecast", min_value=0.0, max_value=500.0, value=100.0, step=0.1
    )
    inventory_level = st.number_input(
        "Inventory Level", min_value=0, max_value=1000, value=200, step=1
    )
    discount = st.number_input(
        "Discount (%)", min_value=0, max_value=100, value=10, step=1
    )

with col2:
    seasonality = st.selectbox(
        "Seasonality",
        options=["Autumn", "Spring", "Summer", "Winter"],
        index=0,
    )
    holiday_promo = st.radio(
        "Holiday / Promotion", options=[0, 1], format_func=lambda x: "Yes" if x else "No", index=0
    )
    units_ordered = st.number_input(
        "Units Ordered", min_value=0, max_value=500, value=50, step=1
    )

# Predict when button is clicked
if st.button("Predict Price", type="primary"):
    # Build input dataframe exactly as the pipeline expects
    input_data = pd.DataFrame({
        "Competitor Pricing": [competitor_price],
        "Demand Forecast": [demand_forecast],
        "Inventory Level": [inventory_level],
        "Discount": [discount],
        "Seasonality": [seasonality],
        "Holiday/Promotion": [holiday_promo],
        "Units Ordered": [units_ordered],
    })

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 **Recommended Price: ${prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Model trained on retail data with features: competitor pricing, demand, inventory, discount, seasonality, holiday/promotion, and units ordered.")