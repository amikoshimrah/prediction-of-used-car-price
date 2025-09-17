import streamlit as st
import pandas as pd
import pickle

# ===============================
# Load Saved Model & Dataset
# ===============================
with open("used_car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset for brand/model dropdowns
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# Extract brand from name
df["brand"] = df["name"].apply(lambda x: str(x).split()[0])
brands = sorted(df["brand"].unique())

# Required features (must match training)
required_features = ["name", "brand", "km_driven", "fuel", "seller_type",
                     "transmission", "owner", "car_age"]

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the details of the car to predict its selling price.")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Car Details")

# Brand & Model
brand = st.sidebar.selectbox("Select Brand", brands)
models_for_brand = sorted(df[df["brand"] == brand]["name"].unique())
car_model = st.sidebar.selectbox("Select Model", models_for_brand)

# Year → Car Age
year = st.sidebar.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1, value=2015)
current_year = 2025
car_age = current_year - year

# Other details
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=500, value=50000)
fuel = st.sidebar.selectbox("Fuel Type", sorted(df["fuel"].dropna().unique().tolist()))
seller_type = st.sidebar.selectbox("Seller Type", sorted(df["seller_type"].dropna().unique().tolist()))
transmission = st.sidebar.selectbox("Transmission", sorted(df["transmission"].dropna().unique().tolist()))
owner = st.sidebar.selectbox("Owner", sorted(df["owner"].dropna().unique().tolist()))

# ===============================
# Prepare Input for Model
# ===============================
input_data = pd.DataFrame({
    "name": [car_model],
    "brand": [brand],
    "km_driven": [km_driven],   # raw value, pipeline applies log1p
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "car_age": [car_age]
})

# Reorder columns
input_data = input_data[required_features]

# ===============================
# Prediction
# ===============================
if st.sidebar.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]   # already in rupees because of TransformedTargetRegressor
        if prediction < 0:
            st.error("⚠️ Model predicted a negative value. Please check training or input ranges.")
        else:
            st.success(f"💰 Estimated Selling Price: ₹ {prediction:,.0f}")
    except Exception as e:
        st.error(f"❌ Prediction failed. Error: {e}")

# ===============================
# Optional: Dataset Preview
# ===============================
with st.expander("🔍 See Sample Dataset"):
    st.dataframe(df.head(20))
