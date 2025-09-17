import streamlit as st
import pandas as pd
import pickle

# ===============================
# Load Saved Model & Dataset
# ===============================
with open("used_car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset (for brand/model dropdowns)
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# Extract brand (first word of name)
df["brand"] = df["name"].apply(lambda x: str(x).split()[0])
brands = sorted(df["brand"].unique())

# Required features (must match training pipeline)
required_features = ["name", "km_driven", "fuel", "seller_type",
                     "transmission", "owner", "car_age"]

# ===============================
# Streamlit App Config
# ===============================
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the details of the car to predict its selling price.")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Car Details")

# Brand selection
brand = st.sidebar.selectbox("Select Brand", brands)

# Car models filtered by brand
models_for_brand = sorted(df[df["brand"] == brand]["name"].unique())
car_model = st.sidebar.selectbox("Select Model", models_for_brand)

# Year → Car Age
year = st.sidebar.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1, value=2015)
current_year = 2025
car_age = current_year - year

# Other inputs
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000, value=50000)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# ===============================
# Prepare Input for Model
# ===============================
input_data = pd.DataFrame({
    "name": [car_model],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "car_age": [car_age]
})

# Reorder to match training
input_data = input_data[required_features]

# ===============================
# Prediction
# ===============================
if st.sidebar.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Selling Price: ₹ {prediction:,.0f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed. Error: {e}")

# ===============================
# Optional: Show Dataset Preview
# ===============================
with st.expander("🔍 See Sample Dataset"):
    st.dataframe(df.head(20))
