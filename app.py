import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load trained model
model = pickle.load(open("used_car_price_model.pkl", "rb"))

# Load dataset to extract brands
data = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# Extract brand names (first word of 'name')
data['brand'] = data['name'].apply(lambda x: x.split()[0])
brands = sorted(data['brand'].unique())

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the details of the car to predict its selling price.")

# --- Inputs ---
brand = st.selectbox("Car Brand", brands)
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, step=0.1, format="%.2f")
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500)   # ✅ correct variable
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# --- Preprocessing ---
fuel_petrol = 1 if fuel_type == "Petrol" else 0
fuel_diesel = 1 if fuel_type == "Diesel" else 0
seller_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

car_age = 2025 - year

# Encode brand as one-hot like training
brand_dummies = pd.get_dummies(data['brand'])
if brand not in brand_dummies.columns:
    st.warning("⚠️ This brand was not seen during training. Prediction may be inaccurate.")
    brand_vector = np.zeros(len(brand_dummies.columns))
else:
    brand_vector = np.zeros(len(brand_dummies.columns))
    brand_vector[list(brand_dummies.columns).index(brand)] = 1

# ✅ Fixed: using kms_driven (not km_driven)
features = np.array([[present_price, kms_driven, owner, car_age,
                      fuel_diesel, fuel_petrol, seller_individual, transmission_manual, *brand_vector]])

# --- Prediction ---
if st.button("Predict Price"):
    prediction = model.predict(features)[0]
    if prediction < 0:
        st.error("Sorry, this car cannot be sold.")
    else:
        st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
