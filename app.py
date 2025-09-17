import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
model = pickle.load(open("used_car_price_model.pkl", "rb"))

# Load dataset for brand + model mapping
data = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# Extract brand (first word) and model (rest)
data['brand'] = data['name'].apply(lambda x: x.split()[0])
data['model_name'] = data['name'].apply(lambda x: " ".join(x.split()[1:]))

brands = sorted(data['brand'].unique())

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Prediction App")
st.write("Enter the details of the car to predict its selling price.")

# --- Inputs ---
brand = st.selectbox("Car Brand", brands)

# Filter models for selected brand
filtered_models = sorted(data[data['brand'] == brand]['model_name'].unique())
model_name = st.selectbox("Car Model", filtered_models)

year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500)
owner = st.selectbox("Owner", data['owner'].unique())  # use same unique values from dataset
fuel_type = st.selectbox("Fuel Type", data['fuel'].unique())
seller_type = st.selectbox("Seller Type", data['seller_type'].unique())
transmission = st.selectbox("Transmission", data['transmission'].unique())

# --- Build 'name' column exactly like dataset ---
full_name = f"{brand} {model_name}"

# Create input dataframe with SAME columns as training dataset
input_df = pd.DataFrame([{
    "name": full_name,
    "year": year,
    "km_driven": kms_driven,
    "fuel": fuel_type,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner
}])

# --- Prediction ---
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        if prediction < 0:
            st.error("Sorry, this car cannot be sold.")
        else:
            st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
