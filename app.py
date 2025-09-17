import streamlit as st
import pandas as pd
import pickle
import datetime

# -------------------------
# Load dataset for dropdowns
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    # Extract brand (first word) and model (rest of the string)
    df['brand'] = df['name'].apply(lambda x: str(x).split(" ")[0])
    df['car_model'] = df['name'].apply(lambda x: " ".join(str(x).split(" ")[1:]))
    return df

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    with open("used_car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
    st.title("🚗 Used Car Price Prediction")
    st.write("Fill in the car details below to predict its selling price.")

    # Load data & model
    df = load_data()
    model = load_model()

    # Dropdowns for brand & model
    brand = st.selectbox("Select Brand", sorted(df['brand'].unique()))
    model_options = sorted(df[df['brand'] == brand]['car_model'].unique())
    car_model = st.selectbox("Select Model", model_options)

    # Other features
    year = st.slider("Year of Purchase", int(df['year'].min()), int(df['year'].max()), 2015)
    current_year = datetime.datetime.now().year
    car_age = current_year - year
    st.write(f"📅 Car Age: **{car_age} years**")

    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
    fuel = st.selectbox("Fuel Type", df['fuel'].unique())
    seller_type = st.selectbox("Seller Type", df['seller_type'].unique())
    transmission = st.selectbox("Transmission", df['transmission'].unique())
    owner = st.selectbox("Owner", df['owner'].unique())

    if st.button("🔮 Predict Price"):
        # Prepare input for model
        input_data = pd.DataFrame({
            "name": [f"{brand} {car_model}"],
            "year": [year],
            "car_age": [car_age],
            "km_driven": [km_driven],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "owner": [owner],
        })

        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Estimated Selling Price: ₹ {prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
