# app.py (debug-friendly, handles common causes of negative predictions)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

st.set_page_config(page_title="Used Car Price Predictor (Debug)", layout="centered")

# ---------------------------
# Load model + dataset
# ---------------------------
@st.cache_resource
def load_resources():
    with open("used_car_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    # Ensure brand column exists (used earlier while training)
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0])
    return model, df

model, df = load_resources()

# Extract brands/models for UI
brands = sorted(df["brand"].unique())
required_features = ["name", "brand", "km_driven", "fuel", "seller_type",
                     "transmission", "owner", "car_age"]

st.title("🚗 Used Car Price Prediction (debug-friendly)")
st.write("Enter car details and check debug info if prediction looks wrong.")

# ---------------------------
# Sidebar / Inputs
# ---------------------------
st.sidebar.header("Car Details")
brand = st.sidebar.selectbox("Select Brand", brands)
models_for_brand = sorted(df[df["brand"] == brand]["name"].unique())
car_model = st.sidebar.selectbox("Select Model", models_for_brand)

year = st.sidebar.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1, value=2015)
current_year = 2025
car_age = current_year - year

km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=1000000, step=500, value=50000)
fuel = st.sidebar.selectbox("Fuel Type", sorted(df["fuel"].dropna().unique().tolist()))
seller_type = st.sidebar.selectbox("Seller Type", sorted(df["seller_type"].dropna().unique().tolist()))
transmission = st.sidebar.selectbox("Transmission", sorted(df["transmission"].dropna().unique().tolist()))
owner = st.sidebar.selectbox("Owner", sorted(df["owner"].dropna().unique().tolist()))

# Ask user what transform (if any) was applied to target while training
st.sidebar.markdown("**If your training used a transform on target (y), pick it:**")
target_transform = st.sidebar.selectbox("Target transform used during training",
                                        ["None", "log1p", "log", "sqrt", "square", "other / unknown"])

# Let user choose display unit
unit = st.sidebar.radio("Display price in:", ["Rupees (₹)", "Thousands", "Lakhs"])

# Prepare input DataFrame (must match training columns)
input_data = pd.DataFrame({
    "name": [car_model],
    "brand": [brand],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "car_age": [car_age]
})

# Reorder columns if possible, otherwise keep as-is
try:
    input_data = input_data[required_features]
except Exception:
    # If ordering fails, keep input_data as-is but show warning in debug panel
    pass

# ---------------------------
# Utilities / Debug helpers
# ---------------------------
def pipeline_summary(m):
    info = {}
    try:
        if isinstance(m, Pipeline):
            info["is_pipeline"] = True
            info["steps"] = list(m.named_steps.keys())
            # preprocessor columns if present
            pre = m.named_steps.get("preprocessor")
            if pre is not None:
                info["preprocessor_str"] = str(pre)
                try:
                    # list transformer specs (name, columns)
                    transformers = [(t[0], t[2]) for t in pre.transformers_]
                    info["preprocessor_transformers"] = transformers
                except Exception as e:
                    info["preprocessor_transformers_error"] = str(e)
            # model step
            mdl = m.named_steps.get("model")
            info["model_type"] = str(type(mdl))
            # check if model is TransformedTargetRegressor
            info["is_TTR"] = isinstance(mdl, TransformedTargetRegressor)
            if info["is_TTR"]:
                try:
                    info["TTR_transformer"] = str(mdl.transformer)
                except Exception:
                    info["TTR_transformer"] = "unknown"
            # features known on pipeline (if present)
            try:
                info["feature_names_in"] = list(m.feature_names_in_)
            except Exception:
                info["feature_names_in_error"] = "not available"
        else:
            info["is_pipeline"] = False
            info["model_type"] = str(type(m))
    except Exception as e:
        info["summary_error"] = str(e)
    return info

def apply_inverse_transform(raw_val, transform_name):
    """Apply inverse of common y-transforms"""
    if transform_name == "None":
        return raw_val
    if transform_name == "log1p":
        return np.expm1(raw_val)
    if transform_name == "log":
        return np.exp(raw_val)
    if transform_name == "sqrt":
        return raw_val ** 2
    if transform_name == "square":
        return np.sqrt(raw_val)
    return raw_val  # unknown

# ---------------------------
# Prediction action
# ---------------------------
if st.sidebar.button("Predict Price"):
    with st.spinner("Predicting..."):
        try:
            raw_pred = model.predict(input_data)[0]
        except Exception as e:
            st.error("❌ Model predict failed. See debug panel for details.")
            raw_pred = None
            exc = e

        # Show debug / diagnostics
        with st.expander("🧰 Debug info (input + model)"):
            st.write("**Input data passed to model:**")
            st.dataframe(input_data.T)

            st.write("**Model / pipeline summary:**")
            summary = pipeline_summary(model)
            st.json(summary)

            if raw_pred is None:
                st.write("**Exception from predict()**:")
                st.exception(exc)
            else:
                st.write(f"**Raw model output:** {raw_pred!r}")

        # If raw_pred exists, try to create final price
        if raw_pred is not None:
            # First try auto-detection: if model is Pipeline and model.named_steps['model'] is TransformedTargetRegressor,
            # its predict() should already return original y. We still allow manual override via sidebar select.
            tried_inverse = False
            try:
                # detect pipeline + TTR
                if isinstance(model, Pipeline):
                    mdl = model.named_steps.get("model")
                    if isinstance(mdl, TransformedTargetRegressor):
                        # predict() already returns original y value
                        final_val = raw_pred
                        st.info("Model contains TransformedTargetRegressor: predict() should already be in original target scale.")
                        tried_inverse = True
            except Exception:
                pass

            # if not TTR (or user selected specific transform), apply inverse as per user's choice
            if not tried_inverse:
                adjusted = apply_inverse_transform(raw_pred, target_transform)
                # If user selected 'other / unknown' we still show the raw; but show adjusted attempts for log variants:
                if target_transform == "other / unknown":
                    # show common guesses
                    alt1 = np.expm1(raw_pred)   # guess log1p
                    alt2 = np.exp(raw_pred)     # guess log
                    alt3 = raw_pred ** 2        # guess sqrt
                    st.warning("Target transform unknown — showing common inverse attempts below. Choose the correct one from the sidebar for final result.")
                    st.write("expm1 (for log1p):", alt1)
                    st.write("exp (for log):", alt2)
                    st.write("square (for sqrt):", alt3)
                    final_val = raw_pred  # keep raw by default
                else:
                    final_val = adjusted

            # Units conversion
            # Assume model predicts price in Rupees by default. If your model was trained with 'lakhs' set this dropdown to 'Lakhs'
            if unit == "Rupees (₹)":
                display_val = final_val
            elif unit == "Thousands":
                display_val = final_val * 1_000
            elif unit == "Lakhs":
                display_val = final_val * 100_000
            else:
                display_val = final_val

            # Clip negative → 0 with a warning (safer than showing negative money)
            if display_val < 0:
                st.error("⚠️ The final value is negative. This likely means the model was trained with a different target transform/units or is extrapolating.")
                clipped = max(display_val, 0.0)
                st.warning(f"Clipping negative prediction to ₹ {clipped:,.2f} (displayed). Raw value: {display_val:.6f}")
                display_val = clipped
            else:
                st.success(f"💰 Estimated Selling Price: ₹ {display_val:,.2f}")

            # Show both raw and final numbers for traceability
            st.write("---")
            st.write(f"**Raw model output:** {raw_pred!r}")
            st.write(f"**After inverse-transform & units conversion:** {display_val!r}")

            # Helpful suggestions for the user
            with st.expander("⚡ If the result still looks wrong — next steps"):
                st.write("""
                * 1) If you used `np.log1p(y)` during training, set **Target transform** → **log1p** in the sidebar.
                * 2) If you used `np.log(y)`, set **log**. If you used `sqrt`, set **sqrt**.
                * 3) If your training target was in **lakhs**, pick **Lakhs** in the unit selector.
                * 4) If model is producing strange values because of new/unseen categories, retrain the pipeline including those categories or save the fitted encoders.
                * 5) Best fix: retrain and save your full pipeline including any `TransformedTargetRegressor` (this makes deploy trivial).
                """)
