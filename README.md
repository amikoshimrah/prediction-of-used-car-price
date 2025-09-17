This project predicts the selling price of used cars based on historical listings from the CarDekho dataset.

Key steps covered:

Data Preprocessing: Handling duplicates, outliers, and missing values.

Feature Engineering: Extracting car age, brand, and model from raw columns.

Exploratory Data Analysis (EDA): Visualizing distributions and correlations with selling price.

Model Training: Implementing regression models such as Linear Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and CatBoost.

Target Transformation: Applying log1p to normalize skewed price distribution.

Model Evaluation: Using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

Deployment: Exporting the trained pipeline as a .pkl file and building a simple Streamlit web app for interactive predictions.

The result is a practical tool that can help car owners, buyers, and dealerships estimate fair market values.

📂 Dataset

Source: CarDekho Used Car Dataset

Features:

name (car name including brand + model)

year (manufacturing year)

km_driven (distance driven)

fuel, seller_type, transmission, owner

selling_price (target variable, INR)
