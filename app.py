import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI House Price Predictor", layout="wide")

st.title("🏠 California House Price Prediction")
st.write("Predict house prices using Machine Learning")

# ----------- LOAD DATA (Cached) -----------
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

df = load_data()

# ----------- TRAIN MODEL (Cached) -----------
@st.cache_resource
def train_model(data):

    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)

    return model

model = train_model(df)

# ----------- SIDEBAR INPUT -----------
st.sidebar.header("Enter House Details")

longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -122.23)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.88)
housing_median_age = st.sidebar.slider("House Age", 1, 100, 41)
total_rooms = st.sidebar.slider("Total Rooms", 100, 10000, 880)
total_bedrooms = st.sidebar.slider("Total Bedrooms", 50, 3000, 129)
population = st.sidebar.slider("Population", 50, 5000, 322)
households = st.sidebar.slider("Households", 50, 3000, 126)
median_income = st.sidebar.slider("Median Income", 0.5, 15.0, 8.3)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    df["ocean_proximity"].unique()
)

input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    'ocean_proximity': [ocean_proximity]
})

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")