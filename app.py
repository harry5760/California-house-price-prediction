import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: white;
}

.subtitle {
    font-size: 20px;
    text-align: center;
    color: gray;
}

.card {
    background-color: #1c1f26;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
}

.price {
    font-size: 45px;
    color: #00ff88;
    text-align: center;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(data):

    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
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

# ---------------- HEADER ----------------
st.markdown('<p class="title">🏠 AI House Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict California house prices using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🏠 Enter House Details")

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

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 House Location")
    st.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))

with col2:
    st.subheader("📊 Input Summary")
    st.dataframe(input_data, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

center = st.columns([1,2,1])

with center[1]:
    predict = st.button("🚀 Predict Price", use_container_width=True)

# ---------------- PREDICTION ----------------
if predict:
    prediction = model.predict(input_data)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <p style="text-align:center; color:white; font-size:22px;">
        Estimated House Price
        </p>
        <p class="price">
        ${prediction:,.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
---
<center>
Built with ❤️ using Machine Learning, Python, Scikit-Learn & Streamlit  
By Mohammad Haris
</center>
""", unsafe_allow_html=True)
