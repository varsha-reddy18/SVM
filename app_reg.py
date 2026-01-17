import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config("SVM Regression", layout="wide")

def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style_reg.css")

st.markdown('<div class="app-title">SVM Regression Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">End-to-End Machine Learning Pipeline</div>', unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

st.sidebar.header("SVM Hyperparameters")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

st.markdown('<div class="section-header">Step 1: Data Ingestion</div>', unsafe_allow_html=True)

source = st.radio("Choose Data Source", ["Download Iris Dataset", "Upload CSV Dataset"])

df = None

if source == "Download Iris Dataset":
    if st.button("Download Dataset"):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(raw_path)
        st.markdown('<div class="success-box">Dataset downloaded successfully</div>', unsafe_allow_html=True)

if source == "Upload CSV Dataset":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        upload_path = os.path.join(RAW_DIR, uploaded_file.name)
        df.to_csv(upload_path, index=False)
        st.markdown('<div class="success-box">Dataset uploaded successfully</div>', unsafe_allow_html=True)

if df is not None:
    st.markdown('<div class="section-header">Step 2: Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if df is not None:
    st.markdown('<div class="section-header">Step 3: Data Cleaning</div>', unsafe_allow_html=True)

    strategy = st.selectbox("Missing Value Strategy", ["Mean", "Median", "Drop Rows"])
    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.session_state.df_clean = df_clean
    st.markdown('<div class="success-box">Data cleaning completed</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Step 4: Save Cleaned Data</div>', unsafe_allow_html=True)

if st.button("Save Cleaned Dataset"):
    if "df_clean" in st.session_state:
        filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(CLEANED_DIR, filename)
        st.session_state.df_clean.to_csv(path, index=False)
        st.success(f"Saved at {path}")
    else:
        st.error("No cleaned data available")

st.markdown('<div class="section-header">Step 5: Load Cleaned Dataset</div>', unsafe_allow_html=True)

files = os.listdir(CLEANED_DIR)
if files:
    selected = st.selectbox("Select Cleaned Dataset", files)
    df_model = pd.read_csv(os.path.join(CLEANED_DIR, selected))
    st.dataframe(df_model.head())

if 'df_model' in locals():
    st.markdown('<div class="section-header">Step 6: Train SVM Regression</div>', unsafe_allow_html=True)

    target = st.selectbox(
        "Select Target Column (Numeric Only)",
        df_model.select_dtypes(include=np.number).columns
    )

    X = df_model.drop(columns=[target]).select_dtypes(include=np.number)
    y = df_model[target]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = SVR(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.markdown(
        f"""
        <div class="card">
            <h3>Model Performance</h3>
            <p><b>R² Score:</b> {r2:.3f}</p>
            <p><b>Mean Squared Error:</b> {mse:.3f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
