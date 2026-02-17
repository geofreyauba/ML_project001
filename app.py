import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart ML Trainer",
    page_icon="🤖",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.section {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.metric {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🤖 Smart ML App")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

# ================= LOAD DATA =================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

# ================= MAIN =================
if uploaded_file:
    df = load_data(uploaded_file)

    st.title("🤖 Automated Machine Learning Dashboard")

    # ================= DATA OVERVIEW =================
    with st.container():
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("📂 Dataset Overview")
        st.write("Shape:", df.shape)
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= TARGET SELECTION =================
    target = st.selectbox("🎯 Select Target Column", df.columns)

    # ================= PREPROCESS =================
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical features
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object":
        task_type = "classification"
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        task_type = "regression"

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ================= MODELS =================
    if task_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier()
        }

    # ================= TRAIN =================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("⚙️ Model Training Results")

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "regression":
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results[name] = {"RMSE": rmse, "R2": r2}

            st.markdown(f"### 🔹 {name}")
            st.write(f"RMSE: {rmse:.3f}")
            st.write(f"R² Score: {r2:.3f}")

        else:
            acc = accuracy_score(y_test, y_pred)
            results[name] = {"Accuracy": acc}

            st.markdown(f"### 🔹 {name}")
            st.write(f"Accuracy: {acc:.3f}")
            st.text(classification_report(y_test, y_pred))

            fig, ax = plt.subplots()
            sns.heatmap(
                confusion_matrix(y_test, y_pred),
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= PREDICTION =================
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.subheader("🔮 Make Prediction")

    user_input = []
    for col in df.drop(columns=[target]).columns:
        value = st.number_input(f"{col}", value=0.0)
        user_input.append(value)

    user_input = scaler.transform([user_input])

    if st.button("Predict with All Models"):
        for name, model in models.items():
            prediction = model.predict(user_input)
            st.success(f"{name} Prediction: {prediction[0]}")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.title("🤖 Smart ML Training App")
    st.info("Upload a CSV dataset to begin.")
