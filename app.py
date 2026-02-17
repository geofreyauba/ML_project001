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
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.section {
    background: #1a1a24;
    border: 1px solid #2e2e42;
    padding: 28px;
    border-radius: 14px;
    margin-bottom: 24px;
}

.task-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 10px;
}

.badge-regression {
    background: #1e3a5f;
    color: #60aaff;
    border: 1px solid #2d5a9e;
}

.badge-classification {
    background: #1e3d2f;
    color: #4ddd8e;
    border: 1px solid #2d7a50;
}

.metric-box {
    background: #12121a;
    border: 1px solid #2e2e42;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    margin: 6px 0;
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #888;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #f0f0f0;
}

.model-header {
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    color: #a0a0c0;
    padding: 10px 0 4px 0;
    border-bottom: 1px solid #2e2e42;
    margin-bottom: 14px;
}

.prediction-result {
    background: #12121a;
    border: 1px solid #4f46e5;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    color: #a0c4ff;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("## 🤖 Smart ML App")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works**")
st.sidebar.markdown("""
1. Upload your CSV  
2. Select your single target column  
3. App trains **3 Regression** + **3 Classification** models  
4. Compare metrics and make predictions  
""")

# ================= HELPERS =================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def clean_features(X_raw):
    X = X_raw.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.astype(float)

def run_regression(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    }
    results, trained = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "R2": round(r2_score(y_test, y_pred), 4)
        }
        trained[name] = model
    return results, trained

def run_classification(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    }
    results, trained, reports, cms = {}, {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {"Accuracy": round(accuracy_score(y_test, y_pred), 4)}
        trained[name] = model
        reports[name] = classification_report(y_test, y_pred)
        cms[name] = confusion_matrix(y_test, y_pred)
    return results, trained, reports, cms

# ================= MAIN =================
if uploaded_file:
    df = load_data(uploaded_file)

    st.markdown("# 🤖 Smart ML Training Dashboard")
    st.markdown("Trains **Regression** and **Classification** models simultaneously on your chosen target column.")

    # ---- Dataset Overview ----
    with st.expander("📂 Dataset Overview", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        st.dataframe(df.head(8), use_container_width=True)

    st.markdown("---")

    # ---- Target Selection ----
    target = st.selectbox("🎯 Select Target Column", df.columns)

    # ---- Preprocess features ----
    X_raw = df.drop(columns=[target])
    y_raw = df[target]
    feature_cols = X_raw.columns.tolist()

    X = clean_features(X_raw)

    # ---- Regression target: always numeric ----
    y_reg = pd.to_numeric(y_raw, errors="coerce")
    y_reg = y_reg.fillna(y_reg.median())

    # ---- Classification target: encode labels ----
    le = LabelEncoder()
    y_cls = le.fit_transform(y_raw.astype(str))
    cls_labels = le.classes_

    # ---- Scale and Split ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, yr_train, yr_test = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )
    _, _, yc_train, yc_test = train_test_split(
        X_scaled, y_cls, test_size=0.2, random_state=42
    )

    # ---- Train and Display ----
    st.markdown("## ⚙️ Model Training Results")
    tab_reg, tab_cls = st.tabs(["📈 Regression Models", "🔵 Classification Models"])

    # --- Regression Tab ---
    with tab_reg:
        st.markdown("<div class='task-badge badge-regression'>REGRESSION TASK</div>", unsafe_allow_html=True)
        try:
            reg_results, reg_trained = run_regression(X_train, X_test, yr_train, yr_test)
            for name, metrics in reg_results.items():
                st.markdown(f"<div class='model-header'>▸ {name}</div>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>RMSE</div>
                        <div class='metric-value'>{metrics['RMSE']}</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-label'>R² Score</div>
                        <div class='metric-value'>{metrics['R2']}</div>
                    </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Regression error: {e}")
            reg_trained = {}

    # --- Classification Tab ---
    with tab_cls:
        st.markdown("<div class='task-badge badge-classification'>CLASSIFICATION TASK</div>", unsafe_allow_html=True)
        try:
            cls_results, cls_trained, cls_reports, cls_cms = run_classification(
                X_train, X_test, yc_train, yc_test
            )
            for name, metrics in cls_results.items():
                st.markdown(f"<div class='model-header'>▸ {name}</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-label'>Accuracy</div>
                    <div class='metric-value'>{metrics['Accuracy']}</div>
                </div>""", unsafe_allow_html=True)

                with st.expander(f"📋 Report & Confusion Matrix — {name}"):
                    st.text(cls_reports[name])
                    fig, ax = plt.subplots(figsize=(5, 3))
                    fig.patch.set_facecolor("#1a1a24")
                    ax.set_facecolor("#12121a")
                    sns.heatmap(
                        cls_cms[name], annot=True, fmt="d", cmap="Blues",
                        ax=ax, annot_kws={"color": "white"}
                    )
                    ax.set_title("Confusion Matrix", color="#a0a0c0", fontsize=11)
                    ax.tick_params(colors="#888")
                    st.pyplot(fig)
                    plt.close(fig)
        except Exception as e:
            st.error(f"Classification error: {e}")
            cls_trained = {}

    # ---- Prediction Section ----
    st.markdown("---")
    st.markdown("## 🔮 Make a Prediction")
    st.markdown("Enter values for each feature and predict using all 6 trained models.")

    with st.form("prediction_form"):
        cols = st.columns(3)
        user_input = []
        for i, col in enumerate(feature_cols):
            default_val = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
            val = cols[i % 3].number_input(col, value=default_val)
            user_input.append(val)

        submitted = st.form_submit_button("🚀 Predict with All Models")

    if submitted:
        user_array = scaler.transform([user_input])
        col_r, col_c = st.columns(2)

        with col_r:
            st.markdown("### 📈 Regression Predictions")
            for name, model in reg_trained.items():
                try:
                    pred = model.predict(user_array)[0]
                    st.markdown(f"""
                    <div class='prediction-result'>
                        <b>{name}</b><br>
                        Predicted Value: <span style='color:#60aaff; font-size:18px'>{pred:.4f}</span>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{name}: {e}")

        with col_c:
            st.markdown("### 🔵 Classification Predictions")
            for name, model in cls_trained.items():
                try:
                    pred = model.predict(user_array)[0]
                    label = cls_labels[pred] if pred < len(cls_labels) else str(pred)
                    st.markdown(f"""
                    <div class='prediction-result'>
                        <b>{name}</b><br>
                        Predicted Class: <span style='color:#4ddd8e; font-size:18px'>{label}</span>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"{name}: {e}")

else:
    st.markdown("# 🤖 Smart ML Training App")
    st.info("👈 Upload a CSV dataset from the sidebar to get started.")
    st.markdown("""
    **What this app does:**
    - You select **one target column**
    - Automatically trains **3 Regression models** (Linear, Random Forest, Decision Tree) 
    - Automatically trains **3 Classification models** (Logistic, Random Forest, Decision Tree)
    - Shows full metrics, classification reports, and confusion matrices
    - Lets you enter custom inputs to get predictions from all 6 models at once
    """)