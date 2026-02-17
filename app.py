import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Life Expectancy Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
}
.metric-box {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    text-align: center;
}
.section {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🌍 Life Expectancy App")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📄 Dataset", "📈 Visualizations"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Life Expectancy CSV",
    type=["csv"]
)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

# ------------------ MAIN LOGIC ------------------
if uploaded_file:
    df = load_data(uploaded_file)

    numeric_cols = df.select_dtypes(include=np.number).columns

    # ================= DASHBOARD =================
    if page == "📊 Dashboard":
        st.title("📊 Life Expectancy Dashboard")
        st.caption("A clean and interactive overview of global life expectancy data")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"<div class='metric-box'><h3>📈 Avg Life Expectancy</h3><h2>{df['Life expectancy '].mean():.2f}</h2></div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<div class='metric-box'><h3>🌍 Countries</h3><h2>{df['Country'].nunique()}</h2></div>",
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"<div class='metric-box'><h3>📅 Years Covered</h3><h2>{df['Year'].nunique()}</h2></div>",
                unsafe_allow_html=True
            )

        st.markdown("### 📌 Quick Insights")

        fig, ax = plt.subplots()
        sns.histplot(df['Life expectancy '].dropna(), kde=True, ax=ax)
        ax.set_title("Distribution of Life Expectancy")
        st.pyplot(fig)

    # ================= DATASET =================
    elif page == "📄 Dataset":
        st.title("📄 Dataset Overview")

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Preview")
        st.dataframe(df.head(30))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= VISUALIZATIONS =================
    elif page == "📈 Visualizations":
        st.title("📈 Interactive Visualizations")

        st.markdown("<div class='section'>", unsafe_allow_html=True)

        x_axis = st.selectbox("X-axis", numeric_cols)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis)
        ax.set_title(f"{y_axis} vs {x_axis}")
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.title("🌍 Life Expectancy Analysis App")
    st.info("Upload a CSV file from the sidebar to get started.")
