import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Life Expectancy Analysis App",
    page_icon="🌍",
    layout="wide"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.metric-box {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    text-align: center;
}
.section {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.title("🌍 Life Expectancy App")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📄 Dataset", "📈 Visualizations"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Life Expectancy CSV",
    type=["csv"]
)

# ================== LOAD DATA ==================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # CLEAN COLUMN NAMES (VERY IMPORTANT)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

# ================== MAIN APP ==================
if uploaded_file:
    df = load_data(uploaded_file)

    # Detect numeric columns safely
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ================= DASHBOARD =================
    if page == "📊 Dashboard":
        st.title("📊 Life Expectancy Dashboard")
        st.caption("Interactive overview of life expectancy data")

        # SAFETY CHECK
        if "life_expectancy" not in df.columns:
            st.error("❌ Column 'Life Expectancy' not found in dataset.")
            st.write("Available columns:", df.columns.tolist())
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                    <div class='metric-box'>
                        <h3>📈 Avg Life Expectancy</h3>
                        <h2>{df['life_expectancy'].mean():.2f}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                if "country" in df.columns:
                    countries = df["country"].nunique()
                else:
                    countries = "N/A"

                st.markdown(
                    f"""
                    <div class='metric-box'>
                        <h3>🌍 Countries</h3>
                        <h2>{countries}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                if "year" in df.columns:
                    years = df["year"].nunique()
                else:
                    years = "N/A"

                st.markdown(
                    f"""
                    <div class='metric-box'>
                        <h3>📅 Years Covered</h3>
                        <h2>{years}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### 📌 Life Expectancy Distribution")

            fig, ax = plt.subplots()
            sns.histplot(df["life_expectancy"].dropna(), kde=True, ax=ax)
            ax.set_xlabel("Life Expectancy")
            ax.set_ylabel("Frequency")
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

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())
        st.markdown("</div>", unsafe_allow_html=True)

    # ================= VISUALIZATIONS =================
    elif page == "📈 Visualizations":
        st.title("📈 Interactive Visualizations")

        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns for visualization.")
        else:
            st.markdown("<div class='section'>", unsafe_allow_html=True)

            x_axis = st.selectbox("X-axis", numeric_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1)

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis)
            ax.set_title(f"{y_axis} vs {x_axis}")
            st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)

# ================== EMPTY STATE ==================
else:
    st.title("🌍 Life Expectancy Analysis App")
    st.info("👈 Upload a CSV file from the sidebar to get started.")
