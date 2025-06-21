import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import re
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor # Keep this if you plan to use it later, otherwise remove
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pycountry

# --- Session State Initialization ---
if 'selected_region_display' not in st.session_state:
    st.session_state.selected_region_display = "Global & Region-Specific"
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = False
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = pd.DataFrame()
if 'r2_score' not in st.session_state:
    st.session_state.r2_score = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_features' not in st.session_state:
    st.session_state.model_features = []
if 'scaler' not in st.session_state:
    st.session_state.scaler = None


# --- Set page configuration ---
st.set_page_config(
    layout="wide",
    page_title="Life Expectancy Analysis Dashboard",
    page_icon=" 🌿 "
)

# --- Custom CSS for dark elegant theme ---
st.markdown("""
<style>
    /* Base styling */
    * {
        box-sizing: border-box;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    body {
        background-color: #0A0E17;
        color: #E0E0E0;
    }
    .stApp {
        background: linear-gradient(135deg, #0A0E17 0%, #141A29 100%);
        color: #E0E0E0;
        width: 100%; 
        max-width: 100%;
        overflow-x: hidden; /* Prevent horizontal scroll on main app */
    }
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    /* Introduction box */
    .intro-box {
        background: rgba(26, 32, 48, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(64, 128, 255, 0.2);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 20, 60, 0.3);
    }
    /* Metrics styling */
    div[data-testid="stMetric"] {
        background: rgba(20, 25, 38, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin: 8px;
        border: 1px solid rgba(64, 128, 255, 0.15);
        box-shadow: 0 6px 20px rgba(0, 15, 40, 0.25);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 20, 60, 0.4);
    }
    div[data-testid="stMetric"] > div {
        background: transparent !important;
    }
    div[data-testid="stMetric"] label {
        color: #9DB5CE !important;
        font-size: 0.95rem;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2.4rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(100, 180, 255, 0.3);
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #9DB5CE !important;
        font-size: 1.1rem;
    }
    /* Tabs styling */
    .stTabs [data-testid="stTab"] {
        background: rgba(20, 25, 38, 0.6);
        color: #9DB5CE;
        border-radius: 8px 8px 0 0;
        margin-right: 6px;
        padding: 14px 28px;
        font-weight: 500;
        transition: all 0.3s ease;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-testid="stTab"]:hover {
        background: rgba(30, 40, 60, 0.8);
        color: #FFFFFF;
    }
    .stTabs [data-testid="stTab"][aria-selected="true"] {
        background: rgba(30, 40, 60, 0.95);
        color: #FFFFFF;
        border-bottom: 3px solid #4A8BFF;
    }
    .stTabs [data-testid="stVerticalBlock"] {
        background: rgba(20, 25, 38, 0.7);
        border-radius: 0 0 12px 12px;
        padding: 30px;
        border: 1px solid rgba(64, 128, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 15, 40, 0.25);
    }
    /* Sidebar styling */
    .st-emotion-cache-vk33wx {
        background: rgba(15, 20, 30, 0.95);
        border-right: 1px solid rgba(64, 128, 255, 0.1);
        box-shadow: 4px 0 20px rgba(0, 10, 30, 0.4);
    }
    /* Expander styling */
    .st-emotion-cache-1evx060 {
        background: rgba(26, 32, 48, 0.6);
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid rgba(64, 128, 255, 0.15);
    }
    .st-emotion-cache-1evx060 button {
        color: #FFFFFF;
        font-weight: 500;
        padding: 16px;
    }
    .st-emotion-cache-1evx060 div[data-testid="stExpanderDetails"] {
        background: rgba(20, 25, 38, 0.6);
        border-top: 1px solid rgba(64, 128, 255, 0.1);
        padding: 20px;
    }
    /* Input widgets */
    div[data-testid="stMultiSelect"] > div > div:first-child,
    div[data-testid="stSelectbox"] > div > div:first-child {
        background: rgba(20, 25, 38, 0.8) !important;
        border: 1px solid rgba(64, 128, 255, 0.2);
        border-radius: 10px;
        padding: 10px 15px; /* Adjusted padding */
        min-height: 3.5rem; /* Added min-height for consistent box size */
        display: flex; /* Ensure content is centered vertically */
        align-items: center; /* Center content vertically */
    }
    /* Set font color for text inside multiselect/selectbox to orange */
    div[data-testid="stMultiSelect"] .st-emotion-cache-1gx59c3, /* Selected items in multiselect */
    div[data-testid="stMultiSelect"] input[type="text"], /* Input field for typing in multiselect */
    div[data-testid="stSelectbox"] .st-emotion-cache-1gx59c3, /* Selected item in selectbox */
    div[data-testid="stSelectbox"] input[type="text"] { /* Input field in selectbox */
        color: #FD7E14 !important; /* Force orange color for the text */
    }
    /* Specific targeting for the dropdown caret/arrow */
    div[data-testid="stSelectbox"] > div > div:first-child > div[data-testid="stMarkdownContainer"] {
        padding-right: 2rem; /* Add space for the caret */
    }
    div[data-testid="stSelectbox"] .st-emotion-cache-vdhr9d svg { /* Targets the SVG of the caret icon */
        fill: #FD7E14 !important; /* Set caret color to orange */
    }
    
    /* Options in dropdowns */
    .st-emotion-cache-1xarl3l { /* Dropdown menu background */
        background: rgba(26, 32, 48, 0.95) !important;
        border: 1px solid rgba(64, 128, 255, 0.3);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    .st-emotion-cache-1xarl3l div[role="option"] {
        color: #FFFFFF;
        padding: 8px 12px;
    }
    .st-emotion-cache-1xarl3l div[role="option"]:hover,
    .st-emotion-cache-1xarl3l div[aria-selected="true"] {
        background: #4A8BFF; /* Highlight on hover/selected */
        color: white;
    }
    /* Slider specific styling */
    div[data-testid="stSlider"] .st-emotion-cache-1ux495f {
        background: rgba(20, 25, 38, 0.8); /* Dark background for slider track area */
        border: 1px solid rgba(64, 128, 255, 0.2);
        border-radius: 8px;
        color: #FFFFFF;
        padding: 5px 10px; /* Adjust padding for slider value display */
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    .stSlider .st-emotion-cache-1f81tsl { /* Slider track itself */
        background: #5A7CA3;
    }
    .stSlider .st-emotion-cache-1f81tsl > div { /* Slider fill */
        background: #4A8BFF;
    }
    .stSlider .st-emotion-cache-17l1x26 { /* Slider thumb */
        background: #4A8BFF;
        border: 2px solid #FFFFFF;
    }

    /* Info, Success, Warning Alert Styling */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 15px;
        padding: 15px 20px;
        color: #E0E0E0; /* Default text for alerts */
        border: none;
    }
    div[data-testid="stAlert"] .st-emotion-cache-v06ywu { /* Success */
        background-color: rgba(46, 204, 113, 0.2) !important;
        border-left: 5px solid #2ECC71 !important;
        color: #E0E0E0 !important;
    }
    div[data-testid="stAlert"] .st-emotion-cache-1jmveo { /* Info (general alert, used for info/warning unless specific target is hit) */
        background-color: rgba(74, 139, 255, 0.15) !important;
        border-left: 5px solid #4A8BFF !important;
        color: #E0E0E0 !important;
    }
    div[data-testid="stAlert"] .st-emotion-cache-1y5v8a8 { /* Info specific */
        background-color: rgba(74, 139, 255, 0.15) !important;
        border-left: 5px solid #4A8BFF !important;
        color: #E0E0E0 !important;
    }
    div[data-testid="stAlert"] .st-emotion-cache-1f1i1k7 { /* Warning */
        background-color: rgba(255, 126, 95, 0.15) !important;
        border-left: 5px solid #FF7E5F !important;
        color: #E0E0E0 !important;
    }
    div[data-testid="stAlert"] .st-emotion-cache-gsvt5j p { /* Error */
        color: #E0E0E0 !important;
    }

    /* Plot containers */
    .stPlotlyChart, div[data-testid="stFigure"], .folium-map {
        background: rgba(20, 25, 38, 0.7);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(64, 128, 255, 0.15);
        box-shadow: 0 8px 24px rgba(0, 15, 40, 0.2);
    }
    /* Dataframes */
    .dataframe {
        background: rgba(20, 25, 38, 0.7);
        border-radius: 10px;
        border: 1px solid rgba(64, 128, 255, 0.1);
        overflow: hidden; /* Ensure rounded corners are visible */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .dataframe th {
        background: rgba(30, 40, 60, 0.8);
        color: #FFFFFF;
        font-weight: 600;
        padding: 12px 15px;
        border-bottom: 1px solid #444;
    }
    .dataframe td {
        color: #E0E0E0;
        padding: 10px 15px;
        border-bottom: 1px solid rgba(25, 35, 55, 0.5); /* Subtle row separator */
    }
    .dataframe tr:nth-child(even) {
        background: rgba(25, 35, 55, 0.5);
    }
    .dataframe tr:hover {
        background: rgba(40, 60, 100, 0.4);
    }
    /* Custom decorations */
    .glow-text {
        text-shadow: 0 0 12px rgba(100, 180, 255, 0.7);
    }
    
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(74, 139, 255, 0.5), transparent);
        margin: 30px 0;
        border: none;
    }
    
    .feature-card {
        background: rgba(20, 25, 38, 0.6);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(64, 128, 255, 0.15);
        box-shadow: 0 6px 18px rgba(0, 15, 40, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 20, 60, 0.3);
        border-color: rgba(74, 139, 255, 0.3);
    }
    /* CSS for parallel coordinates */
    .parcoords > svg > g > g.tick > text {
        fill: #E0E0E0 !important; /* White text for ticks */
        font-size: 12px !important;
    }
    .parcoords > svg > g > g.tick > line {
        stroke: #4A8BFF !important; /* Blue lines for ticks */
    }
</style>
""", unsafe_allow_html=True)

# --- Introduction ---
st.title(" 🌿  The Longevity Puzzle: Unlocking the Secrets to a Longer Life")
st.markdown(f"""
<div class="intro-box">
    <h3 class="glow-text">Uncover What Shapes a Longer Life</h3>
    <p style="font-size: 1.1rem; line-height: 1.7; color: #CCD6E0;">
        This dashboard explores the socioeconomic and health factors influencing life expectancy across diverse regions. 
        Analyze trends, correlations, and predictive models to uncover insights that could drive meaningful change.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            return None
        
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'[^a-z0-9_]+', '_', regex=True) 
            .str.strip('_') 
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

uploaded_file = st.sidebar.file_uploader(" 📤  Upload Life Expectancy Data (CSV/XLSX)", type=["csv", "xlsx"])
df = None
if uploaded_file is not None:
    with st.spinner(' 🔍  Analyzing data...'):
        df = load_data(uploaded_file)
        if df is not None:
            REQUIRED_COLUMNS = ['life_expectancy', 'country_name', 'year']
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                st.error(f"Missing required columns for analysis: {', '.join(missing)}. Please upload a dataset containing these columns.")
                st.stop() 
            st.success(" ✅  Data loaded successfully!")
            
            # Display metrics
            cols = st.columns(4)
            with cols[0]:
                if 'life_expectancy' in df.columns:
                    st.metric("Avg Life Expectancy", f"{df['life_expectancy'].mean():.1f} yrs")
            with cols[1]:
                if 'gdp_per_capita' in df.columns:
                    st.metric("Avg GDP per Capita", f"${df['gdp_per_capita'].mean():,.0f}")
            with cols[2]:
                if 'health_expenditure_per_capita' in df.columns:
                    st.metric("Healthcare Expenditure", f"${df['health_expenditure_per_capita'].mean():,.0f}")
            with cols[3]:
                if 'school_enrollment_combined' in df.columns:
                    st.metric("School Enrollment", f"{df['school_enrollment_combined'].mean():.1f}%")
            
            st.session_state['initial_load'] = True
        else:
            st.info("Uploaded file could not be processed. Please check the file format or its content.")
else:
    if 'initial_load' not in st.session_state or not st.session_state.initial_load:
        st.info(" ℹ️  Please upload a dataset to begin analysis.")

filtered_df = pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame()
selected_countries = []
if df is not None:
    # --- Data Processing ---
    if 'life_expectancy' in df.columns:
        df['life_expectancy'] = pd.to_numeric(df['life_expectancy'], errors='coerce')
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df.dropna(subset=['country_name'], inplace=True)

    # --- Sidebar Filters ---
    st.sidebar.header(" 🔍  Data Filters")
    years = sorted(df['year'].unique())
    selected_years = []
    if len(years) > 0:
        selected_years = st.sidebar.multiselect(
            "Select Year(s)",
            years,
            default=[]
        )
    else:
        st.sidebar.warning("No year data available")
        
    temp_df = df.copy()
    if selected_years:
        temp_df = temp_df[temp_df['year'].isin(selected_years)]
    elif 'year' in df.columns and len(years) > 0:
        temp_df = pd.DataFrame(columns=df.columns)
        st.sidebar.info("Select at least one year to filter data.")

    # Region filter
    if 'region' in df.columns:
        if not temp_df.empty and 'region' in temp_df.columns:
            regions = sorted(temp_df['region'].unique().tolist())
            selected_region = st.sidebar.selectbox("Select Region", ['All'] + regions)
            st.session_state.selected_region_display = selected_region
            if selected_region != 'All':
                temp_df = temp_df[temp_df['region'] == selected_region]
        else:
            selected_region = st.sidebar.selectbox("Select Region", ['All'])
            st.session_state.selected_region_display = "Global (No regional data for current selection)"
            st.sidebar.info("No regional data available for the current year(s) selection.")
    else:
        st.sidebar.info("Region column not found for region filtering.")
        st.session_state.selected_region_display = "Global (Region column missing)"

    # Country filter
    all_countries = []
    if not temp_df.empty and 'country_name' in temp_df.columns:
        all_countries = sorted(temp_df['country_name'].unique().tolist())
    
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        all_countries,
        default=[]
    )
    if selected_countries and not temp_df.empty and 'country_name' in temp_df.columns:
        filtered_df = temp_df[temp_df['country_name'].isin(selected_countries)].copy()
    else:
        filtered_df = pd.DataFrame(columns=df.columns)

    if 'life_expectancy' not in filtered_df.columns and not filtered_df.empty:
        st.error("The 'life_expectancy' column is missing in your filtered dataset. Please ensure it is present and numeric.")
        filtered_df = pd.DataFrame(columns=df.columns)
    num_cols_filtered = filtered_df.select_dtypes(include=['number']).columns if not filtered_df.empty else pd.Index([])

    # --- Conditional Display of Tabs ---
    if not filtered_df.empty:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            " 📊  Overview",
            " 📈  Relationships",
            " 🗺️  Geographic",
            " 🔍  Deep Analysis",
            " 💡  Recommendations",
            " 🧠  Advanced Analytics"
        ])

        with tab1:
            st.header("Data Overview")
            cols = st.columns(4)
            with cols[0]:
                if 'life_expectancy' in num_cols_filtered:
                    st.metric("Avg Life Expectancy", f"{filtered_df['life_expectancy'].mean():.1f} yrs")
            with cols[1]:
                if 'gdp_per_capita' in num_cols_filtered:
                    st.metric("Avg GDP per Capita", f"${filtered_df['gdp_per_capita'].mean():,.0f}")
            with cols[2]:
                if 'health_expenditure_per_capita' in num_cols_filtered:
                    st.metric("Healthcare Expenditure", f"${filtered_df['health_expenditure_per_capita'].mean():,.0f}")
            with cols[3]:
                if 'school_enrollment_combined' in num_cols_filtered:
                    st.metric("School Enrollment", f"{filtered_df['school_enrollment_combined'].mean():.1f}%")
            
            with st.expander(" 📋  Summary Statistics"):
                st.write(filtered_df.describe())

            if 'year' in filtered_df.columns and 'life_expectancy' in filtered_df.columns:
                st.subheader("Life Expectancy Trend Analysis")
                if len(filtered_df['country_name'].unique()) == 1:
                    country = filtered_df['country_name'].iloc[0]
                    country_time_series_df = df[df['country_name'] == country].set_index('year')['life_expectancy'].sort_index()
                    if len(country_time_series_df) > 2:
                        plt.style.use('seaborn-v0_8-darkgrid')
                        decomposition = seasonal_decompose(country_time_series_df, model='additive', period=1)
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
                        fig.patch.set_facecolor('#0A0E17')
                        ax1.set_facecolor('#141A29')
                        ax2.set_facecolor('#141A29')
                        ax3.set_facecolor('#141A29')
                        ax1.plot(decomposition.trend, color='#4A8BFF', linewidth=2)
                        ax1.set_title('Trend Component', fontsize=16, color='#FFFFFF')
                        ax1.set_ylabel('Life Expectancy', fontsize=12, color='#E0E0E0')
                        ax1.tick_params(axis='x', colors='#9DB5CE')
                        ax1.tick_params(axis='y', colors='#9DB5CE')
                        ax1.grid(True, linestyle='--', alpha=0.3, color='#5A7CA3')
                        ax2.plot(decomposition.seasonal, color='#2ECC71', linewidth=1.5)
                        ax2.set_title('Seasonal Component', fontsize=16, color='#FFFFFF')
                        ax2.set_ylabel('Seasonality', fontsize=12, color='#E0E0E0')
                        ax2.tick_params(axis='x', colors='#9DB5CE')
                        ax2.tick_params(axis='y', colors='#9DB5CE')
                        ax2.grid(True, linestyle='--', alpha=0.3, color='#5A7CA3')
                        ax3.plot(decomposition.resid, color='#FF7E5F', linewidth=1)
                        ax3.set_title('Residuals Component', fontsize=16, color='#FFFFFF')
                        ax3.set_xlabel('Year', fontsize=12, color='#E0E0E0')
                        ax3.set_ylabel('Residual', fontsize=12, color='#E0E0E0')
                        ax3.tick_params(axis='x', colors='#9DB5CE')
                        ax3.tick_params(axis='y', colors='#9DB5CE')
                        ax3.grid(True, linestyle='--', alpha=0.3, color='#5A7CA3')
                        fig.suptitle(f'Time Series Decomposition for {country.title()} Life Expectancy', 
                                     fontsize=20, color='#FFFFFF')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Insufficient data for time series decomposition (requires at least 3 data points for one country across years).")
                else:
                    st.info("Select one country from the sidebar to view time series decomposition across all years.")

        with tab2:
            st.subheader("Advanced Correlation Analysis")
            default_corr_cols = [
                'life_expectancy', 'gdp_per_capita', 'health_expenditure_per_capita',
                'school_enrollment_combined', 'mortality_infant', 'access_to_electricity', 'population_growth_rate'
            ]
            safe_default_corr_cols = [col for col in num_cols_filtered if col in default_corr_cols]
            corr_cols = st.multiselect("Select variables for correlation", num_cols_filtered.tolist(), default=safe_default_corr_cols)

            if corr_cols and len(corr_cols) > 1 and not filtered_df[corr_cols].empty:
                corr_df = filtered_df[corr_cols].corr()
                try:
                    from scipy.cluster import hierarchy
                    dist = hierarchy.distance.pdist(corr_df)
                    linkage = hierarchy.linkage(dist, method='average')
                    order = hierarchy.leaves_list(linkage)
                    clustered_corr = corr_df.iloc[order, order]
                except Exception as e:
                    clustered_corr = corr_df
                    st.warning(f"Could not perform hierarchical clustering for correlation matrix: {e}. Displaying unclustered matrix.")
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=clustered_corr.values,
                    x=clustered_corr.columns,
                    y=clustered_corr.index,
                    colorscale='Viridis',
                    zmin=-1,
                    zmax=1,
                    text=np.round(clustered_corr.values, 2),
                    texttemplate="%{text}",
                    hoverinfo="x+y+z"
                ))
                fig_corr.update_layout(
                    title="Clustered Correlation Matrix",
                    height=700,
                    xaxis_title="Features",
                    yaxis_title="Features",
                    template="plotly_dark", 
                    paper_bgcolor='#141A29',
                    plot_bgcolor='#1A2235',
                    font=dict(color='#E0E0E0')
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                corr_unstacked = corr_df.unstack().reset_index()
                corr_unstacked.columns = ['Var1', 'Var2', 'Correlation']
                corr_unstacked = corr_unstacked[corr_unstacked['Var1'] != corr_unstacked['Var2']]
                top_corrs = corr_unstacked.sort_values('Correlation', ascending=False).head(5)
                bottom_corrs = corr_unstacked.sort_values('Correlation', ascending=True).head(5)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Top Positive Correlations")
                    for _, row in top_corrs.iterrows():
                        st.markdown(f"<div class='feature-card'><b>{row['Var1'].replace('_', ' ').title()}</b> & <b>{row['Var2'].replace('_', ' ').title()}</b><br>Correlation: <b style='color:#4A8BFF'>{row['Correlation']:.2f}</b></div>", 
                                    unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### Top Negative Correlations")
                    for _, row in bottom_corrs.iterrows():
                        st.markdown(f"<div class='feature-card'><b>{row['Var1'].replace('_', ' ').title()}</b> & <b>{row['Var2'].replace('_', ' ').title()}</b><br>Correlation: <b style='color:#FF7E5F'>{row['Correlation']:.2f}</b></div>", 
                                    unsafe_allow_html=True)
            else:
                st.info("Please select at least two numerical variables to view the correlation matrix.")

        with tab3:
            st.header("Geographic Analysis")
            with st.expander(" 🔍  Methodology Details"):
                st.markdown(f"""
                **Data Source:** User-uploaded dataset  
                **Time Period:** {df['year'].min()} - {df['year'].max()}  
                **Countries in Sample:** {df['country_name'].nunique()}  
                **Pre-processing:** - Missing values imputed with median  
                - Country names standardized  
                - Numerical columns scaled  
                """)
            
            st.info("**Mapping Note:** Countries are identified using ISO 3-letter country codes derived from their names.")
            
            if 'year' in df.columns and 'country_name' in df.columns and not filtered_df.empty:
                st.subheader("Interactive World Map")
                
                def get_iso_alpha3(country_name):
                    try:
                        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
                    except:
                        return None
                        
                filtered_df['Name'] = filtered_df['country_name'].apply(get_iso_alpha3)
                map_df = filtered_df.groupby('country_name').agg({
                    'life_expectancy': 'mean',
                    'gdp_per_capita': 'mean',
                    'Name': 'first'
                }).reset_index().dropna(subset=['Name'])
                
                if not map_df.empty:
                    fig_map = px.choropleth(
                        map_df,
                        locations="Name",
                        color='life_expectancy',
                        hover_name="country_name",
                        hover_data=['gdp_per_capita'],
                        color_continuous_scale='Plasma',
                        title="Global Life Expectancy Distribution",
                        height=700
                    )
                    fig_map.update_layout(
                        height=700,
                        template="plotly_dark",
                        paper_bgcolor='#141A29',
                        plot_bgcolor='#1A2235',
                        font=dict(color='#E0E0E0'),
                        geo=dict(
                            bgcolor='rgba(0,0,0,0)',
                            lakecolor='#1F4E79',
                            landcolor='#1A2235',
                            showocean=True,
                            oceancolor='#0A0E17'
                        )
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Could not generate map - missing ISO country codes for selected countries.")
            else:
                st.info("Geographic analysis requires 'year' and 'country_name' columns in the dataset.")

        with tab4:
            st.subheader("Feature Impact Analysis")
            
            if 'life_expectancy' in num_cols_filtered:
                feature = st.selectbox("Select Feature to Analyze", 
                                    [col for col in num_cols_filtered if col != 'life_expectancy'])
                
                if feature:
                    correlation = filtered_df['life_expectancy'].corr(filtered_df[feature])
                    
                    fig = px.scatter(
                        filtered_df,
                        x=feature,
                        y='life_expectancy',
                        color='country_name' if selected_countries else 'region' if 'region' in filtered_df else None,
                        trendline='ols',
                        title=f"Life Expectancy vs {feature.replace('_', ' ').title()} (Correlation: {correlation:.2f})",
                        height=600,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    
                    global_avg = df.groupby('year')['life_expectancy'].mean().reset_index()
                    if not global_avg.empty:
                        fig.add_hline(y=global_avg['life_expectancy'].mean(),
                                      line_dash="dot",
                                      annotation_text="Global Average",
                                      annotation_position="bottom right",
                                      line_color='#9DB5CE')
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='#141A29',
                        plot_bgcolor='#1A2235',
                        font=dict(color='#E0E0E0'),
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if filtered_df[feature].nunique() > 1:
                        try:
                            filtered_df['feature_bin'] = pd.qcut(filtered_df[feature], 5, duplicates='drop')
                            fig_box = px.box(
                                filtered_df,
                                x='feature_bin',
                                y='life_expectancy',
                                title=f"Life Expectancy Distribution by {feature.replace('_', ' ').title()} Group",
                                height=600,
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            fig_box.update_layout(
                                template="plotly_dark",
                                paper_bgcolor='#141A29',
                                plot_bgcolor='#1A2235',
                                font=dict(color='#E0E0E0')
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create bins for '{feature}': {e}")

            st.subheader("Predictive Modeling: Factors Influencing Life Expectancy")
            st.markdown("Build a model to understand feature importance.")
            target = 'life_expectancy'
            
            potential_features = [col for col in num_cols_filtered if col != target and col not in ['year']]
            
            if not potential_features:
                st.warning("No suitable numerical features found for modeling.")
            elif filtered_df.empty:
                st.info("Filtered dataset is empty.")
            else:
                selected_model_features = st.multiselect(
                    "Select features for predictive model",
                    potential_features,
                    default=[pf for pf in ['gdp_per_capita', 'health_expenditure_per_capita', 'school_enrollment_combined', 'mortality_infant', 'access_to_electricity', 'population_growth_rate'] if pf in potential_features]
                )
                if selected_model_features:
                    clean_df = filtered_df[[target] + selected_model_features].dropna()
                    
                    min_total_samples_for_model = 5 
                    
                    if len(clean_df) < min_total_samples_for_model:
                        st.warning(f"Insufficient data rows ({len(clean_df)} samples) for modeling. Minimum required: {min_total_samples_for_model}.")
                    else: 
                        X_clean = clean_df[selected_model_features]
                        y_clean = clean_df[target]
                        if len(np.unique(y_clean)) <= 1:
                            st.warning("Insufficient variation in target values.")
                        else:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_clean)
                            
                            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, random_state=42)

                            n_samples_train = len(X_train)
                            min_splits_cv = 5
                            # Adjusted n_splits_adjusted to ensure it's less than n_samples_train or 2 if n_samples_train is tiny.
                            n_splits_adjusted = max(2, min(n_samples_train - 1, min_splits_cv)) if n_samples_train > 1 else 1 
                            # If n_samples_train is 1, n_splits_adjusted becomes 1, so Ridge will be used.
                            
                            if n_splits_adjusted < 2:
                                # Fallback to simple Ridge if not enough splits for CV
                                model = Ridge(alpha=1.0)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                st.write(f"**Test Set R-squared:** {r2:.2f}")
                                st.write(f"**Mean Squared Error:** {mse:.2f}")
                                
                                st.session_state['feature_importance'] = pd.DataFrame({
                                    'Feature': selected_model_features,
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False)
                            else:
                                alphas = [0.001, 0.01, 0.1, 1, 10, 100]
                                model = RidgeCV(alphas=alphas, cv=n_splits_adjusted) 
                                model.fit(X_train, y_train)
                                
                                # Removed the separate cross_val_score call.
                                # Rely on model.best_score_ for internal CV score.
                                st.write(f"**Best Alpha (Regularization):** {model.alpha_:.4f}")
                                st.write(f"**Internal CV R-squared:** {model.best_score_:.2f}") 
                                
                                y_pred = model.predict(X_test)
                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                
                                st.session_state['r2_score'] = r2
                                st.session_state['feature_importance'] = pd.DataFrame({
                                    'Feature': selected_model_features,
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False)
                                
                                st.write(f"**Test Set R-squared:** {r2:.2f}")
                                st.write(f"**Mean Squared Error:** {mse:.2f}")
                            
                            # Store the trained model and scaler in session state
                            st.session_state['trained_model'] = model
                            st.session_state['scaler'] = scaler
                            st.session_state['model_features'] = selected_model_features

                            st.subheader("Feature Importance (Coefficients)")
                            st.dataframe(st.session_state['feature_importance'])
                            fig_feature_imp = px.bar(st.session_state['feature_importance'], 
                                                    x='Coefficient', y='Feature', orientation='h', 
                                                    title='Feature Impact on Life Expectancy Prediction',
                                                    height=600,
                                                    color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig_feature_imp.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                template="plotly_dark",
                                paper_bgcolor='#141A29',
                                plot_bgcolor='#1A2235',
                                font=dict(color='#E0E0E0')
                            )
                            st.plotly_chart(fig_feature_imp, use_container_width=True)
                else:
                    st.info("Select features to build a predictive model.")
            
            st.subheader("Multi-dimensional Data Relationships")
            st.markdown("Visualize patterns across multiple features.")
            parallel_coord_cols_options = [col for col in num_cols_filtered if col != 'year']
            categorical_color_options = []
            if 'country_name' in filtered_df.columns and filtered_df['country_name'].nunique() > 1 and len(filtered_df['country_name'].unique()) <= 50:
                categorical_color_options.append('country_name')
            if 'region' in filtered_df.columns and filtered_df['region'].nunique() > 1 and len(filtered_df['region'].unique()) <= 20:
                categorical_color_options.append('region')
            all_color_options = ['None'] + sorted(parallel_coord_cols_options) + sorted(categorical_color_options)
            
            default_pc_dimensions = [
                'life_expectancy', 'gdp_per_capita',
                'health_expenditure_per_capita', 'school_enrollment_combined'
            ]
            safe_default_pc_dimensions = [col for col in default_pc_dimensions if col in parallel_coord_cols_options]
            
            selected_parallel_cols = st.multiselect(
                "Select variables for Parallel Coordinates",
                parallel_coord_cols_options,
                default=safe_default_pc_dimensions
            )
            color_choice = st.selectbox(
                "Color lines by",
                all_color_options,
                index=all_color_options.index('life_expectancy') if 'life_expectancy' in all_color_options else (all_color_options.index('None') if 'None' in all_color_options else 0)
            )
            
            if selected_parallel_cols and len(selected_parallel_cols) > 1:
                cols_for_plot = list(set(selected_parallel_cols + ([color_choice] if color_choice != 'None' else [])))
                pc_df = filtered_df[cols_for_plot].dropna().copy()
                
                if not pc_df.empty:
                    dimensions = []
                    for col in selected_parallel_cols:
                        if col in pc_df.columns:
                            col_min = pc_df[col].min()
                            col_max = pc_df[col].max()
                            range_buffer = (col_max - col_min) * 0.1
                            
                            dim_config = {
                                "label": col.replace('_', ' ').title(),
                                "values": pc_df[col],
                                "range": [col_min - range_buffer, col_max + range_buffer]
                            }
                            
                            if col == 'life_expectancy':
                                dim_config["range"] = [40, 90]
                            
                            dimensions.append(dim_config)
                    
                    if len(dimensions) >= 2:
                        fig_par_coords = go.Figure(data=go.Parcoords(
                            line=dict(
                                color=pc_df[color_choice] if color_choice != 'None' and color_choice in pc_df.columns else '#4A8BFF',
                                colorscale='Viridis',
                                showscale=True if color_choice != 'None' and color_choice in pc_df.columns else False,
                                cmin=pc_df[color_choice].min() if color_choice != 'None' else None,
                                cmax=pc_df[color_choice].max() if color_choice != 'None' else None
                            ),
                            dimensions=dimensions
                        ))
                        
                        fig_par_coords.update_layout(
                            title="Parallel Coordinates Analysis",
                            height=700,
                            template="plotly_dark",
                            paper_bgcolor='#141A29',
                            plot_bgcolor='#1A2235',
                            font=dict(color='#E0E0E0'),
                            margin=dict(l=80, r=80, t=80, b=80)
                        )
                        st.plotly_chart(fig_par_coords, use_container_width=True)
                    else:
                        st.info("Not enough dimensions to plot")
                else:
                    st.info("No complete data for selected variables")
            else:
                st.info("Select at least two numerical variables")

        with tab5:
            st.header("Data-Driven Recommendations")
            
            if not filtered_df.empty and 'life_expectancy' in filtered_df.columns:
                st.subheader("Performance Benchmarks")
                
                if 'year' in filtered_df.columns and filtered_df['year'].nunique() > 1:
                    le_diff = filtered_df.groupby('country_name')['life_expectancy'].agg(
                        le_start=('min'),
                        le_end=('max'),
                        le_growth=lambda x: x.max() - x.min()
                    ).reset_index()
                    
                    top_growth = le_diff.sort_values('le_growth', ascending=False).head(3)
                    bottom_growth = le_diff.sort_values('le_growth', ascending=True).head(3)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Top Performers")
                        for _, row in top_growth.iterrows():
                            st.markdown(f"<div class='feature-card'>"
                                        f"<b>{row['country_name'].title()}</b><br>"
                                        f"<span style='color:#4A8BFF'>+{row['le_growth']:.1f} years</span> "
                                        f"({row['le_start']:.1f} → {row['le_end']:.1f})"
                                        f"</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### Countries Needing Improvement")
                        for _, row in bottom_growth.iterrows():
                            st.markdown(f"<div class='feature-card'>"
                                        f"<b>{row['country_name'].title()}</b><br>"
                                        f"<span style='color:#FF7E5F'>{row['le_growth']:.1f} years</span> "
                                        f"({row['le_start']:.1f} → {row['le_end']:.1f})"
                                        f"</div>", unsafe_allow_html=True)
                else:
                    st.info("Select multiple years to calculate life expectancy growth.")
                
                if len(selected_countries) > 1:
                    st.subheader("Country Comparison Analysis")
                    comparison_df = filtered_df.groupby('country_name').agg({
                        'life_expectancy': 'mean',
                        'gdp_per_capita': 'mean',
                        'health_expenditure_per_capita': 'mean',
                        'school_enrollment_combined': 'mean'
                    }).reset_index()
                    
                    base_country = st.selectbox("Select Base Country", comparison_df['country_name'].tolist())
                    
                    if base_country:
                        base_values = comparison_df[comparison_df['country_name'] == base_country].iloc[0]
                        
                        gap_analysis = []
                        for _, row in comparison_df.iterrows():
                            if row['country_name'] != base_country:
                                gap = {
                                    'country': row['country_name'].title(),
                                    'life_expectancy_gap': row['life_expectancy'] - base_values['life_expectancy'],
                                    'gdp_gap_percent': (row['gdp_per_capita'] - base_values['gdp_per_capita']) / base_values['gdp_per_capita'] * 100 if base_values['gdp_per_capita'] != 0 else np.nan,
                                    'health_exp_gap_percent': (row['health_expenditure_per_capita'] - base_values['health_expenditure_per_capita']) / base_values['health_expenditure_per_capita'] * 100 if base_values['health_expenditure_per_capita'] != 0 else np.nan,
                                    'education_gap': row['school_enrollment_combined'] - base_values['school_enrollment_combined']
                                }
                                gap_analysis.append(gap)
                        
                        gap_df = pd.DataFrame(gap_analysis)
                        
                        st.write("**Performance Gap Compared to Baseline:**")
                        st.dataframe(gap_df.style.background_gradient(cmap='RdYlGn', 
                                                                      subset=['life_expectancy_gap', 'gdp_gap_percent', 'health_exp_gap_percent', 'education_gap'], 
                                                                      axis=0))
                    else:
                        st.info("Select a base country to perform gap analysis.")
                else:
                    st.info("Select more than one country to perform comparison analysis.")
            else:
                st.info("Please ensure data is loaded and filtered to see recommendations.")
        
        with tab6: # New tab for Advanced Analytics
            st.header(" 🧠  Advanced Analytics: Decision Support")
            st.markdown("""
            This section provides advanced analytical tools for deeper insights and policy decision support.
            Explore predictive scenarios, conduct what-if analyses, and simulate policy impacts.
            """)

            # Ensure model and scaler are available
            model = st.session_state.trained_model
            scaler = st.session_state.scaler
            features = st.session_state.model_features

            if model is None or scaler is None or not features:
                st.warning("Please train a predictive model in the 'Deep Analysis' tab first to enable Advanced Analytics.")
            else:
                st.markdown("---") # Visual separator

                # ========================= #
                # 1. Predictive Scenarios   #
                # ========================= #
                st.subheader("1. Predictive Scenarios: Future Projections")
                st.markdown("""
                Project life expectancy trends under different hypothetical GDP growth scenarios.
                This assumes other factors change linearly based on historical data.
                """)

                scenario_countries = df['country_name'].unique().tolist()
                selected_scenario_country = st.selectbox("Select Country for Projection", scenario_countries, key="proj_country")
                
                if selected_scenario_country:
                    base_year_options = sorted(df[df['country_name'] == selected_scenario_country]['year'].unique().tolist(), reverse=True)
                    if base_year_options:
                        base_year = st.selectbox("Base Year for Projection", base_year_options, key="proj_base_year")
                        projection_years = st.slider("Projection Years", 1, 30, 10, key="proj_years")
                        gdp_growth_options = {
                            "Low (1%)": 0.01,
                            "Moderate (3%)": 0.03,
                            "High (5%)": 0.05
                        }
                        selected_gdp_growths = st.multiselect(
                            "Select GDP Growth Scenarios",
                            list(gdp_growth_options.keys()),
                            default=list(gdp_growth_options.keys())[1:2], # Default to Moderate
                            key="proj_gdp_growth"
                        )
                        
                        def predict_future(country, base_year, projection_years, gdp_growth_rates, model, features, df, scaler):
                            """Project life expectancy under different GDP growth scenarios"""
                            baseline_country_data = df[df['country_name'] == country].sort_values('year')
                            if baseline_country_data.empty or base_year not in baseline_country_data['year'].values:
                                return pd.DataFrame()
                            
                            last_data = baseline_country_data[baseline_country_data['year'] == base_year].iloc[0] # This is a Series
                            scenario_dfs_list = [] # Will store DataFrames for each year within a scenario
                            
                            # Calculate historical trends for non-gdp features
                            hist_trends = {}
                            for feature in features:
                                if feature != 'gdp_per_capita' and feature in baseline_country_data.columns:
                                    if len(baseline_country_data[feature].dropna()) >= 2:
                                        hist_trend = baseline_country_data[feature].pct_change().mean()
                                        hist_trends[feature] = hist_trend if not np.isnan(hist_trend) else 0.0
                                    else:
                                        hist_trends[feature] = 0.0

                            for growth_label in gdp_growth_rates:
                                growth_rate = gdp_growth_options[growth_label]
                                current_scenario_rows = [] # Stores rows (as DataFrames) for the current growth scenario
                                
                                # Convert the initial Series (last_data) to a DataFrame before assigning 'gdp_growth'
                                initial_row_df = pd.DataFrame([last_data.to_dict()])
                                current_scenario_rows.append(initial_row_df.assign(gdp_growth=growth_label))

                                current_data_series = last_data.copy() # Keep as Series for modification in the loop
                                
                                for year in range(base_year + 1, base_year + projection_years + 1):
                                    projection_series = current_data_series.copy()
                                    projection_series['year'] = year
                                    
                                    # Apply GDP growth
                                    if 'gdp_per_capita' in features:
                                        projection_series['gdp_per_capita'] *= (1 + growth_rate)
                                    
                                    # Assume other factors change linearly (simplified)
                                    for feature in features:
                                        if feature != 'gdp_per_capita' and feature in projection_series.index:
                                            projection_series[feature] *= (1 + hist_trends.get(feature, 0.0))
                                    
                                    # Predict life expectancy
                                    X_proj_df = pd.DataFrame([projection_series[features]]) # This correctly converts for prediction
                                    X_proj_scaled = scaler.transform(X_proj_df)
                                    
                                    projection_series['life_expectancy'] = model.predict(X_proj_scaled)[0]
                                    
                                    # Convert the projection Series to a DataFrame before assigning 'gdp_growth'
                                    current_scenario_rows.append(pd.DataFrame([projection_series.to_dict()]).assign(gdp_growth=growth_label))
                                    current_data_series = projection_series # Use this Series for the next iteration
                                
                                scenario_dfs_list.append(pd.concat(current_scenario_rows).reset_index(drop=True))
                            
                            if scenario_dfs_list:
                                return pd.concat(scenario_dfs_list).reset_index(drop=True)
                            return pd.DataFrame()

                        if st.button("Generate Projections", key="run_projections"):
                            with st.spinner("Generating future scenarios..."):
                                projections_df = predict_future(
                                    selected_scenario_country,
                                    base_year,
                                    projection_years,
                                    selected_gdp_growths,
                                    model,
                                    features,
                                    df, # Pass original df for historical trends
                                    scaler
                                )
                                if not projections_df.empty:
                                    fig_proj = px.line(
                                        projections_df,
                                        x='year',
                                        y='life_expectancy',
                                        color='gdp_growth',
                                        title=f'Life Expectancy Projections for {selected_scenario_country.title()}',
                                        labels={'life_expectancy': 'Life Expectancy (years)', 'year': 'Year'},
                                        markers=True,
                                        color_discrete_sequence=px.colors.qualitative.Plotly
                                    )
                                    fig_proj.update_layout(
                                        template="plotly_dark",
                                        paper_bgcolor='#141A29',
                                        plot_bgcolor='#1A2235',
                                        font=dict(color='#E0E0E0')
                                    )
                                    st.plotly_chart(fig_proj, use_container_width=True)
                                else:
                                    st.warning("No projections could be generated for the selected country and year. Ensure data exists for the base year.")
                    else:
                        st.info(f"No year data available for {selected_scenario_country} to set a base year.")
                else:
                    st.info("Select a country to generate projections.")
                
                st.markdown("---") # Visual separator

                # =================== #
                # 2. What-If Analysis #
                # =================== #
                st.subheader("2. What-If Analysis: Intervention Impact")
                st.markdown("""
                Simulate the immediate impact of changes in a single key factor on life expectancy for a specific country and year.
                """)

                whatif_country_options = df['country_name'].unique().tolist()
                selected_whatif_country = st.selectbox("Select Country for What-If", whatif_country_options, key="whatif_country")

                if selected_whatif_country:
                    whatif_year_options = sorted(df[df['country_name'] == selected_whatif_country]['year'].unique().tolist(), reverse=True)
                    if whatif_year_options:
                        selected_whatif_year = st.selectbox("Select Year for What-If", whatif_year_options, key="whatif_year")
                    else:
                        st.info(f"No year data available for {selected_whatif_country}.")
                        selected_whatif_year = None
                else:
                    selected_whatif_year = None
                
                if selected_whatif_country and selected_whatif_year:
                    
                    baseline_data_row = df[(df['country_name'] == selected_whatif_country) & (df['year'] == selected_whatif_year)][features + ['life_expectancy']]
                    
                    if not baseline_data_row.empty:
                        baseline_data_row = baseline_data_row.iloc[0]
                        baseline_le = baseline_data_row['life_expectancy']

                        whatif_feature_options = [f for f in features if f != 'life_expectancy']
                        selected_whatif_feature = st.selectbox("Choose Feature for Intervention", whatif_feature_options, key="whatif_feature")

                        if selected_whatif_feature:
                            current_feature_value = baseline_data_row[selected_whatif_feature]
                            if pd.isna(current_feature_value):
                                st.warning(f"No data for '{selected_whatif_feature}' in {selected_whatif_country} for {selected_whatif_year}. Cannot perform what-if analysis on this feature.")
                            else:
                                st.write(f"Current {selected_whatif_feature.replace('_', ' ').title()}: **{current_feature_value:,.2f}**")
                                
                                value_change_type = st.radio(
                                    "Change by:",
                                    ("Percentage Change", "Absolute Change"),
                                    key="whatif_change_type"
                                )

                                new_value = None
                                if value_change_type == "Percentage Change":
                                    percentage_change = st.slider("Percentage Change (%)", -50, 50, 10, key="whatif_perc_change")
                                    new_value = current_feature_value * (1 + percentage_change / 100)
                                else: # Absolute Change
                                    # Determine a sensible range for the slider based on the feature's min/max
                                    feature_min = df[selected_whatif_feature].min()
                                    feature_max = df[selected_whatif_feature].max()
                                    # Ensure the slider value is within reasonable bounds for absolute change
                                    abs_change_default = (feature_max - feature_min) * 0.05 # 5% of range as default step
                                    abs_change = st.number_input(
                                        "Absolute Change",
                                        value=abs_change_default if abs_change_default != 0 else 1.0,
                                        min_value=float(feature_min - current_feature_value),
                                        max_value=float(feature_max - current_feature_value),
                                        step=(feature_max - feature_min) * 0.01 if (feature_max - feature_min) * 0.01 != 0 else 0.1,
                                        key="whatif_abs_change",
                                        format="%.2f"
                                    )
                                    new_value = current_feature_value + abs_change

                                if st.button("Simulate What-If", key="simulate_whatif"):
                                    scenario_data = baseline_data_row[features].copy()
                                    scenario_data[selected_whatif_feature] = new_value

                                    X_scenario_scaled = scaler.transform(pd.DataFrame([scenario_data]))
                                    predicted_le = model.predict(X_scenario_scaled)[0]

                                    st.markdown(f"**Scenario Result for {selected_whatif_country.title()} in {selected_whatif_year}:**")
                                    st.write(f"Baseline Life Expectancy: **{baseline_le:.2f} years**")
                                    st.write(f"Predicted Life Expectancy with modified {selected_whatif_feature.replace('_', ' ').title()} ({new_value:,.2f}): **{predicted_le:.2f} years**")
                                    st.write(f"Change: **{(predicted_le - baseline_le):.2f} years**")

                                    # Plot comparison
                                    comparison_df = pd.DataFrame({
                                        'Scenario': ['Baseline', 'Modified'],
                                        'Life Expectancy': [baseline_le, predicted_le]
                                    })
                                    fig_whatif = px.bar(
                                        comparison_df,
                                        x='Scenario',
                                        y='Life Expectancy',
                                        color='Scenario',
                                        title='Impact of Intervention on Life Expectancy',
                                        color_discrete_map={'Baseline': '#4A8BFF', 'Modified': '#2ECC71'}
                                    )
                                    fig_whatif.update_layout(
                                        template="plotly_dark",
                                        paper_bgcolor='#141A29',
                                        plot_bgcolor='#1A2235',
                                        font=dict(color='#E0E0E0'),
                                        yaxis_range=[min(baseline_le, predicted_le) * 0.95, max(baseline_le, predicted_le) * 1.05]
                                    )
                                    st.plotly_chart(fig_whatif, use_container_width=True)
                        else:
                            st.info("Select a feature to perform what-if analysis.")
                    else:
                        st.warning(f"No complete baseline data for {selected_whatif_country} in {selected_whatif_year} for the selected features.")
                else:
                    st.info("Select a country and year for what-if analysis.")
                
                st.markdown("---") # Visual separator

                # ================================== #
                # 3. Policy Impact Simulation        #
                # ================================== #
                st.subheader("3. Policy Impact Simulation: Interactive Explorer")
                st.markdown("""
                Interactively adjust multiple factors and observe the combined effect on predicted life expectancy.
                """)

                policy_country_options = df['country_name'].unique().tolist()
                selected_policy_country = st.selectbox("Select Country for Simulation", policy_country_options, key="policy_country")

                if selected_policy_country:
                    policy_year_options = sorted(df[df['country_name'] == selected_policy_country]['year'].unique().tolist(), reverse=True)
                    if policy_year_options:
                        selected_policy_year = st.selectbox("Select Year for Simulation", policy_year_options, key="policy_year")
                    else:
                        st.info(f"No year data available for {selected_policy_country}.")
                        selected_policy_year = None
                else:
                    selected_policy_year = None

                if selected_policy_country and selected_policy_year:
                    
                    baseline_policy_data = df[(df['country_name'] == selected_policy_country) & (df['year'] == selected_policy_year)][features + ['life_expectancy']]
                    
                    if not baseline_policy_data.empty:
                        baseline_policy_data = baseline_policy_data.iloc[0]
                        base_le = baseline_policy_data['life_expectancy']
                        
                        st.markdown(f"**Baseline Life Expectancy for {selected_policy_country.title()} in {selected_policy_year}: {base_le:.2f} years**")
                        st.markdown("Adjust the sliders below to create your policy scenario:")

                        # Create sliders for each feature used in the model
                        current_scenario_features = baseline_policy_data[features].copy()
                        
                        sliders_values = {}
                        for feat in features:
                            current_value = current_scenario_features[feat]
                            if pd.isna(current_value):
                                st.warning(f"Skipping slider for '{feat}' due to missing baseline data.")
                                continue
                            
                            min_val = df[feat].min()
                            max_val = df[feat].max()
                            step_val = (max_val - min_val) / 50 if (max_val - min_val) > 0 else 0.1
                            if step_val == 0: step_val = 0.1 # Prevent zero step if min/max are identical

                            sliders_values[feat] = st.slider(
                                f"Modify {feat.replace('_', ' ').title()}",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(current_value),
                                step=float(step_val),
                                key=f"policy_slider_{feat}"
                            )
                        
                        modified_policy_scenario = pd.Series(sliders_values)
                        
                        if st.button("Simulate Policy Impact", key="run_policy_sim"):
                            X_modified_scaled = scaler.transform(pd.DataFrame([modified_policy_scenario]))
                            modified_le = model.predict(X_modified_scaled)[0]
                            improvement = modified_le - base_le

                            st.markdown(f"**Simulated Life Expectancy: {modified_le:.2f} years**")
                            st.markdown(f"**Projected Change from Baseline: {improvement:.2f} years**")

                            # Plotting feature importance (re-using model.coef_ from session state)
                            if not st.session_state.feature_importance.empty:
                                fig_feature_imp_sim = px.bar(st.session_state.feature_importance, 
                                                            x='Coefficient', y='Feature', orientation='h', 
                                                            title='Feature Importance in Prediction Model',
                                                            height=400,
                                                            color_discrete_sequence=px.colors.qualitative.Plotly)
                                fig_feature_imp_sim.update_layout(
                                    yaxis={'categoryorder':'total ascending'},
                                    template="plotly_dark",
                                    paper_bgcolor='#141A29',
                                    plot_bgcolor='#1A2235',
                                    font=dict(color='#E0E0E0')
                                )
                                st.plotly_chart(fig_feature_imp_sim, use_container_width=True)
                            else:
                                st.info("Feature importance data not available. Please train the model in 'Deep Analysis' first.")

                            # Comparison plot
                            comparison_sim_df = pd.DataFrame({
                                'Scenario': ['Baseline', 'Simulated'],
                                'Life Expectancy': [base_le, modified_le]
                            })
                            fig_sim_comp = px.bar(
                                comparison_sim_df,
                                x='Scenario',
                                y='Life Expectancy',
                                color='Scenario',
                                title='Comparison: Baseline vs. Simulated Life Expectancy',
                                color_discrete_map={'Baseline': '#4A8BFF', 'Simulated': '#2ECC71'}
                            )
                            fig_sim_comp.update_layout(
                                template="plotly_dark",
                                paper_bgcolor='#141A29',
                                plot_bgcolor='#1A2235',
                                font=dict(color='#E0E0E0'),
                                yaxis_range=[min(base_le, modified_le) * 0.95, max(base_le, modified_le) * 1.05]
                            )
                            st.plotly_chart(fig_sim_comp, use_container_width=True)
                    else:
                        st.warning(f"No complete baseline data for {selected_policy_country} in {selected_policy_year} for the selected features.")
                else:
                    st.info("Select a country and year for policy impact simulation.")

    else:
        if uploaded_file is None:
            st.info("Upload a dataset to unlock all analysis features.")
        elif df is not None and (not selected_years or not selected_countries) and not df.empty:
            st.info("Please make selections in the sidebar filters to view dashboard content.")
        elif df is not None and df.empty:
            st.info("Uploaded data is empty. Please upload a valid dataset.")
