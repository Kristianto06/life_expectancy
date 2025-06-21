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
import colorcet as cc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pycountry # Add this import at the top

# --- Session State Initialization ---
if 'selected_region_display' not in st.session_state:
    st.session_state.selected_region_display = "Global & Region-Specific"
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = False
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = pd.DataFrame()
if 'r2_score' not in st.session_state:
    st.session_state.r2_score = None

# --- Set page configuration ---
st.set_page_config(
    layout="wide",
    page_title="Life Expectancy Analysis Dashboard",
    page_icon="  📊  "
)

# --- Custom CSS for styling (updated for dark elegant theme based on example URL) ---
st.markdown("""
<style>
    /* General body/app background */
    body {
        background-color: #0E1117; /* Darkest background from Streamlit default dark theme */
        color: #FAFAFA; /* Default text color, light gray */
    }

    /* Main Streamlit container background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Set overall dark theme for plotly plots */
    .js-plotly-plot .plotly .modebar {
        background-color: #1F2228 !important; /* Slightly lighter than app background */
    }
    .js-plotly-plot .plotly .cursor-pointer {
        fill: #FAFAFA !important; /* For text within plots */
    }

    /* Styling for the info/success/warning boxes */
    div[data-testid="stAlert"] {
        border-radius: 0.5rem;
        font-weight: bold;
    }
    div[data-testid="stAlert"] .st-emotion-cache-v06ywu { /* Specific for success */
        background-color: #28A745 !important; /* Brighter Green */
        color: white !important;
        border-radius: 0.5rem;
        border: none;
    }
    div[data-testid="stAlert"] .st-emotion-cache-1jmveo { /* General alert background, used for info/warning */
        background-color: #212529 !important; /* Dark grey for info/warning */
        color: #FAFAFA !important;
        border-radius: 0.5rem;
        border: none;
    }
    div[data-testid="stAlert"] .st-emotion-cache-1y5v8a8 { /* Info alert */
        background-color: #17A2B8 !important; /* Teal blue */
        color: white !important;
    }
    div[data-testid="stAlert"] .st-emotion-cache-1f1i1k7 { /* Warning alert */
        background-color: #FFC107 !important; /* Yellow */
        color: #212529 !important; /* Dark text for warning */
    }
     div[data-testid="stAlert"] .st-emotion-cache-gsvt5j p { /* Error alert (if any) */
        color: white !important;
    }


    /* Introduction Box */
    .intro-box {
        background-color: #1F2228; /* Slightly lighter dark grey */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: #FAFAFA;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Subtle shadow */
    }
    .intro-box h3 {
        color: #FAFAFA;
        font-size: 1.8rem;
        margin-bottom: 10px;
    }
    .intro-box p {
        color: #CCCCCC; /* Slightly darker text for body */
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Metrics Styling - Targeted for screenshot appearance */
    div[data-testid="stMetric"] {
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4); /* More pronounced shadow */
        transition: transform 0.2s ease-in-out; /* Add hover effect */
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px); /* Lift on hover */
    }

    /* Specific colors for metric boxes using nth-child */
    /* Records - Blue */
    div[data-testid="stMetric"]:nth-child(1) > div { 
        background-color: #007BFF !important; /* Bootstrap Blue */
        color: white !important;
    }
    /* Features - Green */
    div[data-testid="stMetric"]:nth-child(2) > div { 
        background-color: #28A745 !important; /* Bootstrap Green */
        color: white !important;
    }
    /* Countries - Orange */
    div[data-testid="stMetric"]:nth-child(3) > div { 
        background-color: #FD7E14 !important; /* Bootstrap Orange */
        color: white !important;
    }
    div[data-testid="stMetric"] label { /* Metric label */
        color: white !important;
        font-size: 1rem;
        font-weight: normal;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { /* Metric value */
        color: white !important;
        font-size: 2.5rem;
        font-weight: bold;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { /* Metric delta */
        color: white !important;
        font-size: 1.2rem;
    }

    /* Tabs Styling */
    .stTabs [data-testid="stTab"] {
        background-color: #1F2228; /* Inactive tab background */
        color: #AAAAAA; /* Lighter grey for inactive text */
        border-radius: 8px 8px 0 0; /* Rounded top corners */
        margin-right: 5px;
        padding: 12px 25px;
        font-weight: bold;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stTabs [data-testid="stTab"]:hover {
        background-color: #2A2E35; /* Slightly lighter on hover */
        color: #FAFAFA;
    }
    .stTabs [data-testid="stTab"][aria-selected="true"] {
        background-color: #2A2E35; /* Active tab background */
        color: #FAFAFA; /* White text for active tab */
        border-bottom: 3px solid #DC3545; /* Red underline for active tab */
        animation: tab-active-border 0.3s forwards; /* Subtle animation */
    }
    @keyframes tab-active-border {
        from { border-bottom-color: transparent; }
        to { border-bottom-color: #DC3545; }
    }

    /* Tab content area background */
    .stTabs [data-testid="stVerticalBlock"] {
        background-color: #1F2228; /* Dark grey for tab content */
        padding: 30px;
        border-radius: 0 0 10px 10px; /* Rounded bottom corners */
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
    }

    /* Sidebar styling */
    .st-emotion-cache-vk33wx { /* Sidebar background */
        background-color: #15181C; /* Slightly different dark for sidebar */
        color: #FAFAFA;
        padding: 20px 15px; /* Adjust padding */
    }
    .st-emotion-cache-vk33wx .st-emotion-cache-1w0nxu { /* Sidebar header */
        color: #FAFAFA;
        font-size: 1.5rem;
        margin-bottom: 15px;
        border-bottom: 1px solid #333; /* Subtle separator */
        padding-bottom: 10px;
    }

    /* Expander styling */
    .st-emotion-cache-1evx060 { /* Expander header wrapper */
        background-color: #2A2E35; /* Darker header for expander */
        border-radius: 8px;
        margin-bottom: 10px;
        overflow: hidden; /* Ensure border-radius applies */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-1evx060 button {
        color: #FAFAFA; /* Expander icon/text color */
        font-weight: bold;
        padding: 15px;
    }
    .st-emotion-cache-1evx060 div[data-testid="stExpanderDetails"] {
        background-color: #1F2228; /* Content background */
        border-radius: 0 0 8px 8px;
        padding: 20px;
        border-top: 1px solid #333;
    }
    .st-emotion-cache-1evx060 div[data-testid="stExpanderDetails"] p {
        color: #CCCCCC; /* Text inside expander details */
    }

    /* Dataframe styling */
    .dataframe {
        background-color: #1F2228; /* Match tab content background */
        color: #FAFAFA;
        border-radius: 8px;
        overflow: hidden; /* Ensures rounded corners are visible */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .dataframe th {
        background-color: #2A2E35; /* Darker header */
        color: #FAFAFA;
        padding: 12px 15px;
        border-bottom: 1px solid #444;
        font-weight: bold;
    }
    .dataframe td {
        color: #FAFAFA;
        padding: 10px 15px;
        border-bottom: 1px solid #2A2E35; /* Subtle row separator */
    }
    .dataframe tr:nth-child(even) {
        background-color: #212529; /* Slightly different for even rows */
    }
    .dataframe tr:hover {
        background-color: #2A2E35; /* Hover effect for rows */
    }

    /* Input widgets (multiselect, selectbox, slider) */
    div[data-testid="stMultiSelect"] > div > div:first-child,
    div[data-testid="stSelectbox"] > div > div:first-child,
    div[data-testid="stSlider"] .st-emotion-cache-1ux495f {
        background-color: #2A2E35; /* Dark background for inputs */
        border: 1px solid #444;
        border-radius: 8px;
        color: #FAFAFA;
        padding: 5px 10px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMultiSelect"] input,
    div[data-testid="stSelectbox"] input {
        color: #FAFAFA;
        background-color: transparent; /* Ensure input text color is visible */
    }
    div[data-testid="stMultiSelect"] .st-emotion-cache-1gx59c3, /* Placeholder/selected items */
    div[data-testid="stSelectbox"] .st-emotion-cache-1gx59c3 {
        color: #FAFAFA;
    }
    /* Options in dropdowns */
    .st-emotion-cache-1xarl3l { /* Dropdown menu background */
        background-color: #2A2E35;
        border: 1px solid #444;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .st-emotion-cache-1xarl3l div[role="option"] {
        color: #FAFAFA;
        padding: 8px 12px;
    }
    .st-emotion-cache-1xarl3l div[role="option"]:hover,
    .st-emotion-cache-1xarl3l div[aria-selected="true"] {
        background-color: #007BFF; /* Highlight on hover/selected */
        color: white;
    }
    /* Slider specific styling */
    .stSlider .st-emotion-cache-1f81tsl { /* Slider track */
        background-color: #444;
    }
    .stSlider .st-emotion-cache-1f81tsl > div { /* Slider fill */
        background-color: #007BFF;
    }
    .stSlider .st-emotion-cache-17l1x26 { /* Slider thumb */
        background-color: #007BFF;
        border: 2px solid #FAFAFA;
    }


    /* Headers and titles */
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA; /* White for all headers */
    }

    /* Markdown text */
    p {
        color: #CCCCCC; /* Light gray for general text */
    }

    /* Plotly container background */
    .stPlotlyChart {
        background-color: #1F2228; /* Match tab content/dataframe background */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    /* Matplotlib plot container */
    div[data-testid="stFigure"] {
        background-color: #1F2228;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    /* Streamlit components specific for the screenshot */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background-color: transparent !important; /* To prevent nested blocks from having colored backgrounds */
    }

    /* Folium map container */
    .folium-map {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }

    /* Specific CSS for `st.button` for a more elegant look */
    .st-emotion-cache-19p3v54 { /* This targets the button element's wrapper */
        background-color: #007BFF; /* Bootstrap Blue for buttons */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-19p3v54:hover {
        background-color: #0056B3; /* Darker blue on hover */
        transform: translateY(-2px); /* Slight lift on hover */
    }

    /* For dataframes (more specific targets for clearer styling) */
    .st-emotion-cache-1mmp0p3 thead th { /* Table header */
        background-color: #2A2E35;
        color: #FAFAFA;
        border-bottom: 1px solid #444;
        font-size: 0.95rem;
    }
    .st-emotion-cache-1mmp0p3 tbody tr { /* Table rows */
        background-color: #1F2228;
        color: #FAFAFA;
    }
    .st-emotion-cache-1mmp0p3 tbody tr:nth-child(even) {
        background-color: #212529; /* Alternate row color */
    }
    .st-emotion-cache-1mmp0p3 tbody tr:hover {
        background-color: #2A2E35; /* Hover effect for rows */
    }

    /* CSS for parallel coordinates */
    .parcoords > svg > g > g.tick > text {
        fill: #FAFAFA !important; /* White text for ticks */
        font-size: 12px !important;
    }
    .parcoords > svg > g > g.tick > line {
        stroke: #007BFF !important; /* Blue lines for ticks */
    }
</style>
""", unsafe_allow_html=True)

# --- Introduction ---
st.title(f"  🌏   The Longevity Puzzle: Unlocking the Secrets to a Longer Life in East Asia & Pacific: {st.session_state.selected_region_display}")
st.markdown("""
<div class="intro-box">
    <h3>Uncover What Shapes a Longer Life</h3>
    <p>This dashboard delves into the critical socioeconomic and health forces that influence life expectancy across diverse regions. Upload your dataset to unlock insights that could drive meaningful change.</p>
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

uploaded_file = st.sidebar.file_uploader("Upload Life Expectancy Data (CSV/XLSX)", type=["csv", "xlsx"])
df = None
if uploaded_file is not None:
    with st.spinner('Analyzing data...'):
        df = load_data(uploaded_file)
        if df is not None:
            REQUIRED_COLUMNS = ['life_expectancy', 'country_name', 'year']
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                st.error(f"Missing required columns for analysis: {', '.join(missing)}. Please upload a dataset containing these columns.")
                st.stop() 
            st.success("  ✅   Data loaded successfully!")
            
            # Display metrics using Streamlit's native metric component
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
        st.info("Please upload a dataset to begin analysis.")

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
    st.sidebar.header("Data Filters")
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

    # --- Conditional Display of Tabs based on filtered_df content ---
    if not filtered_df.empty:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "  📊   Overview",
            "  📈   Relationships",
            "  🗺️   Geographic",
            "  🔍   Deep Analysis",
            "  💡   Recommendations"
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
            
            with st.expander("  📋   Summary Statistics"):
                st.write(filtered_df.describe())

            if 'year' in filtered_df.columns and 'life_expectancy' in filtered_df.columns:
                st.subheader("Life Expectancy Trend Analysis")
                if len(filtered_df['country_name'].unique()) == 1:
                    country = filtered_df['country_name'].iloc[0]
                    country_time_series_df = df[df['country_name'] == country].set_index('year')['life_expectancy'].sort_index()
                    if len(country_time_series_df) > 2:
                        plt.style.use('seaborn-v0_8-darkgrid')
                        decomposition = seasonal_decompose(country_time_series_df, model='additive', period=1) # Adjust period as needed
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
                        fig.patch.set_facecolor('#1F2228') # Consistent dark background for Matplotlib figure
                        ax1.set_facecolor('#212529') # Consistent dark background for Matplotlib axes
                        ax2.set_facecolor('#212529')
                        ax3.set_facecolor('#212529')
                        ax1.plot(decomposition.trend, color='#007BFF', linewidth=2) # Consistent blue line
                        ax1.set_title('Trend Component', fontsize=16, color='#FAFAFA')
                        ax1.set_ylabel('Life Expectancy', fontsize=12, color='#FAFAFA')
                        ax1.tick_params(axis='x', colors='#FAFAFA')
                        ax1.tick_params(axis='y', colors='#FAFAFA')
                        ax1.grid(True, linestyle='--', alpha=0.5, color='#6C757D') # Softer grid lines
                        ax2.plot(decomposition.seasonal, color='#28A745', linewidth=1.5) # Consistent green line
                        ax2.set_title('Seasonal Component', fontsize=16, color='#FAFAFA')
                        ax2.set_ylabel('Seasonality', fontsize=12, color='#FAFAFA')
                        ax2.tick_params(axis='x', colors='#FAFAFA')
                        ax2.tick_params(axis='y', colors='#FAFAFA')
                        ax2.grid(True, linestyle='--', alpha=0.5, color='#6C757D')
                        ax3.plot(decomposition.resid, color='#FD7E14', linewidth=1) # Consistent orange line
                        ax3.set_title('Residuals Component', fontsize=16, color='#FAFAFA')
                        ax3.set_xlabel('Year', fontsize=12, color='#FAFAFA')
                        ax3.set_ylabel('Residual', fontsize=12, color='#FAFAFA')
                        ax3.tick_params(axis='x', colors='#FAFAFA')
                        ax3.tick_params(axis='y', colors='#FAFAFA')
                        ax3.grid(True, linestyle='--', alpha=0.5, color='#6C757D')
                        fig.suptitle(f'Time Series Decomposition for {country.title()} Life Expectancy', fontsize=20, color='#FAFAFA')
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        st.pyplot(fig)
                        plt.close(fig)
                        st.info("""
                        **Notes on Seasonal and Residual Plots:**
                        If the 'Seasonal Component' and 'Residual Component' plots appear flat, this often means the `period` parameter in the `seasonal_decompose` function is set to `1`. For annual data, setting `period=1` implies no seasonality within one year, so these components will reflect this assumption. If you expect longer-term seasonal cycles (e.g., cycles spanning several years), you need to adjust the `period` parameter to the length of that cycle.
                        
                        **Interpreting the Seasonal Component:**
                        * **Flat Line:** Indicates no detectable repeating patterns within the specified `period`. For annual data with `period=1`, this is expected as there's no intra-year seasonality.
                        * **Fluctuating Pattern:** If `period` is set appropriately (e.g., if you had monthly data and `period=12`), a fluctuating line would reveal consistent, repeating patterns or cycles in life expectancy that occur regularly over that period. This could highlight annual health trends or policy impacts.
                        
                        **Interpreting the Residuals Component:**
                        * **Random Scatter (around zero):** This is the ideal outcome. It suggests that the trend and seasonal components have successfully captured most of the underlying patterns in the data, leaving only random noise. This implies the model is a good fit.
                        * **Patterns or Trends:** If the residuals show noticeable patterns (e.g., a rising or falling trend, or clear cycles), it indicates that the model has not fully captured all the systematic information in the data. This might suggest the need for a more complex model or different features.
                        * **Large Spikes:** Could indicate outliers or unusual events that significantly impacted life expectancy during those periods. Further investigation into these specific points might reveal external factors.
                        """)
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
                    colorscale='Viridis', # Changed to Viridis for better contrast on dark theme
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
                    paper_bgcolor='#1F2228', # Consistent dark background
                    plot_bgcolor='#212529', # Consistent dark background
                    font=dict(color='#FAFAFA')
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                corr_unstacked = corr_df.unstack().reset_index()
                corr_unstacked.columns = ['Var1', 'Var2', 'Correlation']
                corr_unstacked = corr_unstacked[corr_unstacked['Var1'] != corr_unstacked['Var2']]
                top_corrs = corr_unstacked.sort_values('Correlation', ascending=False).head(5)
                bottom_corrs = corr_unstacked.sort_values('Correlation', ascending=True).head(5)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top Positive Correlations:**")
                    for _, row in top_corrs.iterrows():
                        st.markdown(f"- **{row['Var1'].replace('_', ' ').title()}** & **{row['Var2'].replace('_', ' ').title()}**: {row['Correlation']:.2f}")
                
                with col2:
                    st.write("**Top Negative Correlations:**")
                    for _, row in bottom_corrs.iterrows():
                        st.markdown(f"- **{row['Var1'].replace('_', ' ').title()}** & **{row['Var2'].replace('_', ' ').title()}**: {row['Correlation']:.2f}")
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
            
            st.info("""
            **Mapping Note:** This interactive map visualizes life expectancy by country.
            Countries are identified using ISO 3-letter country codes derived from their names.
            """)
            
            if 'year' in df.columns and 'country_name' in df.columns and not filtered_df.empty:
                st.subheader("Interactive World Map")
                
                # Create ISO alpha3 country codes
                def get_iso_alpha3(country_name):
                    try:
                        return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
                    except:
                        return None
                        
                # Rename the column from 'iso_alpha3' to 'Name' as requested
                filtered_df['Name'] = filtered_df['country_name'].apply(get_iso_alpha3)

                map_df = filtered_df.groupby('country_name').agg({
                    'life_expectancy': 'mean',
                    'gdp_per_capita': 'mean',
                    'Name': 'first' # Use 'Name' here
                }).reset_index().dropna(subset=['Name']) # Drop rows where ISO code couldn't be found
                
                if not map_df.empty:
                    fig_map = px.choropleth(
                        map_df,
                        locations="Name",  # Use 'Name' for locations
                        color='life_expectancy', # Color by life expectancy
                        hover_name="country_name",
                        hover_data=['gdp_per_capita'],
                        color_continuous_scale='Plasma', # Plasma color scale, matching Plotly defaults for dark theme
                        title="Global Life Expectancy Distribution",
                        height=700
                    )
                    fig_map.update_layout(
                        height=700,
                        template="plotly_dark", # Dark theme for the map
                        paper_bgcolor='#1F2228', # Background color for the paper
                        plot_bgcolor='#212529', # Background color for the plot area
                        font=dict(color='#FAFAFA'), # Font color for text
                        geo=dict(
                            bgcolor='rgba(0,0,0,0)', # Transparent background for the geographical area
                            lakecolor='#17A2B8', # Teal for lakes
                            landcolor='#2A2E35', # Darker land color
                            showocean=True, # Show ocean
                            oceancolor='#1F2228' # Ocean color matching background
                        )
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Could not generate map - missing ISO country codes for selected countries or no data.")
            else:
                st.info("Geographic analysis requires 'year' and 'country_name' columns in the dataset, and selections to be made in the sidebar.")

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
                        color_discrete_sequence=px.colors.qualitative.Plotly # Use Plotly's default qualitative colors
                    )
                    
                    global_avg = df.groupby('year')['life_expectancy'].mean().reset_index()
                    if not global_avg.empty:
                        fig.add_hline(y=global_avg['life_expectancy'].mean(),
                                      line_dash="dot",
                                      annotation_text="Global Average",
                                      annotation_position="bottom right",
                                      line_color='#6C757D') # Softer line color
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='#1F2228',
                        plot_bgcolor='#212529',
                        font=dict(color='#FAFAFA'),
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
                                paper_bgcolor='#1F2228',
                                plot_bgcolor='#212529',
                                font=dict(color='#FAFAFA')
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create bins for '{feature}': {e}. Skipping box plot.")

            st.subheader("Predictive Modeling: Factors Influencing Life Expectancy")
            st.markdown("Build a simple predictive model to understand feature importance.")
            target = 'life_expectancy'
            
            potential_features = [col for col in num_cols_filtered if col != target and col not in ['year']]
            
            if not potential_features:
                st.warning("No suitable numerical features found for predictive modeling after filtering.")
            elif filtered_df.empty:
                st.info("Filtered dataset is empty, cannot perform predictive modeling.")
            else:
                selected_model_features = st.multiselect(
                    "Select features for predictive model",
                    potential_features,
                    default=[pf for pf in ['gdp_per_capita', 'health_expenditure_per_capita', 'school_enrollment_combined', 'mortality_infant', 'access_to_electricity', 'population_growth_rate'] if pf in potential_features]
                )
                if selected_model_features:
                    clean_df = filtered_df[[target] + selected_model_features].dropna()
                    
                    min_total_samples_for_model = 10
                    
                    if len(clean_df) < min_total_samples_for_model:
                        st.warning(f"Insufficient complete data rows ({len(clean_df)} samples) for predictive modeling. "
                                   f"Please ensure at least {min_total_samples_for_model} samples after selecting features and handling missing values.")
                    else: 
                        X_clean = clean_df[selected_model_features]
                        y_clean = clean_df[target]
                        if len(np.unique(y_clean)) <= 1:
                            st.warning("Insufficient variation in target (life expectancy) for predictive modeling. All values are the same or too few unique values.")
                        else:
                            min_splits_cv = 5
                            max_test_size = 1 - (min_splits_cv / len(clean_df))
                            test_size_val = max(0.1, min(0.5, max_test_size))
                            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size_val, random_state=42)
                            
                            n_samples_train = len(X_train)
                            n_splits_adjusted = min(n_samples_train, min_splits_cv)
                            
                            if n_splits_adjusted < 2:
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                model = Ridge(alpha=1.0)
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                st.write(f"**Test Set R-squared:** {r2:.2f}")
                                st.write(f"**Mean Squared Error:** {mse:.2f}")
                                
                                st.session_state['feature_importance'] = pd.DataFrame({
                                    'Feature': selected_model_features,
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False)
                            else:
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test) 
                                alphas = [0.001, 0.01, 0.1, 1, 10, 100]
                                model = RidgeCV(alphas=alphas, cv=n_splits_adjusted) 
                                model.fit(X_train_scaled, y_train)
                                
                                kfold = KFold(n_splits=n_splits_adjusted, shuffle=True, random_state=42)
                                cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                                            cv=kfold, scoring='r2')
                                st.write(f"**Best Alpha (Regularization):** {model.alpha_:.4f}")
                                st.write(f"**Average CV R-squared:** {cv_scores.mean():.2f}")
                                
                                y_pred = model.predict(X_test_scaled)
                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                
                                st.session_state['r2_score'] = r2
                                st.session_state['feature_importance'] = pd.DataFrame({
                                    'Feature': selected_model_features,
                                    'Coefficient': model.coef_
                                }).sort_values('Coefficient', key=abs, ascending=False)
                                
                                st.write(f"**Test Set R-squared:** {r2:.2f}")
                                st.write(f"**Mean Squared Error:** {mse:.2f}")
                            st.subheader("Feature Importance (Coefficients)")
                            st.dataframe(st.session_state['feature_importance'])
                            fig_feature_imp = px.bar(st.session_state['feature_importance'], 
                                                    x='Coefficient', y='Feature', orientation='h', 
                                                    title='Feature Impact (Coefficients) on Life Expectancy Prediction',
                                                    height=600,
                                                    color_discrete_sequence=px.colors.qualitative.Plotly)
                            fig_feature_imp.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                template="plotly_dark",
                                paper_bgcolor='#1F2228',
                                plot_bgcolor='#212529',
                                font=dict(color='#FAFAFA')
                            )
                            st.plotly_chart(fig_feature_imp, use_container_width=True)
                else:
                    st.info("Select features to build a predictive model.")
            
            # --- Improved Parallel Coordinates Plot ---
            st.subheader("Multi-dimensional Data Relationships (Parallel Coordinates)")
            st.markdown("Visualize patterns and clusters across multiple numerical features.")
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
                "Select variables for Parallel Coordinates dimensions",
                parallel_coord_cols_options,
                default=safe_default_pc_dimensions
            )
            color_choice = st.selectbox(
                "Color lines by",
                all_color_options,
                index=all_color_options.index('life_expectancy') if 'life_expectancy' in all_color_options else (all_color_options.index('None') if 'None' in all_color_options else 0)
            )
            
            if selected_parallel_cols and len(selected_parallel_cols) > 1:
                # Create a clean DataFrame with only the needed columns
                cols_for_plot = list(set(selected_parallel_cols + ([color_choice] if color_choice != 'None' else [])))
                pc_df = filtered_df[cols_for_plot].dropna().copy()
                
                if not pc_df.empty:
                    # Create custom dimensions with proper ranges
                    dimensions = []
                    for col in selected_parallel_cols:
                        if col in pc_df.columns:
                            col_min = pc_df[col].min()
                            col_max = pc_df[col].max()
                            range_buffer = (col_max - col_min) * 0.1 # 10% buffer for range
                            
                            dim_config = {
                                "label": col.replace('_', ' ').title(),
                                "values": pc_df[col],
                                "range": [col_min - range_buffer, col_max + range_buffer]
                            }
                            
                            # Special handling for life expectancy
                            if col == 'life_expectancy':
                                dim_config["range"] = [40, 90]  # Fixed range for better comparison
                            
                            dimensions.append(dim_config)
                    
                    if len(dimensions) >= 2:
                        fig_par_coords = go.Figure(data=go.Parcoords(
                            line=dict(
                                color=pc_df[color_choice] if color_choice != 'None' and color_choice in pc_df.columns else '#007BFF', # Consistent blue line if no color chosen
                                colorscale='Viridis', # Use Viridis colorscale
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
                            paper_bgcolor='#1F2228',
                            plot_bgcolor='#212529',
                            font=dict(color='#FAFAFA'),
                            margin=dict(l=80, r=80, t=80, b=80)
                        )
                        st.plotly_chart(fig_par_coords, use_container_width=True)
                    else:
                        st.info("Not enough valid dimensions to plot")
                else:
                    st.info("No complete data for selected variables")
            else:
                st.info("Please select at least two numerical variables")

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
                        st.markdown("**Top Performers (Life Expectancy Growth)**")
                        for _, row in top_growth.iterrows():
                            st.markdown(f"- **{row['country_name'].title()}**: +{row['le_growth']:.1f} years")
                    
                    with col2:
                        st.markdown("**Countries Needing Improvement**")
                        for _, row in bottom_growth.iterrows():
                            st.markdown(f"- **{row['country_name'].title()}**: {row['le_growth']:.1f} years")
                else:
                    st.info("Insufficient year data to calculate life expectancy growth. Please select multiple years.")
                
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
                        st.dataframe(gap_df.style.background_gradient(cmap='RdYlGn', subset=['life_expectancy_gap', 'gdp_gap_percent', 'health_exp_gap_percent', 'education_gap'], axis=0))
                    else:
                        st.info("Select a base country to perform gap analysis.")
                else:
                    st.info("Select more than one country in the sidebar to perform country comparison analysis.")
            else:
                st.info("Please ensure data is loaded and filtered to see recommendations.")
    else:
        if uploaded_file is None:
            st.info("Upload a dataset to unlock all analysis features.")
        elif df is not None and (not selected_years or not selected_countries) and not df.empty:
            st.info("Please make selections in the sidebar filters (Years and Countries) to view dashboard content. Currently no data is selected to display.")
        elif df is not None and df.empty:
            st.info("Uploaded data is empty. Please upload a valid dataset.")
