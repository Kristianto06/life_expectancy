import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium, folium_static # Ensure folium_static is imported
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# --- Session State Initialization ---
# Initialize session state variables to maintain their values across Streamlit reruns
if 'selected_region_display' not in st.session_state:
    st.session_state.selected_region_display = "Global & Region-Specific"
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = False
# Initialize feature_importance and r2_score for the predictive model tab
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = pd.DataFrame()
if 'r2_score' not in st.session_state:
    st.session_state.r2_score = None

# --- Set page configuration ---
# Configure the Streamlit page layout to be wide, set the title and icon
st.set_page_config(
    layout="wide",
    page_title="Life Expectancy Analysis Dashboard",
    page_icon="📊"
)

# --- Custom CSS for styling ---
# Apply custom CSS for a dark theme and specific UI element styling
st.markdown("""
<style>
    /* Overall dark theme for the body and main content area */
    body, .main {
        background-color: #283747; /* Slightly softer dark blue-grey */
        color: #EBF5FB; /* Lighter text color for better contrast */
        font-family: 'Inter', sans-serif; /* Professional font */
    }
    
    /* Streamlit widgets styling for a cohesive look */
    .stSelectbox, .stMultiSelect, .stSlider, .stTextInput {
        background-color: #34495E; /* Darker background for input fields */
        border-radius: 8px; /* More rounded corners */
        border: 1px solid #4A698A; /* Subtle border */
        color: #EBF5FB; /* Text color */
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #34495E;
        border-radius: 8px;
    }
    .stSelectbox div[data-baseweb="select"] > div:first-child {
        color: #EBF5FB;
    }
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #5DADE2 !important; /* Blue tags for selected items */
        color: white !important;
        border-radius: 5px;
    }

    /* Styling for Streamlit buttons */
    .stButton>button {
        background-color: #3498DB; /* Professional blue */
        color: white; /* White text */
        border-radius: 8px; /* More rounded corners */
        border: none; /* No border */
        padding: 10px 20px;
        transition: background-color 0.3s ease; /* Smooth transition for hover */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
    .stButton>button:hover {
        background-color: #2874A6; /* Darker blue on hover */
    }
    
    /* Styling for the introduction box */
    .intro-box {
        background-color:#E8F8F5; /* Very light cyan for a welcoming feel */
        padding:25px; /* More padding */
        border-radius:12px; /* More rounded corners */
        margin-bottom:35px; /* More space below */
        box-shadow: 0 6px 10px rgba(0,0,0,0.15); /* Stronger, professional shadow */
        color: #2C3E50; /* Dark text for contrast on light background */
    }

    /* Styling for feature display boxes */
    .feature-box {
        background-color: #34495E; /* Matches input fields for consistency */
        color: #EBF5FB; /* Light text */
        border-radius: 12px; /* More rounded corners */
        padding: 20px; /* More padding */
        margin-bottom: 20px; /* More space below */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); /* Enhanced shadow */
    }

    /* Styling for metric display cards - vibrant gradient */
    .metric-card {
        background: linear-gradient(135deg, #7D3C98 0%, #2980B9 100%); /* Purple to Blue gradient */
        color: white; /* White text */
        border-radius: 12px; /* More rounded corners */
        padding: 20px; /* More padding */
        text-align: center; /* Centered text */
        margin-bottom: 20px; /* More space below */
        box-shadow: 0 6px 12px rgba(0,0,0,0.25); /* Prominent shadow */
        transition: transform 0.2s ease-in-out; /* Pop effect on hover */
    }
    .metric-card:hover {
        transform: translateY(-5px); /* Lift effect */
    }
    .metric-card h3 {
        color: white; /* Ensure metric titles are white */
    }
    .metric-card div[data-testid="stMetricValue"] {
        color: white !important; /* Ensure metric values are white */
    }

    /* Styling for highlighted sections */
    .highlight {
        background-color: #3E546B; /* Darker grey-blue */
        border-left: 5px solid #5DADE2; /* Brighter blue border */
        padding: 15px; /* More padding */
        border-radius: 0 8px 8px 0; /* Slightly more rounded right corners */
        color: #EBF5FB; /* Light text */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Styling for call-to-action boxes */
    .call-to-action-box {
        background-color: #5D6D7E; /* Medium blue-grey, professional */
        padding: 20px; /* More padding */
        border-radius: 12px; /* More rounded corners */
        margin-top: 30px; /* More space above */
        color: #EBF5FB; /* Light text */
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }

    /* Styling for the Folium map container */
    .folium-map {
        border-radius: 12px; /* More rounded corners */
        overflow: hidden; /* Hide overflow to respect border-radius */
        box-shadow: 0 6px 12px rgba(0,0,0,0.25); /* Prominent shadow */
        margin-bottom: 25px; /* More space below */
    }

    /* NEW: Styling for the Streamlit tabs to make them bigger and more prominent */
    div[data-testid="stTabs"] button {
        font-size: 1.3em; /* Make text even bigger */
        padding: 18px 35px; /* Increase padding for a larger touch target */
        margin: 7px; /* Add some margin between tabs */
        border-radius: 12px; /* Keep consistent rounded corners */
        background-color: #3E546B; /* Slightly lighter dark blue for tab background */
        color: #EBF5FB; /* Light text color */
        border: 1px solid #5DADE2; /* A subtle border to define them */
        transition: all 0.3s ease;
    }
    div[data-testid="stTabs"] button:hover {
        background-color: #5DADE2; /* Highlight on hover */
        color: #2C3E50; /* Darker text on hover for contrast */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #3498DB; /* Professional blue for selected tab */
        color: white;
        border: 1px solid #3498DB;
        font-weight: bold; /* Make selected tab text bold */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }

</style>
""", unsafe_allow_html=True)

# --- Introduction ---
# Display the main title of the dashboard, dynamically showing the selected region
st.title(f"🌏 The Longevity Puzzle: Unlocking the Secrets to a Longer Life in East Asia & Pacific: {st.session_state.selected_region_display}")
# Markdown for a brief introduction using custom styling
st.markdown("""
<div class="intro-box">
    <h3>Uncover What Shapes a Longer Life</h3>
    <p>This dashboard delves into the critical socioeconomic and health forces that influence life expectancy across diverse regions. Upload your dataset to unlock insights that could drive meaningful change.</p>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
# Decorator to cache data loading, so it only runs when the file changes
@st.cache_data
def load_data(uploaded_file):
    """
    Loads data from an uploaded CSV or XLSX file and cleans column names.
    """
    try:
        # Read the file based on its extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            # Display an error for unsupported file types
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            return None
        
        # Clean column names: strip spaces, convert to lowercase, replace special chars with underscores
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
        # Catch and display any errors during data loading
        st.error(f"Error loading data: {e}")
        return None

# File uploader widget in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Life Expectancy Data (CSV/XLSX)", type=["csv", "xlsx"])
df = None # Initialize DataFrame to None

# Process uploaded file if available
if uploaded_file is not None:
    with st.spinner('Analyzing data...'): # Show a spinner while data is being processed
        df = load_data(uploaded_file)
        if df is not None:
            # Define required columns for the analysis
            REQUIRED_COLUMNS = ['life_expectancy', 'country_name', 'year']
            # Check for missing required columns
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                st.error(f"Missing required columns for analysis: {', '.join(missing)}. Please upload a dataset containing these columns.")
                st.stop() 
            st.success("✅ Data loaded successfully!")

            # Display key metrics about the loaded data
            cols = st.columns(3)
            # Apply metric card styling
            with cols[0]:
                st.markdown(
                    f'''
                    <div class="metric-card" style="background:#3498db; color:white; display: flex; align-items: center; justify-content: center; padding:10px 12px;">
                        <span style="font-size:1.5em; margin-right: 12px;">📄</span>
                        <div style="text-align:left;">
                            <h4 style="margin:0; font-size:1em;">Records</h4>
                            <span style="margin:0; font-size:2.2em; font-weight:bold; line-height:1;">{df.shape[0]}</span>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            with cols[1]:
                st.markdown(
                    f'''
                    <div class="metric-card" style="background:#2ecc71; color:white; display: flex; align-items: center; justify-content: center; padding:10px 12px;">
                        <span style="font-size:1.5em; margin-right: 12px;">🧬</span>
                        <div style="text-align:left;">
                            <h4 style="margin:0; font-size:1em;">Features</h4>
                            <span style="margin:0; font-size:2.2em; font-weight:bold; line-height:1;">{df.shape[1]}</span>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            with cols[2]:
                st.markdown(
                    f'''
                    <div class="metric-card" style="background:#e67e22; color:white; display: flex; align-items: center; justify-content: center; padding:10px 12px;">
                        <span style="font-size:1.5em; margin-right: 12px;">🌍</span>
                        <div style="text-align:left;">
                            <h4 style="margin:0; font-size:1em;">Countries</h4>
                            <span style="margin:0; font-size:2.2em; font-weight:bold; line-height:1;">{df["country_name"].nunique()}</span>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            st.session_state['initial_load'] = True # Set flag for initial load success
        else:
            st.info("Uploaded file could not be processed. Please check the file format or its content.")
else:
    # Prompt user to upload data if no file is uploaded yet
    if 'initial_load' not in st.session_state or not st.session_state.initial_load:
        st.info("Please upload a dataset to begin analysis.")

# Initialize filtered_df as an empty DataFrame with the original columns if df is loaded, otherwise just empty
filtered_df = pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame() 
selected_countries = [] # Initialize selected countries list

# Only proceed with analysis if a DataFrame is loaded
if df is not None:
    # --- Data Processing ---
    # Convert 'life_expectancy' to numeric, coercing errors to NaN
    if 'life_expectancy' in df.columns:
        df['life_expectancy'] = pd.to_numeric(df['life_expectancy'], errors='coerce')

    # Fill missing numerical values with their median
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop rows where 'country_name' is missing
    df.dropna(subset=['country_name'], inplace=True)

    # --- Sidebar Filters ---
    st.sidebar.header("Data Filters")

    years = sorted(df['year'].unique())
    selected_years = [] # Default to empty
    if len(years) > 0:
        selected_years = st.sidebar.multiselect(
            "Select Year(s)",
            years,
            default=[] # Set default to empty list as per request
        )
    else:
        st.sidebar.warning("No year data available")
        
    temp_df = df.copy() # Start with a copy of the full DataFrame for filtering

    # Apply year filter. If no years are selected, make temp_df empty.
    if selected_years:
        temp_df = temp_df[temp_df['year'].isin(selected_years)]
    elif 'year' in df.columns and len(years) > 0: # Years exist in original df but none selected
        temp_df = pd.DataFrame(columns=df.columns) # Create an empty DataFrame with original columns to avoid KeyError later
        st.sidebar.info("Select at least one year to filter data.")

    # Region filter
    if 'region' in df.columns: # Check if original df has 'region' column
        if not temp_df.empty and 'region' in temp_df.columns: # If temp_df is not empty and has 'region'
            regions = sorted(temp_df['region'].unique().tolist())
            selected_region = st.sidebar.selectbox("Select Region", ['All'] + regions)
            st.session_state.selected_region_display = selected_region
            if selected_region != 'All':
                temp_df = temp_df[temp_df['region'] == selected_region]
        else:
            # If temp_df is empty or lacks 'region' (e.g., due to year filter)
            # still offer 'All' but set display info.
            selected_region = st.sidebar.selectbox("Select Region", ['All']) # Still show 'All' option
            st.session_state.selected_region_display = "Global (No regional data for current selection)"
            st.sidebar.info("No regional data available for the current year(s) selection.")
    else: # Original df did not have region column
        st.sidebar.info("Region column not found for region filtering.")
        st.session_state.selected_region_display = "Global (Region column missing)"


    # Country filter
    all_countries = []
    # Only populate all_countries if temp_df is not empty and has 'country_name' column
    if not temp_df.empty and 'country_name' in temp_df.columns:
        all_countries = sorted(temp_df['country_name'].unique().tolist())
    
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        all_countries,
        default=[] # Always start empty
    )

    # Apply final country filter to get filtered_df
    # filtered_df should be non-empty only if countries are selected and data exists in temp_df
    if selected_countries and not temp_df.empty and 'country_name' in temp_df.columns:
        filtered_df = temp_df[temp_df['country_name'].isin(selected_countries)].copy()
    else:
        # If no countries selected, or temp_df is empty, or 'country_name' is missing,
        # filtered_df should be an empty DataFrame.
        filtered_df = pd.DataFrame(columns=df.columns) # Use df's columns for consistency

    # Final check for 'life_expectancy' column in the filtered data.
    # This check is now less critical as filtered_df can be explicitly empty due to filters.
    if 'life_expectancy' not in filtered_df.columns and not filtered_df.empty: # Only raise if df is not empty
        st.error("The 'life_expectancy' column is missing in your filtered dataset. Please ensure it is present and numeric.")
        filtered_df = pd.DataFrame(columns=df.columns) # Ensure filtered_df is empty to prevent further errors

    # --- Conditional Display of Tabs based on filtered_df content ---
    if not filtered_df.empty:
        # Get numerical columns from the filtered DataFrame (re-evaluate as filtered_df might change)
        num_cols_filtered = filtered_df.select_dtypes(include=['number']).columns

        # --- Tabs for Analysis Sections ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "📈 Relationships",
            "🗺️ Geographic",
            "🔍 Deep Analysis",
            "💡 Recommendations"
        ])

        with tab1:  # Overview Tab
            st.header("Data Overview")
            cols = st.columns(4)
            # Display key metrics using Streamlit's metric widget
            # Apply metric card styling to these as well for consistency
            with cols[0]:
                if 'life_expectancy' in num_cols_filtered:
                    st.markdown(f'<div class="metric-card"><h3>Avg Life Expectancy</h3><h1>{filtered_df["life_expectancy"].mean():.1f} yrs</h1></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><h3>Avg Life Expectancy</h3><p>N/A</p></div>', unsafe_allow_html=True)
            with cols[1]:
                if 'gdp_per_capita' in num_cols_filtered:
                    st.markdown(f'<div class="metric-card"><h3>Avg GDP per Capita</h3><h1>${filtered_df["gdp_per_capita"].mean():,.0f}</h1></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><h3>Avg GDP per Capita</h3><p>N/A</p></div>', unsafe_allow_html=True)
            with cols[2]:
                if 'health_expenditure_per_capita' in num_cols_filtered:
                    st.markdown(f'<div class="metric-card"><h3>Healthcare Expenditure</h3><h1>${filtered_df["health_expenditure_per_capita"].mean():,.0f}</h1></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><h3>Healthcare Exp.</h3><p>N/A</p></div>', unsafe_allow_html=True)
            with cols[3]:
                if 'school_enrollment_combined' in num_cols_filtered:
                    st.markdown(f'<div class="metric-card"><h3>School Enrollment</h3><h1>{filtered_df["school_enrollment_combined"].mean():.1f}%</h1></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><h3>School Enrollment</h3><p>N/A</p></div>', unsafe_allow_html=True)
            
            # Expandable section for summary statistics
            with st.expander("📋 Summary Statistics"):
                st.write(filtered_df.describe())

            # Time Series Decomposition
            if 'year' in filtered_df.columns and 'life_expectancy' in filtered_df.columns:
                st.subheader("Life Expectancy Trend Analysis")
                if len(filtered_df['country_name'].unique()) == 1:
                    country = filtered_df['country_name'].iloc[0]
                    # Filter data for the single selected country across all years to show trend
                    country_time_series_df = df[df['country_name'] == country].set_index('year')['life_expectancy'].sort_index()

                    if len(country_time_series_df) > 2: # Requires at least 3 points for decomposition
                        # Set a professional Matplotlib style
                        plt.style.use('seaborn-v0_8-darkgrid') # 'seaborn-darkgrid' for a clean, professional look
                        
                        decomposition = seasonal_decompose(country_time_series_df, model='additive', period=1)
                        
                        # Plot trend, seasonality, and residuals with improved styling
                        # Increased figsize for bigger charts
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
                        
                        # Set background color for the figure and axes
                        fig.patch.set_facecolor('#283747') # Dark background for the entire figure
                        ax1.set_facecolor('#34495E') # Darker background for each subplot
                        ax2.set_facecolor('#34495E')
                        ax3.set_facecolor('#34495E')

                        # Trend plot
                        ax1.plot(decomposition.trend, color='#5DADE2', linewidth=2) # Lighter blue line for trend
                        ax1.set_title('Trend Component', fontsize=16, color='#EBF5FB') # Bigger title
                        ax1.set_ylabel('Life Expectancy', fontsize=12, color='#EBF5FB')
                        ax1.tick_params(axis='x', colors='#EBF5FB')
                        ax1.tick_params(axis='y', colors='#EBF5FB')
                        ax1.grid(True, linestyle='--', alpha=0.5, color='#7F8C8D') # Softer grid lines
                        
                        # Seasonality plot
                        ax2.plot(decomposition.seasonal, color='#82E0AA', linewidth=1.5) # Lighter green line for seasonality
                        ax2.set_title('Seasonal Component', fontsize=16, color='#EBF5FB')
                        ax2.set_ylabel('Seasonality', fontsize=12, color='#EBF5FB')
                        ax2.tick_params(axis='x', colors='#EBF5FB')
                        ax2.tick_params(axis='y', colors='#EBF5FB')
                        ax2.grid(True, linestyle='--', alpha=0.5, color='#7F8C8D') # Softer grid lines

                        # Residuals plot
                        ax3.plot(decomposition.resid, color='#F5B7B1', linewidth=1) # Lighter red line for residuals
                        ax3.set_title('Residuals Component', fontsize=16, color='#EBF5FB')
                        ax3.set_xlabel('Year', fontsize=12, color='#EBF5FB')
                        ax3.set_ylabel('Residual', fontsize=12, color='#EBF5FB')
                        ax3.tick_params(axis='x', colors='#EBF5FB')
                        ax3.tick_params(axis='y', colors='#EBF5FB')
                        ax3.grid(True, linestyle='--', alpha=0.5, color='#7F8C8D') # Softer grid lines
                        
                        fig.suptitle(f'Time Series Decomposition for {country.title()} Life Expectancy', fontsize=20, color='#EBF5FB') # Bigger main title
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                        st.pyplot(fig)
                        plt.close(fig) # Close the figure to free memory
                        
                        st.info("""
                        **Note on Seasonal and Residual Plots:**
                        If the 'Seasonal Component' and 'Residuals Component' plots appear flat, it often means the `period` parameter in the `seasonal_decompose` function is set to `1`. For yearly data, setting `period=1` implies no within-year seasonality, thus these components will reflect this assumption. If you expect a longer-term seasonal cycle (e.g., a cycle spanning multiple years), you would need to adjust the `period` parameter to that cycle's length.
                        """)

                    else:
                        st.warning("Insufficient data for time series decomposition (at least 3 data points needed for a single country across all years).")
                else:
                    st.info("Select a single country from the sidebar to view time series decomposition across all years.")

        with tab2:  # Relationships Tab
            st.subheader("Advanced Correlation Analysis")
            # Define default columns for correlation matrix
            default_corr_cols = [
                'life_expectancy', 'gdp_per_capita', 'health_expenditure_per_capita',
                'school_enrollment_combined', 'mortality_infant', 'access_to_electricity', 'population_growth_rate' # Added population_growth_rate
            ]
            # Filter default columns to only include those present in the filtered DataFrame
            safe_default_corr_cols = [col for col in num_cols_filtered if col in default_corr_cols]
            
            # Multiselect for users to choose variables for correlation
            corr_cols = st.multiselect("Select variables for correlation", num_cols_filtered.tolist(), default=safe_default_corr_cols)

            if corr_cols and len(corr_cols) > 1 and not filtered_df[corr_cols].empty:
                corr_df = filtered_df[corr_cols].corr()
                
                # Try to cluster correlations for better visualization
                try:
                    from scipy.cluster import hierarchy
                    dist = hierarchy.distance.pdist(corr_df)
                    linkage = hierarchy.linkage(dist, method='average')
                    order = hierarchy.leaves_list(linkage)
                    clustered_corr = corr_df.iloc[order, order]
                except Exception as e:
                    # Fallback to unclustered if clustering fails (e.g., due to scipy not being available or small data)
                    clustered_corr = corr_df
                    st.warning(f"Could not perform hierarchical clustering for correlation matrix: {e}. Displaying unclustered matrix.")
                
                # Create a Plotly Heatmap for the correlation matrix
                fig_corr = go.Figure(data=go.Heatmap(
                    z=clustered_corr,
                    x=clustered_corr.columns,
                    y=clustered_corr.index,
                    colorscale='RdBu', # Red-Blue color scale for correlations (-1 to 1)
                    zmin=-1, # Minimum value for color scale
                    zmax=1, # Maximum value for color scale
                    text=np.round(clustered_corr.values, 2), # Display correlation values as text
                    texttemplate="%{text}" # Format for text display
                ))
                fig_corr.update_layout(
                    title="Clustered Correlation Matrix",
                    height=700,  # Increased height
                    xaxis_title="Features",
                    yaxis_title="Features",
                    # Professional theme for Plotly
                    template="plotly_dark", 
                    paper_bgcolor='#283747',  # Match overall background
                    plot_bgcolor='#34495E',   # Match feature box background
                    font=dict(color='#EBF5FB') # Light font color
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Highlight top positive and negative correlations
                st.subheader("Strongest Correlations")
                corr_unstacked = corr_df.unstack().reset_index()
                corr_unstacked.columns = ['Var1', 'Var2', 'Correlation']
                corr_unstacked = corr_unstacked[corr_unstacked['Var1'] != corr_unstacked['Var2']] # Exclude self-correlations
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

        with tab3:  # Geographic Analysis Tab
            st.header("Geographic Analysis")
            st.info("""
            **Mapping Note:** For accurate geographic display, ensure your dataset includes
            latitude and longitude columns, or an ISO 3-letter country code (`iso_alpha3`).
            Currently, markers are placed randomly as a placeholder.
            """)

            if 'year' in df.columns and 'country_name' in df.columns and not filtered_df.empty:
                st.subheader("Interactive World Map")
                
                # Define the aggregation dictionary for map_df
                aggregation_dict = {
                    'life_expectancy': 'mean',
                    'gdp_per_capita': 'mean',
                }

                # --- FIX: Conditionally add 'population_growth_rate' to aggregation ---
                if 'population_growth_rate' in filtered_df.columns:
                    aggregation_dict['population_growth_rate'] = 'mean'
                else:
                    st.warning("The 'population_growth_rate' column was not found. Map marker sizes will be static (default 5).")

                # Group filtered data by country and calculate means for relevant metrics
                map_df = filtered_df.groupby('country_name').agg(aggregation_dict).reset_index()
                
                # Create a Folium map centered globally
                m = folium.Map(location=[20, 0], zoom_start=2, tiles='cartodbpositron')
                
                try:
                    # Add markers for each country based on aggregated data
                    for idx, row in map_df.iterrows():
                        popup_text = f"""
                        <b>{row['country_name']}</b><br>
                        Life Expectancy: {row['life_expectancy']:.1f} yrs<br>
                        GDP per capita: ${row['gdp_per_capita']:,.0f}
                        """
                        marker_radius = 5 # Default minimum radius for markers

                        # --- FIX: Use population_growth_rate for marker radius ---
                        if 'population_growth_rate' in row:
                            popup_text += f"<br>Population Growth Rate: {row['population_growth_rate']:.2f}%"
                            # Scale the growth rate for radius. Abs is used to handle negative growth rates.
                            # Multiplying by a factor (e.g., 10) and ensuring a minimum size (e.g., 2)
                            marker_radius = max(2, abs(row['population_growth_rate']) * 10)
                        
                        # --- IMPORTANT: Placeholder for actual country coordinates ---
                        # Replace these random coordinates with actual latitude and longitude
                        # for each country from your dataset for accurate mapping.
                        # Example: location=[row['latitude_column'], row['longitude_column']]
                        folium.CircleMarker(
                            location=[np.random.uniform(-60, 70), np.random.uniform(-180, 180)], # Placeholder: REPLACE with actual country lat/lon
                            radius=marker_radius,
                            popup=popup_text,
                            color='#3186cc', # Blue color for markers
                            fill=True,
                            fill_color='#3186cc'
                        ).add_to(m)
                    
                    # Add layer control to the map
                    folium.LayerControl().add_to(m)
                    
                    # Display the map within a custom styled container
                    with st.container():
                        st.markdown('<div class="folium-map">', unsafe_allow_html=True)
                        # Use folium_static to render the Folium map in Streamlit
                        folium_static(m, width=1200, height=600)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error creating map: {e}")
                    st.info("Using Plotly as fallback for mapping due to an error.")
                    # Fallback to Plotly choropleth map
                    # Ensure 'population_growth_rate' is also added to hover_data if needed for Plotly
                    hover_data_cols = ['gdp_per_capita']
                    if 'population_growth_rate' in map_df.columns:
                        hover_data_cols.append('population_growth_rate')

                    fig_map = px.choropleth(
                        map_df,
                        locations="country_name",
                        locationmode="country names",
                        color='life_expectancy',
                        hover_name="country_name",
                        hover_data=hover_data_cols, # Dynamically add population_growth_rate to hover info
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Global Life Expectancy Distribution"
                    )
                    fig_map.update_layout(
                        height=700, # Increased height
                        template="plotly_dark",
                        paper_bgcolor='#283747',
                        plot_bgcolor='#34495E',
                        font=dict(color='#EBF5FB')
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("Geographic analysis requires 'year' and 'country_name' columns and data to be present in the dataset, and selections to be made in the sidebar.")

        with tab4:  # Deep Analysis Tab
            st.subheader("Feature Impact Analysis")
            
            if 'life_expectancy' in num_cols_filtered:
                # Dropdown to select a feature to analyze against life expectancy
                feature = st.selectbox("Select Feature to Analyze", 
                                    [col for col in num_cols_filtered if col != 'life_expectancy'])
                
                if feature:
                    # Calculate Pearson correlation between the selected feature and life expectancy
                    correlation = filtered_df['life_expectancy'].corr(filtered_df[feature])
                    
                    # Create bins for categorical analysis if the feature has many unique values
                    if filtered_df[feature].nunique() > 10:
                        try:
                            filtered_df['feature_bin'] = pd.qcut(filtered_df[feature], 5, duplicates='drop')
                            # FIX: Convert Interval objects to strings for Plotly compatibility
                            filtered_df['feature_bin'] = filtered_df['feature_bin'].astype(str)
                        except Exception as e:
                            st.warning(f"Could not create bins for '{feature}': {e}. Displaying scatter plot without binned analysis.")
                            filtered_df['feature_bin'] = filtered_df[feature] # Fallback to original values
                    else:
                        filtered_df['feature_bin'] = filtered_df[feature]
                    
                    # Plot scatter relationship with OLS trendline
                    fig = px.scatter(
                        filtered_df,
                        x=feature,
                        y='life_expectancy',
                        # Color by country if selected, otherwise by region if available
                        color='country_name' if selected_countries else 'region' if 'region' in filtered_df else None,
                        trendline='ols', # Ordinary Least Squares regression line
                        title=f"Life Expectancy vs {feature.replace('_', ' ').title()} (Correlation: {correlation:.2f})",
                        height=600 # Increased height
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='#283747',
                        plot_bgcolor='#34495E',
                        font=dict(color='#EBF5FB')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot for binned categorical analysis, if successful
                    if 'feature_bin' in filtered_df and filtered_df['feature_bin'].nunique() > 1:
                        fig_box = px.box(
                            filtered_df,
                            x='feature_bin',
                            y='life_expectancy',
                            title=f"Life Expectancy Distribution by {feature.replace('_', ' ').title()} Groups",
                            height=600 # Increased height
                        )
                        fig_box.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='#283747',
                            plot_bgcolor='#34495E',
                            font=dict(color='#EBF5FB')
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    elif 'feature_bin' in filtered_df:
                        st.info(f"Not enough unique bins created for '{feature}' to show box plot.")

            st.subheader("Predictive Modeling: Factors Affecting Life Expectancy")
            st.markdown("Build a simple predictive model to understand feature importance.")

            # Define features (X) and target (y) for the model
            target = 'life_expectancy'
            
            # Exclude non-numeric columns, identifiers, and the target itself from features
            potential_features = [col for col in num_cols_filtered if col != target and col not in ['year']]
            
            if not potential_features:
                st.warning("No suitable numerical features found for predictive modeling after filtering.")
            elif filtered_df.empty:
                st.info("Filtered dataset is empty, cannot perform predictive modeling.")
            else:
                # Allow user to select features for the model
                selected_model_features = st.multiselect(
                    "Select features for the predictive model",
                    potential_features,
                    default=[pf for pf in ['gdp_per_capita', 'health_expenditure_per_capita', 'school_enrollment_combined', 'mortality_infant', 'access_to_electricity', 'population_growth_rate'] if pf in potential_features]
                )

                if selected_model_features:
                    X = filtered_df[selected_model_features]
                    y = filtered_df[target]

                    # Drop rows with NaN values in X or y if any, as models don't handle them directly
                    # (though median imputation was done earlier, this is a safety check for complex cases)
                    clean_df = filtered_df[[target] + selected_model_features].dropna()
                    if clean_df.empty:
                        st.warning("No complete data rows available for selected features and target after dropping NaNs. Please check your data.")
                    else:
                        X_clean = clean_df[selected_model_features]
                        y_clean = clean_df[target]

                        if len(X_clean) > 1 and len(np.unique(y_clean)) > 1: # Need at least 2 samples and >1 unique target value
                            # Split data into training and testing sets
                            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

                            # Standardize features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)

                            # Train a RandomForestRegressor model
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X_train_scaled, y_train)

                            # Make predictions and evaluate the model
                            y_pred = model.predict(X_test_scaled)
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)

                            # Store results in session state
                            st.session_state['r2_score'] = r2
                            st.session_state['feature_importance'] = pd.DataFrame({
                                'Feature': selected_model_features,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)

                            st.write(f"**Model R-squared Score:** `{st.session_state['r2_score']:.2f}`")
                            st.write(f"**Mean Squared Error:** `{mse:.2f}`")

                            st.subheader("Feature Importance")
                            st.dataframe(st.session_state['feature_importance'])

                            # Plot feature importance
                            fig_feature_imp = px.bar(st.session_state['feature_importance'], 
                                                    x='Importance', y='Feature', orientation='h',
                                                    title='Feature Importance in Life Expectancy Prediction',
                                                    height=600 # Increased height
                            )
                            fig_feature_imp.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                template="plotly_dark",
                                paper_bgcolor='#283747',
                                plot_bgcolor='#34495E',
                                font=dict(color='#EBF5FB')
                            )
                            st.plotly_chart(fig_feature_imp, use_container_width=True)
                        else:
                            st.warning("Not enough data or variety in target for predictive modeling with selected features.")
                else:
                    st.info("Select features to build the predictive model.")
            
            # --- NEW: Parallel Coordinates Plot ---
            st.subheader("Multi-Dimensional Data Relationships (Parallel Coordinates)")
            st.markdown("Visualize patterns and clusters across multiple numerical features.")

            # Exclude 'year' and 'feature_bin' from options as they are typically not suitable
            parallel_coord_cols_options = [col for col in num_cols_filtered if col not in ['year', 'feature_bin']]

            # Add categorical columns as options for the color dimension
            categorical_color_options = []
            if 'country_name' in filtered_df.columns and filtered_df['country_name'].nunique() > 1 and len(filtered_df['country_name'].unique()) <= 50:
                categorical_color_options.append('country_name')
            if 'region' in filtered_df.columns and filtered_df['region'].nunique() > 1 and len(filtered_df['region'].unique()) <= 20:
                categorical_color_options.append('region')

            all_color_options = ['None'] + sorted(parallel_coord_cols_options) + sorted(categorical_color_options)
            
            # User selects variables for the dimensions of the parallel coordinates plot
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

            # User selects the color dimension for the parallel coordinates plot
            color_choice = st.selectbox(
                "Color lines by",
                all_color_options,
                index=all_color_options.index('life_expectancy') if 'life_expectancy' in all_color_options else 0 # Default to 'life_expectancy' or 'None'
            )

            if selected_parallel_cols and len(selected_parallel_cols) > 1:
                color_dimension_for_plot = None
                plot_df_for_pc = filtered_df[selected_parallel_cols].copy() # Start with only the dimensions

                if color_choice != 'None':
                    color_dimension_for_plot = color_choice
                    if color_dimension_for_plot not in plot_df_for_pc.columns:
                        plot_df_for_pc[color_dimension_for_plot] = filtered_df[color_dimension_for_plot]

                    # Convert categorical color dimension to numerical codes for Plotly
                    # This ensures Plotly can plot it, and we use hover_data to show original labels
                    if plot_df_for_pc[color_dimension_for_plot].dtype == 'object' or pd.api.types.is_categorical_dtype(plot_df_for_pc[color_dimension_for_plot]):
                        # Create a temporary numerical column for coloring
                        plot_df_for_pc[f'{color_dimension_for_plot}_codes'] = plot_df_for_pc[color_dimension_for_plot].astype('category').cat.codes
                        
                        par_coords_kwargs = {
                            "data_frame": plot_df_for_pc.dropna(), # Drop NaNs before plotting
                            "dimensions": selected_parallel_cols,
                            "title": "Parallel Coordinates Analysis",
                            "color": f'{color_dimension_for_plot}_codes',
                            "color_continuous_scale": px.colors.qualitative.Plotly, # Use a qualitative scale for categories
                            "custom_data": [color_dimension_for_plot],
                            "hover_data": {f'{color_dimension_for_plot}_codes': False, color_dimension_for_plot: True},
                            "height": 600 # Increased height
                        }
                    else:
                        # For numerical color dimension
                        par_coords_kwargs = {
                            "data_frame": plot_df_for_pc.dropna(), # Drop NaNs before plotting
                            "dimensions": selected_parallel_cols,
                            "title": "Parallel Coordinates Analysis",
                            "color": color_dimension_for_plot,
                            "color_continuous_scale": px.colors.sequential.Viridis,
                            "height": 600 # Increased height
                        }
                else:
                    # No color dimension selected
                    par_coords_kwargs = {
                        "data_frame": plot_df_for_pc.dropna(), # Drop NaNs before plotting
                        "dimensions": selected_parallel_cols,
                        "title": "Parallel Coordinates Analysis",
                        "height": 600 # Increased height
                    }

                if not plot_df_for_pc.dropna().empty:
                    fig_par_coords = px.parallel_coordinates(**par_coords_kwargs)
                    fig_par_coords.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='#283747',
                        plot_bgcolor='#34495E',
                        font=dict(color='#EBF5FB')
                    )
                    st.plotly_chart(fig_par_coords, use_container_width=True)
                else:
                    st.info("No complete data rows for selected variables and color dimension to plot Parallel Coordinates after dropping NaNs.")
            else:
                st.info("Please select at least two numerical variables for the Parallel Coordinates plot dimensions.")


        with tab5:  # Recommendations Tab
            st.header("Data-Driven Recommendations")
            
            if not filtered_df.empty and 'life_expectancy' in filtered_df.columns:
                st.subheader("Performance Benchmarks")
                
                # Calculate improvement metrics (Life Expectancy Growth)
                if 'year' in filtered_df.columns and filtered_df['year'].nunique() > 1:
                    le_diff = filtered_df.groupby('country_name')['life_expectancy'].agg(
                        le_start=('min'), # Life expectancy at the earliest year
                        le_end=('max'),   # Life Expectancy at the latest year
                        le_growth=lambda x: x.max() - x.min() # Difference between max and min
                    ).reset_index()
                    
                    # Sort and display top/bottom performers in life expectancy growth
                    top_growth = le_diff.sort_values('le_growth', ascending=False).head(3)
                    bottom_growth = le_diff.sort_values('le_growth', ascending=True).head(3)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Top Performers (Life Expectancy Growth)**")
                        for _, row in top_growth.iterrows():
                            st.markdown(f"- **{row['country_name'].title()}**: +{row['le_growth']:.1f} years")
                    
                    with col2:
                        st.markdown("**Countries Needing Improvement**")
                        # Fix: Changed 'bottom_corrs' to 'bottom_growth'
                        for _, row in bottom_growth.iterrows():
                            st.markdown(f"- **{row['country_name'].title()}**: {row['le_growth']:.1f} years")
                else:
                    st.info("Insufficient years data to calculate life expectancy growth. Please select multiple years.")
                
                # Gap analysis for country comparison
                if len(selected_countries) > 1:
                    st.subheader("Country Comparison Analysis")
                    # Aggregate key metrics for selected countries
                    comparison_df = filtered_df.groupby('country_name').agg({
                        'life_expectancy': 'mean',
                        'gdp_per_capita': 'mean',
                        'health_expenditure_per_capita': 'mean',
                        'school_enrollment_combined': 'mean'
                    }).reset_index()
                    
                    # Allow user to select a baseline country for comparison
                    base_country = st.selectbox("Select Baseline Country", comparison_df['country_name'].tolist())
                    
                    if base_country:
                        # Get baseline values
                        base_values = comparison_df[comparison_df['country_name'] == base_country].iloc[0]
                        
                        gap_analysis = []
                        # Calculate gaps for other countries relative to the baseline
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
                        
                        st.write("**Performance Gaps Compared to Baseline:**")
                        # Display performance gaps with a color gradient
                        st.dataframe(gap_df.style.background_gradient(cmap='RdYlGn', subset=['life_expectancy_gap', 'gdp_gap_percent', 'health_exp_gap_percent', 'education_gap'], axis=0))
                    else:
                        st.info("Select a baseline country to perform gap analysis.")
                else:
                    st.info("Select more than one country in the sidebar to perform country comparison analysis.")
            else:
                st.info("Please ensure data is loaded and filtered to see recommendations.")

    else: # filtered_df is empty, show general message
        if uploaded_file is None:
            st.info("Upload a dataset to unlock all analysis features.")
        elif df is not None and (not selected_years or not selected_countries) and not df.empty: # More specific for "no data after filters"
            st.info("Please make selections in the sidebar filters (Year(s) and Countries) to view the dashboard content. No data is currently selected for display.")
        elif df is not None and df.empty: # If file was uploaded but it was empty
            st.info("Uploaded data is empty. Please upload a valid dataset.")
