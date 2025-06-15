import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

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
    page_icon="📊"
)

# --- Custom CSS for styling ---
st.markdown("""
<style>
    /* Dark theme styling */
    body, .main {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    .stButton>button {
        background-color: #1976D2;
        color: white;
        border-radius: 5px;
    }
    .intro-box {
        background-color:#e3f2fd;
        padding:20px;
        border-radius:10px;
        margin-bottom:30px;
    }
    .feature-box {
        background-color: #1e3d59;
        color: #ecf0f1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .highlight {
        background-color: #34495e;
        border-left: 4px solid #4a90e2;
        padding: 10px;
        border-radius: 0 5px 5px 0;
        color: #ecf0f1;
    }
    .call-to-action-box {
        background-color: #4a698a;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        color: #ecf0f1;
    }
    /* Folium map container */
    .folium-map {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Introduction ---
st.title(f"🌏 Life Expectancy Dashboard: {st.session_state.selected_region_display}")
st.markdown("""
<div class="intro-box">
    <h3>Comprehensive Analysis of Factors Influencing Life Expectancy</h3>
    <p>This dashboard explores key socioeconomic and health factors affecting life expectancy in various regions.
    Upload your dataset to begin the analysis.</p>
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
            st.success("✅ Data loaded successfully!")

            cols = st.columns(3)
            cols[0].metric("Total Records", df.shape[0])
            cols[1].metric("Features", df.shape[1])
            cols[2].metric("Countries", df['country_name'].nunique())
            st.session_state['initial_load'] = True 
        else:
            st.info("Uploaded file could not be processed. Please check the file format or its content.")
else:
    if 'initial_load' not in st.session_state or not st.session_state.initial_load:
        st.info("Please upload a dataset to begin analysis.")

filtered_df = None
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

    # Enhanced Year Range Filter
    years = sorted(df['year'].unique())
    if len(years) > 0:
        min_year, max_year = int(min(years)), int(max(years))
        selected_years = st.sidebar.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        st.sidebar.warning("No year data available")
        selected_years = (2000, 2020)  # Default range

    temp_df = df.copy()
    temp_df = temp_df[(temp_df['year'] >= selected_years[0]) & (temp_df['year'] <= selected_years[1])]

    if 'region' in temp_df.columns:
        regions = sorted(temp_df['region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Select Region", ['All'] + regions)
        st.session_state.selected_region_display = selected_region
        if selected_region != 'All':
            temp_df = temp_df[temp_df['region'] == selected_region]
    else:
        st.sidebar.info("Region column not found for region filtering.")
        st.session_state.selected_region_display = "Global (Region column missing)"

    all_countries = sorted(temp_df['country_name'].unique().tolist())
    default_countries_selection = []
    if 'japan' in all_countries and 'papua_new_guinea' in all_countries:
        default_countries_selection = ['japan', 'papua_new_guinea']
    elif len(all_countries) >= 2:
        default_countries_selection = all_countries[:2]
    elif len(all_countries) == 1:
        default_countries_selection = all_countries

    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        all_countries,
        default=default_countries_selection
    )

    if selected_countries and 'country_name' in temp_df.columns:
        filtered_df = temp_df[temp_df['country_name'].isin(selected_countries)].copy()
    else:
        filtered_df = temp_df.copy()

    num_cols_filtered = filtered_df.select_dtypes(include=['number']).columns

    if 'life_expectancy' not in num_cols_filtered:
        st.error("The 'life_expectancy' column is missing in your filtered dataset.")
        st.stop()

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
        if 'life_expectancy' in num_cols_filtered:
            cols[0].metric("Avg Life Expectancy", f"{filtered_df['life_expectancy'].mean():.1f} yrs")
        if 'gdp_per_capita' in num_cols_filtered:
            cols[1].metric("Avg GDP per Capita", f"${filtered_df['gdp_per_capita'].mean():,.0f}")
        if 'health_expenditure_per_capita' in num_cols_filtered:
            cols[2].metric("Healthcare Expenditure", f"${filtered_df['health_expenditure_per_capita'].mean():,.0f}")
        if 'school_enrollment_combined' in num_cols_filtered:
            cols[3].metric("School Enrollment", f"{filtered_df['school_enrollment_combined'].mean():.1f}%")
        
        # Summary Statistics
        with st.expander("📋 Summary Statistics"):
            st.write(filtered_df.describe())

        # Time Series Decomposition
        if 'year' in filtered_df.columns and 'life_expectancy' in filtered_df.columns:
            st.subheader("Life Expectancy Trend Analysis")
            if len(filtered_df['country_name'].unique()) == 1:
                country = filtered_df['country_name'].iloc[0]
                country_df = filtered_df.set_index('year')['life_expectancy']
                
                if len(country_df) > 2:
                    decomposition = seasonal_decompose(country_df, model='additive', period=1)
                    
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
                    decomposition.trend.plot(ax=ax1, title='Trend')
                    decomposition.seasonal.plot(ax=ax2, title='Seasonality')
                    decomposition.resid.plot(ax=ax3, title='Residuals')
                    fig.suptitle(f'Time Series Decomposition for {country}', fontsize=16)
                    st.pyplot(fig)
                else:
                    st.warning("Insufficient data for time series decomposition")
            else:
                st.info("Select a single country to view time series decomposition")

        # Distribution and Trends Plots (same as before)

    with tab2:  # Relationships Tab
        # Enhanced Correlation Matrix with clustering
        st.subheader("Advanced Correlation Analysis")
        default_corr_cols = [
            'life_expectancy', 'gdp_per_capita', 'health_expenditure_per_capita',
            'school_enrollment_combined', 'mortality_infant', 'access_to_electricity'
        ]
        safe_default_corr_cols = [col for col in default_corr_cols if col in num_cols_filtered]
        corr_cols = st.multiselect("Select variables for correlation", num_cols_filtered.tolist(), default=safe_default_corr_cols)

        if corr_cols and len(corr_cols) > 1 and not filtered_df[corr_cols].empty:
            corr_df = filtered_df[corr_cols].corr()
            
            # Cluster correlations
            try:
                from scipy.cluster import hierarchy
                dist = hierarchy.distance.pdist(corr_df)
                linkage = hierarchy.linkage(dist, method='average')
                order = hierarchy.leaves_list(linkage)
                clustered_corr = corr_df.iloc[order, order]
            except:
                clustered_corr = corr_df
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=clustered_corr,
                x=clustered_corr.columns,
                y=clustered_corr.index,
                colorscale='RdBu', 
                zmin=-1, 
                zmax=1, 
                text=np.round(clustered_corr.values, 2), 
                texttemplate="%{text}"
            ))
            fig_corr.update_layout(title="Clustered Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Highlight top correlations
            st.subheader("Strongest Correlations")
            corr_unstacked = corr_df.unstack().reset_index()
            corr_unstacked.columns = ['Var1', 'Var2', 'Correlation']
            corr_unstacked = corr_unstacked[corr_unstacked['Var1'] != corr_unstacked['Var2']]
            top_corrs = corr_unstacked.sort_values('Correlation', ascending=False).head(5)
            bottom_corrs = corr_unstacked.sort_values('Correlation', ascending=True).head(5)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top Positive Correlations:**")
                for _, row in top_corrs.iterrows():
                    st.markdown(f"- **{row['Var1']}** & **{row['Var2']}**: {row['Correlation']:.2f}")
            
            with col2:
                st.write("**Top Negative Correlations:**")
                for _, row in bottom_corrs.iterrows():
                    st.markdown(f"- **{row['Var1']}** & **{row['Var2']}**: {row['Correlation']:.2f}")
        else:
            st.info("Please select at least two numerical variables to view the correlation matrix.")

    with tab3:
        st.header("Geographic Analysis")
        st.info("""
        **Mapping Note:** Country names must match standard naming conventions.
        Using a 3-letter ISO code column (e.g., 'iso_alpha3') can improve reliability.
        """)

        if 'year' in df.columns and 'country_name' in df.columns and not df.empty:
            # Enhanced map with Folium
            st.subheader("Interactive World Map")
            
            # Calculate average values for the selected year range
            map_df = filtered_df.groupby('country_name').agg({
                'life_expectancy': 'mean',
                'gdp_per_capita': 'mean',
                'population': 'mean'
            }).reset_index()
            
            # Create Folium map
            m = folium.Map(location=[20, 0], zoom_start=2, tiles='cartodbpositron')
            
            # Add choropleth layer if we have geo data
            try:
                from streamlit_folium import folium_static
                
                # Add markers for each country
                for idx, row in map_df.iterrows():
                    popup_text = f"""
                    <b>{row['country_name']}</b><br>
                    Life Expectancy: {row['life_expectancy']:.1f} yrs<br>
                    GDP per capita: ${row['gdp_per_capita']:,.0f}
                    """
                    folium.CircleMarker(
                        location=[np.random.uniform(-60, 70), np.random.uniform(-180, 180)],
                        radius=row['population']/100000000 if 'population' in row else 5,
                        popup=popup_text,
                        color='#3186cc',
                        fill=True,
                        fill_color='#3186cc'
                    ).add_to(m)
                
                # Add layer control
                folium.LayerControl().add_to(m)
                
                # Display the map
                with st.container():
                    st.markdown('<div class="folium-map">', unsafe_allow_html=True)
                    folium_static(m, width=1200, height=600)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error creating map: {e}")
                st.info("Using Plotly as fallback")
                fig_map = px.choropleth(
                    map_df,
                    locations="country_name",
                    locationmode="country names",
                    color='life_expectancy',
                    hover_name="country_name",
                    hover_data=['gdp_per_capita', 'population'],
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Global Life Expectancy Distribution"
                )
                st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Geographic analysis requires 'year' and 'country_name' columns and data to be present.")

    with tab4:  # Deep Analysis Tab
        # Enhanced feature analysis
        st.subheader("Feature Impact Analysis")
        
        if 'life_expectancy' in num_cols_filtered:
            # Feature vs Life Expectancy
            feature = st.selectbox("Select Feature to Analyze", 
                                 [col for col in num_cols_filtered if col != 'life_expectancy'])
            
            if feature:
                # Calculate correlation
                correlation = filtered_df['life_expectancy'].corr(filtered_df[feature])
                
                # Create bins for categorical analysis
                if filtered_df[feature].nunique() > 10:
                    filtered_df['feature_bin'] = pd.qcut(filtered_df[feature], 5, duplicates='drop')
                else:
                    filtered_df['feature_bin'] = filtered_df[feature]
                
                # Plot relationship
                fig = px.scatter(
                    filtered_df,
                    x=feature,
                    y='life_expectancy',
                    color='country_name' if selected_countries else 'region' if 'region' in filtered_df else None,
                    trendline='ols',
                    title=f"Life Expectancy vs {feature.replace('_', ' ').title()} (Correlation: {correlation:.2f})"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot for categorical analysis
                if 'feature_bin' in filtered_df:
                    fig_box = px.box(
                        filtered_df,
                        x='feature_bin',
                        y='life_expectancy',
                        title=f"Life Expectancy Distribution by {feature.replace('_', ' ').title()} Groups"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
        
        # Predictive Modeling (same as before)

    with tab5:  
        st.header("Data-Driven Recommendations")
        
        # Enhanced insights with actionable metrics
        if not filtered_df.empty and 'life_expectancy' in filtered_df.columns:
            st.subheader("Performance Benchmarks")
            
            # Calculate improvement metrics
            if 'year' in filtered_df.columns:
                years = filtered_df['year'].unique()
                if len(years) > 1:
                    min_year = min(years)
                    max_year = max(years)
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
                            st.markdown(f"- **{row['country_name']}**: +{row['le_growth']:.1f} years")
                    
                    with col2:
                        st.markdown("**Countries Needing Improvement**")
                        for _, row in bottom_growth.iterrows():
                            st.markdown(f"- **{row['country_name']}**: {row['le_growth']:.1f} years")
            
            # Gap analysis
            if len(selected_countries) > 1:
                st.subheader("Country Comparison Analysis")
                comparison_df = filtered_df.groupby('country_name').agg({
                    'life_expectancy': 'mean',
                    'gdp_per_capita': 'mean',
                    'health_expenditure_per_capita': 'mean',
                    'school_enrollment_combined': 'mean'
                }).reset_index()
                
                # Find performance gaps
                base_country = st.selectbox("Select Baseline Country", comparison_df['country_name'])
                if base_country:
                    base_values = comparison_df[comparison_df['country_name'] == base_country].iloc[0]
                    
                    gap_analysis = []
                    for _, row in comparison_df.iterrows():
                        if row['country_name'] != base_country:
                            gap = {
                                'country': row['country_name'],
                                'life_expectancy_gap': row['life_expectancy'] - base_values['life_expectancy'],
                                'gdp_gap_percent': (row['gdp_per_capita'] - base_values['gdp_per_capita']) / base_values['gdp_per_capita'] * 100,
                                'health_exp_gap_percent': (row['health_expenditure_per_capita'] - base_values['health_expenditure_per_capita']) / base_values['health_expenditure_per_capita'] * 100,
                                'education_gap': row['school_enrollment_combined'] - base_values['school_enrollment_combined']
                            }
                            gap_analysis.append(gap)
                    
                    gap_df = pd.DataFrame(gap_analysis)
                    
                    st.write("**Performance Gaps Compared to Baseline:**")
                    st.dataframe(gap_df.style.background_gradient(cmap='RdYlGn', axis=0))
        
        # Recommendations (same as before)

else:
    st.info("Upload a dataset to unlock all analysis features")
