import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- Session State Initialization ---
# Initialize session state variables at the beginning to ensure consistency
if 'selected_region_display' not in st.session_state:
    st.session_state.selected_region_display = "Global & Region-Specific"
if 'initial_load' not in st.session_state:
    st.session_state.initial_load = False
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = pd.DataFrame() # Initialize as empty DataFrame
if 'r2_score' not in st.session_state:
    st.session_state.r2_score = None

# --- Set page configuration ---
st.set_page_config(
    layout="wide",
    page_title="Life Expectancy Analysis Dashboard",
    page_icon=" 📊 "
)

# --- Custom CSS for styling ---
# Enhances the dashboard's visual appeal with a dark theme and custom elements.
st.markdown("""
<style>
    /* General body and main container styling for dark theme */
    body, .main {
        background-color: #2c3e50; /* Dark charcoal/navy blue for main background */
        color: #ecf0f1; /* Light grey for all general text */
    }
    .stButton>button {
        background-color: #1976D2; /* A shade of blue for buttons */
        color: white;
        border-radius: 5px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .header {
        color: #ecf0f1; /* Light text for main headers on dark background */
        padding-bottom: 1rem;
    }
    /* Styling for the introductory box (still light blue as per original request) */
    .intro-box {
        background-color:#e3f2fd;
        padding:20px;
        border-radius:10px;
        margin-bottom:30px;
    }
    .intro-box h3 {
        color:#1e3d59 !important; /* Dark text for heading in intro box */
    }
    .intro-box p {
        color:#1e3d59 !important; /* Dark text for paragraph in intro box */
    }
    /* Feature box styling (navy blue background with light text) */
    .feature-box {
        background-color: #1e3d59; /* Navy blue background */
        color: #ecf0f1; /* Light grey for general text inside */
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-box h4 { /* Headers within feature boxes */
        color: white; /* White for high contrast */
    }
    .feature-box b { /* Bold text within feature boxes */
        color: white; /* White for high contrast */
    }
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); /* Retaining original vibrant gradient */
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    /* Highlight box styling */
    .highlight {
        background-color: #34495e; /* Darker blue/grey, distinct from main background */
        border-left: 4px solid #4a90e2; /* Elegant blue border */
        padding: 10px;
        border-radius: 0 5px 5px 0;
        color: #ecf0f1; /* Light text */
    }
    .highlight h4 {
        color: #ecf0f1; /* Ensure header color matches text */
    }
    /* Call-to-action box styling */
    .call-to-action-box {
        background-color: #4a698a; /* Medium-dark blue, distinct from main/feature */
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        color: #ecf0f1; /* Light text */
    }
    .call-to-action-box h4 {
        color: #ecf0f1; /* Ensure header color matches box text color */
    }
    .call-to-action-box b {
        color: white; /* Ensure bold text is white for emphasis */
    }
</style>
""", unsafe_allow_html=True)

# --- Introduction ---
st.title(f" 🌏  Life Expectancy Dashboard: {st.session_state.selected_region_display}")
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
    """
    Loads data from a CSV or XLSX file and cleans column names.
    Handles potential errors during file loading.
    Uses st.cache_data to cache the loaded data, improving performance on reruns.
    """
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
            # --- Data Loading & Validation ---
            REQUIRED_COLUMNS = ['life_expectancy', 'country_name', 'year']
            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                st.error(f"Missing required columns for analysis: {', '.join(missing)}. Please upload a dataset containing these columns.")
                st.stop() 
            st.success(" ✅  Data loaded successfully!")

            # Display key metrics from the loaded dataset.
            cols = st.columns(3)
            cols[0].metric("Total Records", df.shape[0])
            cols[1].metric("Features", df.shape[1])
            cols[2].metric("Countries", df['country_name'].nunique())
            st.session_state['initial_load'] = True 
        else:
            st.info("Uploaded file could not be processed. Please check the file format or its content.")
else:
    # Display message to upload data if not already loaded.
    if 'initial_load' not in st.session_state or not st.session_state.initial_load:
        st.info("Please upload a dataset to begin analysis.")

# Global dataframe for filtering.
filtered_df = None
selected_countries = []

if df is not None:
    # --- Data Processing ---
    if 'life_expectancy' in df.columns:
        df['life_expectancy'] = pd.to_numeric(df['life_expectancy'], errors='coerce')

    # Impute missing numerical values with the median of their respective columns.
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop rows where 'country_name' is missing, as it's crucial for country-specific analysis.
    df.dropna(subset=['country_name'], inplace=True)

    # --- Sidebar Filters ---
    st.sidebar.header("Data Filters")

    # Use a temporary dataframe for applying filters sequentially to avoid SettingWithCopyWarning.
    temp_df = df.copy()

    # Year selection slider.
    years = sorted(temp_df['year'].unique())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    temp_df = temp_df[(temp_df['year'] >= selected_years[0]) & (temp_df['year'] <= selected_years[1])]

    # Region filter, displayed only if 'region' column exists.
    if 'region' in temp_df.columns:
        regions = sorted(temp_df['region'].unique().tolist())
        selected_region = st.sidebar.selectbox("Select Region", ['All'] + regions)
        st.session_state.selected_region_display = selected_region # Update display in page title.
        if selected_region != 'All':
            temp_df = temp_df[temp_df['region'] == selected_region]
    else:
        st.sidebar.info("Region column not found for region filtering.")
        st.session_state.selected_region_display = "Global (Region column missing)" # Update display accordingly.

    # Country selection multiselect.
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

    # Apply country filter or use the year/region filtered dataframe if no countries are selected.
    if selected_countries and 'country_name' in temp_df.columns:
        filtered_df = temp_df[temp_df['country_name'].isin(selected_countries)].copy()
    else:
        filtered_df = temp_df.copy() # Use a copy to prevent chained assignment warnings.

    # Re-evaluate numerical columns based on the filtered dataset.
    num_cols_filtered = filtered_df.select_dtypes(include=['number']).columns

    # Crucial check: Ensure 'life_expectancy' is still present after filtering.
    if 'life_expectancy' not in num_cols_filtered:
        st.error("The 'life_expectancy' column (or numerical version) is missing or contains no data in your filtered dataset. Please adjust your filters or uploaded file.")
        st.stop() # Stop execution if the primary target variable is absent.

    # --- Tabs for Analysis Sections ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " 📊  Overview",
        " 📈  Relationships",
        " 🗺️  Geographic",
        " 🔍  Deep Analysis",
        " 💡  Recommendations"
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

        st.subheader("Distribution Analysis")
        default_dist_idx = num_cols_filtered.get_loc('life_expectancy') if 'life_expectancy' in num_cols_filtered else 0
        dist_col = st.selectbox("Select Variable for Distribution", num_cols_filtered.tolist(), index=default_dist_idx)

        col1, col2 = st.columns(2)
        with col1:
            # Histogram for distribution.
            if not filtered_df.empty and dist_col in filtered_df.columns:
                fig_hist = px.histogram(
                    filtered_df,
                    x=dist_col,
                    nbins=30,
                    title=f'Distribution of {dist_col.replace("_", " ").title()}',
                    color='country_name' if selected_countries and 'country_name' in filtered_df.columns else None,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning(f"No data available for {dist_col} or filtered dataset is empty for histogram.")
        with col2:
            # Box plot for distribution by country.
            if not filtered_df.empty and dist_col in filtered_df.columns:
                fig_box = px.box(
                    filtered_df,
                    y=dist_col,
                    x='country_name' if selected_countries and 'country_name' in filtered_df.columns else None,
                    title=f'Distribution by Country',
                    color='country_name' if selected_countries and 'country_name' in filtered_df.columns else None,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning(f"No data available for {dist_col} or filtered dataset is empty for box plot.")

        st.subheader("Time Trends")
        # Line chart to show trends over time.
        if 'year' in filtered_df.columns and not filtered_df.empty:
            default_trend_idx = num_cols_filtered.get_loc('life_expectancy') if 'life_expectancy' in num_cols_filtered else 0
            trend_col = st.selectbox("Select Metric to Track Over Time", num_cols_filtered.tolist(), index=default_trend_idx)
            fig_trend = px.line(
                filtered_df,
                x='year',
                y=trend_col,
                # Color by country if selected, else by region if available.
                color='country_name' if selected_countries and 'country_name' in filtered_df.columns else ('region' if 'region' in filtered_df.columns else None),
                title=f'{trend_col.replace("_", " ").title()} Over Time',
                markers=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Time trends cannot be displayed: 'year' column is missing or filtered dataset is empty.")

    with tab2:  # Relationships Tab
        st.header("Variable Relationships")
        col1, col2 = st.columns(2)
        # Set default X and Y variables for scatter plot.
        default_x_idx = num_cols_filtered.get_loc('gdp_per_capita') if 'gdp_per_capita' in num_cols_filtered else 0
        default_y_idx = num_cols_filtered.get_loc('life_expectancy') if 'life_expectancy' in num_cols_filtered else 0

        with col1:
            x_axis = st.selectbox("X-axis Variable", num_cols_filtered.tolist(), index=default_x_idx)
        with col2:
            y_axis = st.selectbox("Y-axis Variable", num_cols_filtered.tolist(), index=default_y_idx)

        # Scatter plot with OLS trendline.
        if not filtered_df.empty and x_axis in filtered_df.columns and y_axis in filtered_df.columns:
            fig_scatter = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                color='country_name' if selected_countries and 'country_name' in filtered_df.columns else ('region' if 'region' in filtered_df.columns else None),
                size='population' if 'population' in filtered_df.columns else None, 
                hover_name='country_name' if 'country_name' in filtered_df.columns else None,
                trendline='ols',
                title=f'{y_axis.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}',
                labels={x_axis: x_axis.replace('_', ' ').title(),
                        y_axis: y_axis.replace('_', ' ').title()}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No data available for selected variables or filtered dataset is empty for scatter plot.")

        st.subheader("Correlation Matrix")
        default_corr_cols = [
            'life_expectancy', 'gdp_per_capita', 'health_expenditure_per_capita',
            'school_enrollment_combined', 'mortality_infant', 'access_to_electricity'
        ]
        safe_default_corr_cols = [col for col in default_corr_cols if col in num_cols_filtered]
        corr_cols = st.multiselect("Select variables for correlation", num_cols_filtered.tolist(), default=safe_default_corr_cols)

        # Heatmap for correlation matrix.
        if corr_cols and len(corr_cols) > 1 and not filtered_df[corr_cols].empty:
            corr_df = filtered_df[corr_cols].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale='RdBu', 
                zmin=-1, 
                zmax=1, 
                text=np.round(corr_df.values, 2), 
                texttemplate="%{text}"
            ))
            fig_corr.update_layout(title="Correlation Between Variables")
            st.plotly_chart(fig_corr, use_container_width=True)
        elif len(corr_cols) <= 1:
            st.info("Please select at least two numerical variables to view the correlation matrix.")
        else:
            st.warning("No data available for selected correlation variables or filtered dataset is empty.")

    with tab3:
        st.header("Geographic Analysis")
        # Informative note about country name matching for mapping.
        st.info("""
        **Mapping Note:** Country names must match Plotly's standard naming.
        Using a 3-letter ISO code column (e.g., 'iso_alpha3') can improve reliability.
        """)

        if 'year' in df.columns and 'country_name' in df.columns and not df.empty:
            year_to_map = st.slider("Select Year for Map", min_value=int(df['year'].min()),
                                     max_value=int(df['year'].max()),
                                     value=int(df['year'].max()))
            map_df = df[df['year'] == year_to_map].copy() 

            location_key = 'iso_alpha3' if 'iso_alpha3' in map_df.columns else 'country_name'
            location_mode = 'ISO-3' if 'iso_alpha3' in map_df.columns else 'country names'

            default_map_var_idx = num_cols_filtered.get_loc('life_expectancy') if 'life_expectancy' in num_cols_filtered else 0
            map_var = st.selectbox("Map Variable", num_cols_filtered.tolist(), index=default_map_var_idx)

            # Choropleth map for global distribution.
            if not map_df.empty and map_var in map_df.columns:
                fig_map = px.choropleth(
                    map_df,
                    locations=location_key, 
                    locationmode=location_mode, 
                    color=map_var,
                    hover_name="country_name",
                    hover_data=[c for c in num_cols_filtered if c != map_var],
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title=f"Global Distribution of {map_var.replace('_', ' ').title()} ({year_to_map})"
                )
                fig_map.update_layout(margin={"r":0, "t":40, "l":0, "b":0}) 
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning(f"No data available for the year {year_to_map} or selected map variable '{map_var}' is missing.")
        else:
            st.info("Geographic analysis requires 'year' and 'country_name' columns and data to be present.")

    with tab4:  # Deep Analysis Tab
        st.header("Advanced Analysis")
        # Feature importance using Random Forest Regressor.
        st.subheader("Feature Importance for Life Expectancy")
        st.markdown("Using Random Forest Regressor to determine which factors most influence life expectancy.")

        # Allow user to select features for the model and configure model parameters.
        available_model_features = [col for col in num_cols_filtered if col != 'life_expectancy' and filtered_df[col].nunique() > 1]
        default_model_features = safe_default_corr_cols if not st.session_state.feature_importance.empty else available_model_features[:5]

        with st.expander("Model Configuration & Feature Selection"):
            model_features = st.multiselect(
                "Select features for model (excluding Life Expectancy)",
                available_model_features,
                default=[f for f in default_model_features if f in available_model_features] 
            )
            n_estimators = st.slider("Number of trees (n_estimators)", 50, 500, 100)
            max_depth = st.slider("Max depth of trees", 3, 20, 10)

        if 'life_expectancy' in num_cols_filtered and model_features:
            model_df = filtered_df[model_features + ['life_expectancy']].dropna()
            MIN_ROWS_FOR_ML = 30 

            if len(model_df) >= MIN_ROWS_FOR_ML:
                X = model_df[model_features]
                y = model_df['life_expectancy']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                @st.cache_data
                def train_model(X_train_data, y_train_data, n_estimators_val, max_depth_val):
                    """Trains a RandomForestRegressor model with given parameters. Cached for performance."""
                    model = RandomForestRegressor(
                        n_estimators=n_estimators_val,
                        max_depth=max_depth_val,
                        random_state=42
                    )
                    model.fit(X_train_data, y_train_data)
                    return model

                with st.spinner('Training model...'):
                    model = train_model(X_train_scaled, y_train, n_estimators, max_depth)

                importance = model.feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': model_features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                st.session_state['feature_importance'] = feat_imp 

                y_pred = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                st.session_state['r2_score'] = r2 

                # Bar chart for feature importance.
                fig_imp = px.bar(
                    feat_imp,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Life Expectancy Prediction',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                # Display RMSE and R2 metrics.
                col1, col2 = st.columns(2)
                col1.metric("RMSE", f"{rmse:.2f}")
                col2.metric("R² Score", f"{r2:.2f}")

                st.markdown("""
                    - **RMSE (Root Mean Squared Error):** Measures the average magnitude of the errors between predicted and actual life expectancy. Lower RMSE indicates better model accuracy.
                    - **R² Score:** Represents the proportion of variance in the dependent variable (life expectancy) that is predictable from the independent variables (features). A higher R² (closer to 1) indicates a better fit of the model to the data.
                """)

                # Scatter plot for actual vs. predicted values.
                fig_pred = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual Life Expectancy', 'y': 'Predicted Life Expectancy'},
                    title='Actual vs Predicted Life Expectancy',
                    trendline='ols'
                )
                fig_pred.add_shape( 
                    type="line", line=dict(dash='dash'),
                    x0=y_test.min(), y0=y_test.min(),
                    x1=y_test.max(), y1=y_test.max()
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.warning(f"Not enough data ({len(model_df)} rows) to train the model. At least {MIN_ROWS_FOR_ML} rows are required after cleaning. Adjust filters or check data quality.")
        else:
            st.info("Please select features and ensure 'life_expectancy' is present in the data for building the prediction model.")

        st.subheader("Multidimensional Analysis")
        st.markdown("Explore relationships across multiple dimensions simultaneously.")

        # Default columns for parallel coordinates plot.
        default_pc_cols = [
            'life_expectancy', 'gdp_per_capita',
            'health_expenditure_per_capita', 'school_enrollment_combined'
        ]
        safe_default_pc_cols = [col for col in default_pc_cols if col in num_cols_filtered]
        pc_cols = st.multiselect(
            "Select variables for parallel coordinates",
            num_cols_filtered.tolist(),
            default=safe_default_pc_cols
        )

        # Parallel coordinates plot.
        if pc_cols and len(pc_cols) > 1 and not filtered_df[pc_cols].empty:
            fig_pc = px.parallel_coordinates(
                filtered_df,
                dimensions=pc_cols,
                color='life_expectancy' if 'life_expectancy' in pc_cols else pc_cols[0], 
                color_continuous_scale=px.colors.sequential.Viridis,
                title='Parallel Coordinates Analysis'
            )
            st.plotly_chart(fig_pc, use_container_width=True)
        elif len(pc_cols) <= 1:
            st.info("Please select at least two variables for parallel coordinates analysis.")
        else:
            st.warning("No data available for selected parallel coordinates variables or filtered dataset is empty.")

    with tab5:  
        st.header("Data-Driven Recommendations")
        st.subheader("Key Insights from Model")
        
        if 'feature_importance' in st.session_state and not st.session_state.feature_importance.empty:
            feat_imp = st.session_state.feature_importance
            top_features = feat_imp.head(5)['Feature'].tolist()
            st.markdown(f"""
            <div class="highlight">
                <h4>Primary Findings from Predictive Model:</h4>
                <ul>
                    <li>The most influential factors identified by the model include:
                        <b>{', '.join([f.replace('_', ' ').title() for f in top_features])}</b>.</li>
                    <li>The model achieved an R² score of <b>{st.session_state.get('r2_score', 'N/A'):.2f}</b>,
                        indicating its predictive power.</li>
                    <li>Further analysis of the correlation matrix (in 'Relationships' tab) can
                        provide deeper understanding of variable interdependencies.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Generate dynamic recommendation based on the top influencing factor.
            if not feat_imp.empty:
                top_factor = feat_imp.iloc[0]['Feature']
                factor_name = top_factor.replace('_', ' ').title()
                rec_map = {
                    'gdp_per_capita': "economic development initiatives and job creation programs",
                    'health_expenditure_per_capita': "healthcare infrastructure investment and public health programs",
                    'school_enrollment_combined': "education system reforms and access expansion, especially in primary and secondary education",
                    'mortality_infant': "maternal and child health programs, improved sanitation, and access to clean water",
                    'access_to_electricity': "rural electrification projects and energy infrastructure development",
                    'population': "sustainable population management and resource allocation strategies",
                    'urban_population_percent': "urban planning and infrastructure development to support growing city populations",
                    'fertility_rate': "family planning initiatives and access to reproductive health services"
                }
                recommendation = rec_map.get(top_factor, f"targeted interventions in key socioeconomic areas related to {factor_name}")
                st.markdown(f"""
                <div class="highlight">
                    <p><b> ✨  Priority Recommendation:</b> Focus on **{recommendation}**.</p>
                    <p>This is identified as crucial because **{factor_name}** was the most influential factor in predicting life expectancy based on our model.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feature importance data available yet. Please ensure you have uploaded data and the 'Deep Analysis' tab has run.")

        st.subheader("Selected Country Comparison (Normalized Metrics)")
        # Radar chart for comparing selected countries on key normalized metrics.
        if selected_countries and 'life_expectancy' in df.columns and len(selected_countries) > 1:
            comp_df = filtered_df.groupby('country_name').agg({
                'life_expectancy': 'mean',
                'gdp_per_capita': 'mean',
                'health_expenditure_per_capita': 'mean',
                'school_enrollment_combined': 'mean'
            }).reset_index()

            radar_cols = [col for col in ['life_expectancy', 'gdp_per_capita', 'health_expenditure_per_capita', 'school_enrollment_combined'] if col in comp_df.columns]
            if not radar_cols:
                st.warning("No suitable numerical columns found for radar chart in selected countries data.")
            else:
                radar_df = comp_df.copy()
                # Normalize values for radar chart (min-max scaling).
                for col in radar_cols:
                    min_val = radar_df[col].min()
                    max_val = radar_df[col].max()
                    if max_val > min_val:
                        radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
                    else:
                        radar_df[col] = 0.5 

                fig_radar = go.Figure()
                for i, row in radar_df.iterrows():
                    r_values = row[radar_cols].values.tolist() + [row[radar_cols[0]]] 
                    theta_values = [col.replace('_', ' ').title() for col in radar_cols] + [radar_cols[0].replace('_', ' ').title()]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=r_values,
                        theta=theta_values,
                        fill='toself',
                        name=row['country_name'].title()
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Selected Country Comparison (Normalized Metrics)"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Select two or more countries in the sidebar to enable country comparison.")

        st.subheader("Strategic Recommendations")
        # General strategic recommendations.
        st.markdown("""
        <p>Based on the analysis and common factors influencing life expectancy, here are strategic recommendations:</p>
        <div class="feature-box">
            <h4> 🏥  Healthcare Enhancement</h4>
            <p><b>Problem:</b> Disparities in healthcare access and quality often impact life expectancy significantly.<br/></p>
            <p><b>Recommendations:</b></p>
            <ul>
                <li>Increase healthcare workforce training and deployment in underserved areas.</li>
                <li>Improve medicine availability and supply chain resilience.</li>
                <li>Develop specialized healthcare infrastructure, especially for rural communities.</li>
                <li>Implement telemedicine programs to extend specialist access.</li>
            </ul>
        </div>
        <div class="feature-box">
            <h4> 🎓  Education Investment</h4>
            <p><b>Problem:</b> Educational attainment is strongly correlated with improved health outcomes and life expectancy.<br/></p>
            <p><b>Recommendations:</b></p>
            <ul>
                <li>Increase funding for primary and secondary education, focusing on quality and accessibility.</li>
                <li>Develop vocational training programs aligned with local economic needs.</li>
                <li>Implement school nutrition programs to support child development and learning.</li>
                <li>Expand digital literacy initiatives to empower communities.</li>
            </ul>
        </div>
        <div class="feature-box">
            <h4> 🤝  Strategic Partnerships</h4>
            <p><b>Problem:</b> Resource limitations can hinder progress in public health and development.<br/></p>
            <p><b>Recommendations:</b></p>
            <ul>
                <li>Establish technology transfer programs with developed nations for medical and educational advancements.</li>
                <li>Create regional health task forces for collaborative resource sharing and epidemic response.</li>
                <li>Develop public-private partnerships for essential infrastructure projects (e.g., sanitation, clean water, energy).</li>
                <li>Actively participate in international health initiatives for improved vaccine access and disease control.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Comparative example between Japan and Papua New Guinea.
        if 'japan' in filtered_df['country_name'].values and 'papua_new_guinea' in filtered_df['country_name'].values:
            japan_le = filtered_df[filtered_df['country_name'] == 'japan']['life_expectancy'].mean()
            png_le = filtered_df[filtered_df['country_name'] == 'papua_new_guinea']['life_expectancy'].mean()
            gap = japan_le - png_le

            # Safely get mean values, providing "N/A" if column is missing.
            japan_gdp = filtered_df[filtered_df['country_name'] == 'japan']['gdp_per_capita'].mean() if 'gdp_per_capita' in filtered_df.columns else "N/A"
            png_gdp = filtered_df[filtered_df['country_name'] == 'papua_new_guinea']['gdp_per_capita'].mean() if 'gdp_per_capita' in filtered_df.columns else "N/A"
            japan_health_exp = filtered_df[filtered_df['country_name'] == 'japan']['health_expenditure_per_capita'].mean() if 'health_expenditure_per_capita' in filtered_df.columns else "N/A"
            png_health_exp = filtered_df[filtered_df['country_name'] == 'papua_new_guinea']['health_expenditure_per_capita'].mean() if 'health_expenditure_per_capita' in filtered_df.columns else "N/A"
            japan_school = filtered_df[filtered_df['country_name'] == 'japan']['school_enrollment_combined'].mean() if 'school_enrollment_combined' in filtered_df.columns else "N/A"
            png_school = filtered_df[filtered_df['country_name'] == 'papua_new_guinea']['school_enrollment_combined'].mean() if 'school_enrollment_combined' in filtered_df.columns else "N/A"

            st.markdown(f"""
            <div class="call-to-action-box">
                <h4 style="color: inherit;">Closing the Gap: A Comparative Example</h4>
                <p>Comparing two countries often highlights areas for intervention. For example, the average life expectancy gap between
                Japan ({japan_le:.1f} years) and Papua New Guinea ({png_le:.1f} years)
                is approximately <b>{gap:.1f} years</b> (within the selected data range). While many factors contribute, typical differences might include:</p>
                <ul>
                    <li>**GDP per Capita:** Japan (~${japan_gdp:,.0f}) vs. Papua New Guinea (~${png_gdp:,.0f})</li>
                    <li>**Healthcare Expenditure per Capita:** Japan (~${japan_health_exp:,.0f}) vs. Papua New Guinea (~${png_health_exp:,.0f})</li>
                    <li>**School Enrollment:** Japan (~{japan_school:.1f}%) vs. Papua New Guinea (~{png_school:.1f}%)</li>
                </ul>
                <p>Addressing these disparities through targeted policies and investments could contribute significantly to improving life expectancy in nations facing similar challenges.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("For a detailed country comparison, ensure 'japan' and 'papua_new_guinea' are present in your dataset.")
