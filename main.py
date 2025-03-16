import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processing import load_data, preprocess_data, calculate_stress_metrics
from utils.visualization import create_stress_distribution, create_stress_by_factor_chart
from utils.navbar import create_navbar

# Set page configuration
st.set_page_config(
    page_title="Corporate Stress Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Display the navbar
current_page = create_navbar()

# Hide default sidebar
st.markdown("""
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""", unsafe_allow_html=True)

# Conditional content based on the selected page
if current_page == "Home":
    # Title and introduction
    st.title("Corporate Stress Analysis Dashboard")
    st.markdown("""
        This interactive dashboard analyzes corporate stress levels based on multiple factors.
        Use the navigation bar above to explore demographics, stress factors, departmental analysis, and correlations.
    """)
elif current_page == "Demographics":
    # Import and run Demographics page
    import pages.demographics as demographics
    demographics.show()
    st.stop()
elif current_page == "Stress Factors":
    # Import and run Stress Factors page
    import pages.stress_factors as stress_factors
    stress_factors.show()
    st.stop()
elif current_page == "Departments":
    # Import and run Departments page
    import pages.departments as departments
    departments.show()
    st.stop()
elif current_page == "Correlations":
    # Import and run Correlations page
    import pages.correlations as correlations
    correlations.show()
    st.stop()
elif current_page == "Predictions":
    # Import and run Predictions page
    import pages.predictions as predictions
    predictions.show()
    st.stop()

# Data loading section
st.header("Data Overview")

# Data upload option
uploaded_file = st.file_uploader("Upload your corporate stress dataset (CSV)", type="csv")

# Load data
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    # Use the provided dataset
    df = load_data("attached_assets/corporate_stress_dataset.csv")

if df is not None:
    # Preprocess data
    df = preprocess_data(df)
    
    # Display data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Sample")
        st.dataframe(df.head(5), use_container_width=True)
    
    with col2:
        st.subheader("Data Summary")
        st.write(f"Total Records: {df.shape[0]}")
        st.write(f"Features: {df.shape[1]}")
        
        # Missing values if any
        missing_values = df.isnull().sum().sum()
        st.write(f"Missing Values: {missing_values}")
    
    # Key Metrics
    st.header("Key Stress Metrics")
    metrics = calculate_stress_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Stress Level", f"{metrics['avg_stress']:.2f}/10")
    col2.metric("High Stress Percentage", f"{metrics['high_stress_percentage']:.1f}%")
    col3.metric("Low Stress Percentage", f"{metrics['low_stress_percentage']:.1f}%")
    col4.metric("Burnout Symptoms", f"{metrics['burnout_percentage']:.1f}%")
    
    # Stress distribution visualization
    st.subheader("Stress Level Distribution")
    fig = create_stress_distribution(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top stress factors
    st.subheader("Top Factors Correlated with Stress")
    
    # Get only numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlations with Stress_Level
    stress_corr = df[numerical_cols].corr()['Stress_Level'].sort_values(ascending=False)
    # Remove the stress level itself from the list
    stress_corr = stress_corr.drop('Stress_Level')
    
    # Get top 5 positive and negative correlations
    top_pos = stress_corr.head(5)
    top_neg = stress_corr.tail(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Factors that increase stress")
        fig = px.bar(
            x=top_pos.values,
            y=top_pos.index,
            orientation='h',
            color=top_pos.values,
            color_continuous_scale='reds',
            title="Positive Correlations with Stress Level"
        )
        fig.update_layout(xaxis_title="Correlation", yaxis_title="Factor", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.caption("Factors that reduce stress")
        fig = px.bar(
            x=top_neg.values,
            y=top_neg.index,
            orientation='h',
            color=top_neg.values,
            color_continuous_scale='blues_r',
            title="Negative Correlations with Stress Level"
        )
        fig.update_layout(xaxis_title="Correlation", yaxis_title="Factor", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Data Explorer with enhanced filters
    st.header("Interactive Data Explorer")
    
    with st.expander("Configure Visualization Parameters", expanded=True):
        # Tabs for different filter categories
        filter_tabs = st.tabs(["Demographics", "Work Environment", "Visualization"])
        
        with filter_tabs[0]:  # Demographics filters
            col1, col2 = st.columns(2)
            
            with col1:
                selected_dept = st.selectbox(
                    "Department",
                    options=["All"] + sorted(df['Department'].unique().tolist())
                )
                
                age_range = st.slider(
                    "Age Range",
                    min_value=int(df['Age'].min()),
                    max_value=int(df['Age'].max()),
                    value=(int(df['Age'].min()), int(df['Age'].max()))
                )
                
                experience_range = st.slider(
                    "Experience (Years)",
                    min_value=int(df['Experience_Years'].min()),
                    max_value=int(df['Experience_Years'].max()),
                    value=(int(df['Experience_Years'].min()), int(df['Experience_Years'].max()))
                )
            
            with col2:
                selected_gender = st.selectbox(
                    "Gender",
                    options=["All"] + sorted(df['Gender'].unique().tolist())
                )
                
                marital_status = st.selectbox(
                    "Marital Status",
                    options=["All"] + sorted(df['Marital_Status'].unique().tolist())
                )
                
                job_role = st.selectbox(
                    "Job Role",
                    options=["All"] + sorted(df['Job_Role'].unique().tolist())
                )
        
        with filter_tabs[1]:  # Work Environment filters
            col1, col2 = st.columns(2)
            
            with col1:
                work_hours_range = st.slider(
                    "Working Hours per Week",
                    min_value=int(df['Working_Hours_per_Week'].min()),
                    max_value=int(df['Working_Hours_per_Week'].max()),
                    value=(int(df['Working_Hours_per_Week'].min()), int(df['Working_Hours_per_Week'].max()))
                )
                
                sleep_hours_range = st.slider(
                    "Sleep Hours",
                    min_value=int(df['Sleep_Hours'].min()),
                    max_value=int(df['Sleep_Hours'].max()),
                    value=(int(df['Sleep_Hours'].min()), int(df['Sleep_Hours'].max()))
                )
            
            with col2:
                work_pressure_range = st.slider(
                    "Work Pressure Level",
                    min_value=int(df['Work_Pressure_Level'].min()),
                    max_value=int(df['Work_Pressure_Level'].max()),
                    value=(int(df['Work_Pressure_Level'].min()), int(df['Work_Pressure_Level'].max()))
                )
                
                job_satisfaction_range = st.slider(
                    "Job Satisfaction",
                    min_value=int(df['Job_Satisfaction'].min()),
                    max_value=int(df['Job_Satisfaction'].max()),
                    value=(int(df['Job_Satisfaction'].min()), int(df['Job_Satisfaction'].max()))
                )
        
        with filter_tabs[2]:  # Visualization options
            col1, col2 = st.columns(2)
            
            with col1:
                selected_factor = st.selectbox(
                    "Factor to Analyze",
                    options=[
                        "Age", "Experience_Years", "Monthly_Salary_INR", 
                        "Working_Hours_per_Week", "Sleep_Hours",
                        "Manager_Support_Level", "Work_Pressure_Level",
                        "Work_Life_Balance", "Job_Satisfaction"
                    ]
                )
                
                color_scale = st.selectbox(
                    "Color Scale",
                    options=["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "blues", "reds"]
                )
            
            with col2:
                chart_type = st.radio(
                    "Chart Type",
                    options=["Scatter", "Box", "Violin", "Bar"],
                    horizontal=True
                )
                
                trend_line = st.checkbox("Show Trend Line", value=True)
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    # Apply demographic filters
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    if marital_status != "All":
        filtered_df = filtered_df[filtered_df['Marital_Status'] == marital_status]
    if job_role != "All":
        filtered_df = filtered_df[filtered_df['Job_Role'] == job_role]
    
    # Apply age and experience filters
    filtered_df = filtered_df[
        (filtered_df['Age'] >= age_range[0]) & 
        (filtered_df['Age'] <= age_range[1]) &
        (filtered_df['Experience_Years'] >= experience_range[0]) & 
        (filtered_df['Experience_Years'] <= experience_range[1])
    ]
    
    # Apply work environment filters
    filtered_df = filtered_df[
        (filtered_df['Working_Hours_per_Week'] >= work_hours_range[0]) & 
        (filtered_df['Working_Hours_per_Week'] <= work_hours_range[1]) &
        (filtered_df['Sleep_Hours'] >= sleep_hours_range[0]) & 
        (filtered_df['Sleep_Hours'] <= sleep_hours_range[1]) &
        (filtered_df['Work_Pressure_Level'] >= work_pressure_range[0]) & 
        (filtered_df['Work_Pressure_Level'] <= work_pressure_range[1]) &
        (filtered_df['Job_Satisfaction'] >= job_satisfaction_range[0]) & 
        (filtered_df['Job_Satisfaction'] <= job_satisfaction_range[1])
    ]
    
    # Display filtered data summary
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Filtered Dataset: {len(filtered_df)} records**")
    with col2:
        st.caption(f"**Average Stress Level: {filtered_df['Stress_Level'].mean():.2f}/10**")
    
    # Create chart based on selections and filtered data
    if not filtered_df.empty:
        if chart_type == "Scatter":
            fig = px.scatter(
                filtered_df, 
                x=selected_factor, 
                y="Stress_Level",
                color="Stress_Level",
                size="Stress_Level",
                hover_data=["Department", "Gender", "Age", "Experience_Years"],
                color_continuous_scale=color_scale,
                title=f"Stress Level vs {selected_factor}",
                trendline="ols" if trend_line else None
            )
        elif chart_type == "Box":
            # For box plots, we need to bin numerical data if selected
            if selected_factor in ["Age", "Experience_Years", "Monthly_Salary_INR", "Working_Hours_per_Week", "Sleep_Hours"]:
                # Create bins for numerical data
                if selected_factor == "Age":
                    filtered_df["Binned"] = pd.cut(filtered_df[selected_factor], bins=[20, 30, 40, 50, 60], labels=["20-30", "31-40", "41-50", "51-60"])
                elif selected_factor == "Experience_Years":
                    filtered_df["Binned"] = pd.cut(filtered_df[selected_factor], bins=[0, 5, 10, 15, 20, 25], labels=["0-5", "6-10", "11-15", "16-20", "21-25"])
                elif selected_factor == "Monthly_Salary_INR":
                    filtered_df["Binned"] = pd.qcut(filtered_df[selected_factor], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
                else:
                    filtered_df["Binned"] = pd.qcut(filtered_df[selected_factor], q=4, duplicates='drop')
                
                fig = px.box(
                    filtered_df,
                    x="Binned",
                    y="Stress_Level",
                    color="Binned",
                    title=f"Stress Level Distribution by {selected_factor} (Binned)",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
            else:
                fig = px.box(
                    filtered_df,
                    x=selected_factor,
                    y="Stress_Level",
                    color=selected_factor,
                    title=f"Stress Level Distribution by {selected_factor}",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
        elif chart_type == "Violin":
            if selected_factor in ["Age", "Experience_Years", "Monthly_Salary_INR", "Working_Hours_per_Week", "Sleep_Hours"]:
                # Create bins for numerical data
                if selected_factor == "Age":
                    filtered_df["Binned"] = pd.cut(filtered_df[selected_factor], bins=[20, 30, 40, 50, 60], labels=["20-30", "31-40", "41-50", "51-60"])
                elif selected_factor == "Experience_Years":
                    filtered_df["Binned"] = pd.cut(filtered_df[selected_factor], bins=[0, 5, 10, 15, 20, 25], labels=["0-5", "6-10", "11-15", "16-20", "21-25"])
                else:
                    filtered_df["Binned"] = pd.qcut(filtered_df[selected_factor], q=4, duplicates='drop')
                
                fig = px.violin(
                    filtered_df,
                    x="Binned",
                    y="Stress_Level",
                    color="Binned",
                    box=True,
                    title=f"Stress Level Distribution by {selected_factor} (Binned)",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
            else:
                fig = px.violin(
                    filtered_df,
                    x=selected_factor,
                    y="Stress_Level",
                    color=selected_factor,
                    box=True,
                    title=f"Stress Level Distribution by {selected_factor}",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
        else:  # Bar chart
            if selected_factor in ["Age", "Experience_Years", "Monthly_Salary_INR", "Working_Hours_per_Week", "Sleep_Hours"]:
                # Create bins for numerical data
                if selected_factor == "Age":
                    filtered_df["Binned"] = pd.cut(filtered_df[selected_factor], bins=[20, 30, 40, 50, 60], labels=["20-30", "31-40", "41-50", "51-60"])
                elif selected_factor == "Experience_Years":
                    filtered_df["Binned"] = pd.cut(filtered_df[selected_factor], bins=[0, 5, 10, 15, 20, 25], labels=["0-5", "6-10", "11-15", "16-20", "21-25"])
                else:
                    filtered_df["Binned"] = pd.qcut(filtered_df[selected_factor], q=4, duplicates='drop')
                
                group_df = filtered_df.groupby("Binned")["Stress_Level"].mean().reset_index()
                
                fig = px.bar(
                    group_df,
                    x="Binned",
                    y="Stress_Level",
                    color="Stress_Level",
                    title=f"Average Stress Level by {selected_factor} (Binned)",
                    color_continuous_scale=color_scale
                )
            else:
                group_df = filtered_df.groupby(selected_factor)["Stress_Level"].mean().reset_index()
                
                fig = px.bar(
                    group_df,
                    x=selected_factor,
                    y="Stress_Level",
                    color="Stress_Level",
                    title=f"Average Stress Level by {selected_factor}",
                    color_continuous_scale=color_scale
                )
        
        # Update layout
        fig.update_layout(
            xaxis_title=selected_factor.replace('_', ' '),
            yaxis_title="Stress Level",
            legend_title=selected_factor.replace('_', ' '),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a sample data view
        with st.expander("View Sample of Filtered Data", expanded=False):
            st.dataframe(filtered_df.head(10), use_container_width=True)
            
            # Add download button for filtered data
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"stress_data_filtered_{selected_factor}.csv",
                mime="text/csv"
            )
    else:
        st.warning("No data available for the selected filters. Please adjust your filter criteria.")
    
    # Call to action for detailed analysis
    st.info("⬆️ Navigate to the different analysis pages from the navigation bar above for more detailed insights.")
else:
    st.error("Failed to load data. Please check your dataset file.")
