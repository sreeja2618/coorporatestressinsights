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
create_navbar()

# Title and introduction
st.title("Corporate Stress Analysis Dashboard")
st.markdown("""
    This interactive dashboard analyzes corporate stress levels based on multiple factors.
    Navigate through different pages to explore demographics, stress factors, departmental analysis, and correlations.
""")

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
    
    # Quick filters for exploration
    st.header("Quick Data Explorer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_dept = st.selectbox(
            "Select Department",
            options=["All"] + sorted(df['Department'].unique().tolist())
        )
    
    with col2:
        selected_gender = st.selectbox(
            "Select Gender",
            options=["All"] + sorted(df['Gender'].unique().tolist())
        )
    
    with col3:
        selected_factor = st.selectbox(
            "Select Factor to Analyze",
            options=[
                "Age", "Experience_Years", "Monthly_Salary_INR", 
                "Working_Hours_per_Week", "Sleep_Hours",
                "Manager_Support_Level", "Work_Pressure_Level",
                "Work_Life_Balance", "Job_Satisfaction"
            ]
        )
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    
    # Create chart based on selections
    if not filtered_df.empty:
        fig = create_stress_by_factor_chart(filtered_df, selected_factor)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")
    
    # Call to action for detailed analysis
    st.info("⬆️ Navigate to the different analysis pages from the navigation bar above for more detailed insights.")
else:
    st.error("Failed to load data. Please check your dataset file.")
