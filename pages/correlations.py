import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_processing import load_data, preprocess_data
from utils.navbar import create_navbar

def show():
    """Display the correlation analysis page content"""
    # Apply custom CSS
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Title
    st.title("Correlation Analysis")
    st.markdown("Explore relationships between different factors and their impact on stress levels.")
    
    # Load data
    try:
        df = load_data("attached_assets/corporate_stress_dataset.csv")
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Correlation matrix
    st.header("Correlation Matrix")
    
    # Get numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Generate heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    plt.title('Correlation Matrix of Numerical Factors')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Factor selection for scatter plots
    st.header("Factor Relationship Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_factor = st.selectbox(
            "Select X-Axis Factor",
            options=numerical_cols,
            index=numerical_cols.index("Work_Pressure_Level") if "Work_Pressure_Level" in numerical_cols else 0
        )
    
    with col2:
        y_factor = st.selectbox(
            "Select Y-Axis Factor",
            options=numerical_cols,
            index=numerical_cols.index("Stress_Level") if "Stress_Level" in numerical_cols else 0
        )
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_factor,
        y=y_factor,
        color='Department',
        size='Age',
        hover_data=['Gender', 'Job_Role'],
        title=f'Relationship between {x_factor} and {y_factor}',
        trendline='ols'
    )
    
    st.plotly_chart(fig, use_container_width=True)