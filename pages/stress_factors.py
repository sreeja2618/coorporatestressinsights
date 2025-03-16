import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data, preprocess_data
from utils.navbar import create_navbar

def show():
    """Display the stress factors analysis page content"""
    # Apply custom CSS
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Title
    st.title("Stress Factors Analysis")
    st.markdown("Analyze the various factors that contribute to workplace stress.")
    
    # Load data
    try:
        df = load_data("attached_assets/corporate_stress_dataset.csv")
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Key stress factors analysis
    st.header("Key Stress Factors")
    
    # Get numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlations with Stress_Level
    stress_corr = df[numerical_cols].corr()['Stress_Level'].sort_values(ascending=False)
    # Remove the stress level itself from the list
    stress_corr = stress_corr.drop('Stress_Level')
    
    # Display the correlation heatmap
    fig = px.bar(
        x=stress_corr.values, 
        y=stress_corr.index,
        orientation='h',
        color=stress_corr.values,
        color_continuous_scale='RdBu_r',
        title='Factors Correlated with Stress Level'
    )
    fig.update_layout(xaxis_title="Correlation Coefficient", yaxis_title="Factor")
    st.plotly_chart(fig, use_container_width=True)
    
    # Work pressure analysis
    st.header("Work Pressure Analysis")
    
    work_pressure = df.groupby('Work_Pressure_Level')['Stress_Level'].mean().reset_index()
    
    fig = px.scatter(
        df, 
        x='Work_Pressure_Level', 
        y='Stress_Level',
        color='Stress_Level',
        size='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        title='Work Pressure vs. Stress Level',
        trendline='ols'
    )
    fig.update_layout(xaxis_title="Work Pressure Level", yaxis_title="Stress Level")
    st.plotly_chart(fig, use_container_width=True)