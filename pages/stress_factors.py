import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data, preprocess_data
from utils.navbar import create_navbar

def show():
    """Display the stress factors analysis page content"""
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.title("Stress Factors Analysis")
    st.markdown("Analyze the various factors that contribute to workplace stress.")
    try:
        df = load_data("attached_assets/corporate_stress_dataset.csv")
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    st.header("Key Stress Factors")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stress_corr = df[numerical_cols].corr()['Stress_Level'].sort_values(ascending=False)
    stress_corr = stress_corr.drop('Stress_Level')
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