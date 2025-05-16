import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data, preprocess_data
from utils.navbar import create_navbar

def show():
    """Display the departments analysis page content"""
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.title("Department Analysis")
    st.markdown("Compare stress levels and factors across different departments.")
    try:
        df = load_data("attached_assets/corporate_stress_dataset.csv")
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    st.header("Department Stress Comparison")
    
    dept_stress = df.groupby('Department')['Stress_Level'].mean().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        dept_stress,
        x='Department',
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        title='Average Stress Level by Department'
    )
    fig.update_layout(xaxis_title="Department", yaxis_title="Average Stress Level")
    st.plotly_chart(fig, use_container_width=True)
    st.header("Department Distribution")
    
    dept_counts = df['Department'].value_counts().reset_index()
    dept_counts.columns = ['Department', 'Count']
    
    fig = px.pie(
        dept_counts,
        values='Count',
        names='Department',
        title='Department Distribution',
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)