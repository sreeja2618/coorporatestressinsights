import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data, preprocess_data
from utils.navbar import create_navbar

def show():
    """Display the demographics analysis page content"""
    # Apply custom CSS
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Title
    st.title("Demographics Analysis")
    st.markdown("Explore how stress levels vary across different demographic groups in the organization.")
    
    # Load data
    try:
        df = load_data("attached_assets/corporate_stress_dataset.csv")
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Age distribution and stress analysis
    st.header("Age Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age_bins = [20, 30, 40, 50, 60, 70]
        df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=['20-29', '30-39', '40-49', '50-59', '60+'])
        
        age_stress = df.groupby('Age_Group')['Stress_Level'].mean().reset_index()
        fig = px.bar(
            age_stress, 
            x='Age_Group', 
            y='Stress_Level',
            color='Stress_Level',
            color_continuous_scale='RdYlGn_r',
            title='Average Stress Level by Age Group'
        )
        fig.update_layout(xaxis_title="Age Group", yaxis_title="Average Stress Level", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        age_counts = df['Age_Group'].value_counts().reset_index()
        age_counts.columns = ['Age_Group', 'Count']
        
        fig = px.pie(
            age_counts, 
            values='Count', 
            names='Age_Group',
            title='Age Group Distribution',
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Gender analysis
    st.header("Gender Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        gender_stress = df.groupby('Gender')['Stress_Level'].mean().reset_index()
        
        fig = px.bar(
            gender_stress, 
            x='Gender', 
            y='Stress_Level',
            color='Gender',
            title='Average Stress Level by Gender'
        )
        fig.update_layout(xaxis_title="Gender", yaxis_title="Average Stress Level", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender',
            title='Gender Distribution',
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)