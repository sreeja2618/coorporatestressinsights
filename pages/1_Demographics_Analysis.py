import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data, preprocess_data
from utils.navbar import create_navbar

# Set page configuration
st.set_page_config(
    page_title="Demographics Analysis - Corporate Stress Dashboard",
    page_icon="👥",
    layout="wide"
)

# Apply custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Display the navbar
create_navbar()

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

# Demographic filters
st.sidebar.header("Demographic Filters")

age_min = int(df['Age'].min())
age_max = int(df['Age'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
selected_gender = st.sidebar.multiselect("Gender", gender_options, default=["All"])

marital_options = ["All"] + sorted(df["Marital_Status"].unique().tolist())
selected_marital = st.sidebar.multiselect("Marital Status", marital_options, default=["All"])

company_size_options = ["All"] + sorted(df["Company_Size"].unique().tolist())
selected_company_size = st.sidebar.multiselect("Company Size", company_size_options, default=["All"])

location_options = ["All"] + sorted(df["Location"].unique().tolist())
selected_location = st.sidebar.multiselect("Location", location_options, default=["All"])

# Filter data based on selections
filtered_df = df.copy()

# Apply filters
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

if "All" not in selected_gender:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_gender)]

if "All" not in selected_marital:
    filtered_df = filtered_df[filtered_df['Marital_Status'].isin(selected_marital)]

if "All" not in selected_company_size:
    filtered_df = filtered_df[filtered_df['Company_Size'].isin(selected_company_size)]

if "All" not in selected_location:
    filtered_df = filtered_df[filtered_df['Location'].isin(selected_location)]

# Display filtered data info
st.markdown(f"### Analyzing {len(filtered_df)} employees")

# Age distribution and stress analysis
st.header("Age Demographics")
col1, col2 = st.columns(2)

with col1:
    age_bins = [20, 30, 40, 50, 60, 70]
    filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=age_bins, labels=['20-29', '30-39', '40-49', '50-59', '60+'])
    
    age_stress = filtered_df.groupby('Age_Group')['Stress_Level'].mean().reset_index()
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
    age_counts = filtered_df['Age_Group'].value_counts().reset_index()
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
    gender_stress = filtered_df.groupby('Gender')['Stress_Level'].mean().reset_index()
    
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
    gender_counts = filtered_df['Gender'].value_counts().reset_index()
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

# Marital Status analysis
st.header("Marital Status Analysis")
col1, col2 = st.columns(2)

with col1:
    marital_stress = filtered_df.groupby('Marital_Status')['Stress_Level'].mean().reset_index()
    
    fig = px.bar(
        marital_stress, 
        x='Marital_Status', 
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        title='Average Stress Level by Marital Status'
    )
    fig.update_layout(xaxis_title="Marital Status", yaxis_title="Average Stress Level", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    marital_counts = filtered_df['Marital_Status'].value_counts().reset_index()
    marital_counts.columns = ['Marital_Status', 'Count']
    
    fig = px.pie(
        marital_counts, 
        values='Count', 
        names='Marital_Status',
        title='Marital Status Distribution',
        hole=0.4
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Location analysis
st.header("Location Analysis")

location_stress = filtered_df.groupby('Location')['Stress_Level'].mean().reset_index()
location_stress = location_stress.sort_values(by='Stress_Level', ascending=False)

fig = px.bar(
    location_stress, 
    x='Location', 
    y='Stress_Level',
    color='Stress_Level',
    color_continuous_scale='RdYlGn_r',
    title='Average Stress Level by Location'
)
fig.update_layout(xaxis_title="Location", yaxis_title="Average Stress Level")
st.plotly_chart(fig, use_container_width=True)

# Gender bias experience by gender
st.header("Discrimination & Bias Analysis")
col1, col2 = st.columns(2)

with col1:
    gender_bias = filtered_df.groupby(['Gender', 'Gender_Bias_Experienced']).size().reset_index(name='Count')
    
    fig = px.bar(
        gender_bias, 
        x='Gender', 
        y='Count',
        color='Gender_Bias_Experienced',
        barmode='group',
        title='Gender Bias Experience by Gender'
    )
    fig.update_layout(xaxis_title="Gender", yaxis_title="Count", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    discrimination = filtered_df.groupby(['Gender', 'Discrimination_Experienced']).size().reset_index(name='Count')
    
    fig = px.bar(
        discrimination, 
        x='Gender', 
        y='Count',
        color='Discrimination_Experienced',
        barmode='group',
        title='Discrimination Experience by Gender'
    )
    fig.update_layout(xaxis_title="Gender", yaxis_title="Count", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Download the filtered data
st.sidebar.header("Export Data")
csv = filtered_df.to_csv(index=False)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='demographic_analysis.csv',
    mime='text/csv',
)
