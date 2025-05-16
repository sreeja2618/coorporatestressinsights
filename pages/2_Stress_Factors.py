import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data, preprocess_data
st.set_page_config(
    page_title="Stress Factors Analysis - Corporate Stress Dashboard",
    page_icon="📈",
    layout="wide"
)
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.title("Stress Factors Analysis")
st.markdown("Analyze key factors contributing to workplace stress and their relationships.")
try:
    df = load_data("attached_assets/corporate_stress_dataset.csv")
    df = preprocess_data(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
work_factors = [
    'Working_Hours_per_Week', 
    'Commute_Time_Hours', 
    'Remote_Work', 
    'Monthly_Salary_INR',
    'Experience_Years',
    'Manager_Support_Level',
    'Work_Pressure_Level',
    'Work_Life_Balance',
    'Team_Size',
    'Job_Satisfaction'
]

health_factors = [
    'Health_Issues',
    'Sleep_Hours',
    'Physical_Activity_Hours_per_Week',
    'Mental_Health_Leave_Taken',
    'Burnout_Symptoms'
]

social_factors = [
    'Family_Support_Level',
    'Gender_Bias_Experienced',
    'Discrimination_Experienced'
]
st.sidebar.header("Filter Options")

department_options = ["All"] + sorted(df["Department"].unique().tolist())
selected_department = st.sidebar.selectbox("Department", department_options)

job_role_options = ["All"] + sorted(df["Job_Role"].unique().tolist())
selected_job_role = st.sidebar.selectbox("Job Role", job_role_options)
filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df['Department'] == selected_department]

if selected_job_role != "All":
    filtered_df = filtered_df[filtered_df['Job_Role'] == selected_job_role]
st.markdown(f"### Analyzing {len(filtered_df)} employees")
st.header("Factors Correlated with Stress Level")
numerical_cols = work_factors + ['Sleep_Hours', 'Physical_Activity_Hours_per_Week', 
                               'Annual_Leaves_Taken', 'Team_Size']
numerical_df = filtered_df[numerical_cols + ['Stress_Level']]
correlations = numerical_df.corr()['Stress_Level'].drop('Stress_Level').sort_values(ascending=False)
fig = px.bar(
    x=correlations.values,
    y=correlations.index,
    orientation='h',
    color=correlations.values,
    color_continuous_scale='RdBu_r',
    title="Correlation of Factors with Stress Level"
)
fig.update_layout(
    xaxis_title="Correlation Coefficient", 
    yaxis_title="Factor",
    height=500
)
st.plotly_chart(fig, use_container_width=True)
st.header("Detailed Factor Analysis")
selected_factor = st.selectbox(
    "Select a factor for detailed analysis", 
    options=work_factors + health_factors + ['Annual_Leaves_Taken'],
    index=0
)
if selected_factor in numerical_cols:
    fig = px.scatter(
        filtered_df, 
        x=selected_factor, 
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        opacity=0.7,
        title=f"Relationship between {selected_factor} and Stress Level"
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=500)
    x = filtered_df[selected_factor]
    y = filtered_df['Stress_Level']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=sorted(x),
        y=p(sorted(x)),
        mode='lines',
        name='Trend',
        line=dict(color='black', dash='dash')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(
        filtered_df, 
        x=selected_factor,
        nbins=30,
        title=f"Distribution of {selected_factor}"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_factor == 'Health_Issues':
    health_stress = filtered_df.groupby('Health_Issues')['Stress_Level'].mean().reset_index()
    health_count = filtered_df.groupby('Health_Issues').size().reset_index(name='Count')
    health_data = pd.merge(health_stress, health_count, on='Health_Issues')
    
    fig = px.bar(
        health_data,
        x='Health_Issues',
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        text='Count',
        title="Average Stress Level by Health Issues"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_factor == 'Remote_Work':
    remote_stress = filtered_df.groupby('Remote_Work')['Stress_Level'].mean().reset_index()
    remote_count = filtered_df.groupby('Remote_Work').size().reset_index(name='Count')
    remote_data = pd.merge(remote_stress, remote_count, on='Remote_Work')
    
    fig = px.bar(
        remote_data,
        x='Remote_Work',
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        text='Count',
        title="Average Stress Level by Remote Work Status"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    remote_dept = filtered_df.groupby(['Department', 'Remote_Work'])['Stress_Level'].mean().reset_index()
    
    fig = px.bar(
        remote_dept,
        x='Department',
        y='Stress_Level',
        color='Remote_Work',
        barmode='group',
        title="Impact of Remote Work on Stress Level by Department"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_factor == 'Mental_Health_Leave_Taken':
    mental_stress = filtered_df.groupby('Mental_Health_Leave_Taken')['Stress_Level'].mean().reset_index()
    mental_count = filtered_df.groupby('Mental_Health_Leave_Taken').size().reset_index(name='Count')
    mental_data = pd.merge(mental_stress, mental_count, on='Mental_Health_Leave_Taken')
    
    fig = px.bar(
        mental_data,
        x='Mental_Health_Leave_Taken',
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        text='Count',
        title="Average Stress Level by Mental Health Leave Status"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    mental_dept = filtered_df.groupby(['Department', 'Mental_Health_Leave_Taken'])['Stress_Level'].mean().reset_index()
    
    fig = px.bar(
        mental_dept,
        x='Department',
        y='Stress_Level',
        color='Mental_Health_Leave_Taken',
        barmode='group',
        title="Mental Health Leave Impact on Stress Level by Department"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
elif selected_factor == 'Burnout_Symptoms':
    burnout_stress = filtered_df.groupby('Burnout_Symptoms')['Stress_Level'].mean().reset_index()
    burnout_count = filtered_df.groupby('Burnout_Symptoms').size().reset_index(name='Count')
    burnout_data = pd.merge(burnout_stress, burnout_count, on='Burnout_Symptoms')
    
    fig = px.bar(
        burnout_data,
        x='Burnout_Symptoms',
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        text='Count',
        title="Average Stress Level by Burnout Symptoms"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
st.header("Working Hours Impact")
hours_bins = [30, 40, 50, 60, 70, 80, 90]
filtered_df['Hours_Group'] = pd.cut(filtered_df['Working_Hours_per_Week'], bins=hours_bins, labels=['30-40', '40-50', '50-60', '60-70', '70-80', '80-90'])

hours_stress = filtered_df.groupby('Hours_Group')['Stress_Level'].mean().reset_index()
hours_count = filtered_df.groupby('Hours_Group').size().reset_index(name='Count')
hours_data = pd.merge(hours_stress, hours_count, on='Hours_Group')

fig = px.bar(
    hours_data,
    x='Hours_Group',
    y='Stress_Level',
    color='Stress_Level',
    color_continuous_scale='RdYlGn_r',
    text='Count',
    title="Impact of Working Hours on Stress Level"
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)
st.sidebar.header("Export Data")
csv = filtered_df.to_csv(index=False)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='stress_factors_analysis.csv',
    mime='text/csv',
)
