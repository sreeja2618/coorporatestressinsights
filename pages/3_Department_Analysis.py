import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data, preprocess_data

# Set page configuration
st.set_page_config(
    page_title="Department Analysis - Corporate Stress Dashboard",
    page_icon="🏢",
    layout="wide"
)

# Apply custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.title("Department Analysis")
st.markdown("Compare stress levels and factors across different departments in your organization.")

# Load data
try:
    df = load_data("attached_assets/corporate_stress_dataset.csv")
    df = preprocess_data(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Department Filters")

company_size_options = ["All"] + sorted(df["Company_Size"].unique().tolist())
selected_company_size = st.sidebar.selectbox("Company Size", company_size_options)

location_options = ["All"] + sorted(df["Location"].unique().tolist())
selected_location = st.sidebar.selectbox("Location", location_options)

# Apply filters
filtered_df = df.copy()

if selected_company_size != "All":
    filtered_df = filtered_df[filtered_df['Company_Size'] == selected_company_size]

if selected_location != "All":
    filtered_df = filtered_df[filtered_df['Location'] == selected_location]

# Display filtered data info
st.markdown(f"### Analyzing {len(filtered_df)} employees across departments")

# Department stress overview
st.header("Department Stress Overview")

# Average stress by department
dept_stress = filtered_df.groupby('Department')['Stress_Level'].mean().reset_index()
dept_stress = dept_stress.sort_values(by='Stress_Level', ascending=False)

fig = px.bar(
    dept_stress,
    x='Department',
    y='Stress_Level',
    color='Stress_Level',
    color_continuous_scale='RdYlGn_r',
    title="Average Stress Level by Department"
)
fig.update_layout(xaxis_title="Department", yaxis_title="Average Stress Level", height=500)
st.plotly_chart(fig, use_container_width=True)

# Department demographics
st.header("Department Demographics")
col1, col2 = st.columns(2)

with col1:
    # Gender distribution by department
    gender_dept = filtered_df.groupby(['Department', 'Gender']).size().reset_index(name='Count')
    
    fig = px.bar(
        gender_dept,
        x='Department',
        y='Count',
        color='Gender',
        title="Gender Distribution by Department",
        barmode='stack'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Age distribution by department
    age_bins = [20, 30, 40, 50, 60, 70]
    filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=age_bins, labels=['20-29', '30-39', '40-49', '50-59', '60+'])
    
    age_dept = filtered_df.groupby(['Department', 'Age_Group']).size().reset_index(name='Count')
    
    fig = px.bar(
        age_dept,
        x='Department',
        y='Count',
        color='Age_Group',
        title="Age Distribution by Department",
        barmode='stack'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Departmental work factors
st.header("Work Factors by Department")

# Select work factor to analyze
work_factor = st.selectbox(
    "Select work factor to analyze by department",
    options=[
        'Working_Hours_per_Week',
        'Monthly_Salary_INR',
        'Work_Pressure_Level',
        'Manager_Support_Level',
        'Job_Satisfaction',
        'Work_Life_Balance',
        'Performance_Rating'
    ]
)

# Create box plot for selected factor
fig = px.box(
    filtered_df,
    x='Department',
    y=work_factor,
    color='Department',
    title=f"{work_factor} Distribution by Department"
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Average of the selected factor by department
dept_factor_avg = filtered_df.groupby('Department')[work_factor].mean().reset_index()
dept_factor_avg = dept_factor_avg.sort_values(by=work_factor, ascending=False)

fig = px.bar(
    dept_factor_avg,
    x='Department',
    y=work_factor,
    color=work_factor,
    title=f"Average {work_factor} by Department"
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Department health issues analysis
st.header("Health Issues by Department")

# Health issues distribution by department
health_dept = filtered_df.groupby(['Department', 'Health_Issues']).size().reset_index(name='Count')

fig = px.bar(
    health_dept,
    x='Department',
    y='Count',
    color='Health_Issues',
    title="Health Issues Distribution by Department",
    barmode='stack'
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Burnout symptoms by department
burnout_dept = filtered_df.groupby(['Department', 'Burnout_Symptoms']).size().reset_index(name='Count')

fig = px.bar(
    burnout_dept,
    x='Department',
    y='Count',
    color='Burnout_Symptoms',
    title="Burnout Symptoms Distribution by Department",
    barmode='stack'
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Stress level comparison by job role
st.header("Job Role Analysis within Departments")

# Department selection for job role analysis
selected_dept_for_role = st.selectbox(
    "Select department for job role analysis",
    options=sorted(filtered_df['Department'].unique().tolist())
)

# Filter data for selected department
dept_df = filtered_df[filtered_df['Department'] == selected_dept_for_role]

# Stress by job role within department
role_stress = dept_df.groupby('Job_Role')['Stress_Level'].mean().reset_index()
role_stress = role_stress.sort_values(by='Stress_Level', ascending=False)

fig = px.bar(
    role_stress,
    x='Job_Role',
    y='Stress_Level',
    color='Stress_Level',
    color_continuous_scale='RdYlGn_r',
    title=f"Average Stress Level by Job Role in {selected_dept_for_role} Department"
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Department work conditions
st.header("Work Conditions by Department")
col1, col2 = st.columns(2)

with col1:
    # Remote work distribution by department
    remote_dept = filtered_df.groupby(['Department', 'Remote_Work']).size().reset_index(name='Count')
    remote_total = filtered_df.groupby('Department').size().reset_index(name='Total')
    remote_dept = pd.merge(remote_dept, remote_total, on='Department')
    remote_dept['Percentage'] = (remote_dept['Count'] / remote_dept['Total']) * 100
    
    # Filter for True remote work
    remote_true = remote_dept[remote_dept['Remote_Work'] == True]
    
    fig = px.bar(
        remote_true,
        x='Department',
        y='Percentage',
        color='Percentage',
        color_continuous_scale='Blues',
        title="Remote Work Percentage by Department"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Average working hours by department
    hours_dept = filtered_df.groupby('Department')['Working_Hours_per_Week'].mean().reset_index()
    hours_dept = hours_dept.sort_values(by='Working_Hours_per_Week', ascending=False)
    
    fig = px.bar(
        hours_dept,
        x='Department',
        y='Working_Hours_per_Week',
        color='Working_Hours_per_Week',
        color_continuous_scale='Reds',
        title="Average Working Hours by Department"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Department comparison table
st.header("Department Comparison Table")

# Calculate department stats
dept_stats = filtered_df.groupby('Department').agg({
    'Stress_Level': 'mean',
    'Working_Hours_per_Week': 'mean',
    'Monthly_Salary_INR': 'mean',
    'Work_Pressure_Level': 'mean',
    'Manager_Support_Level': 'mean',
    'Job_Satisfaction': 'mean',
    'Work_Life_Balance': 'mean',
    'ID': 'count'
}).reset_index()

dept_stats.columns = ['Department', 'Avg Stress Level', 'Avg Working Hours', 'Avg Salary (INR)', 
                     'Avg Work Pressure', 'Avg Manager Support', 'Avg Job Satisfaction', 
                     'Avg Work-Life Balance', 'Employee Count']

# Format the table
dept_stats['Avg Stress Level'] = dept_stats['Avg Stress Level'].round(2)
dept_stats['Avg Working Hours'] = dept_stats['Avg Working Hours'].round(1)
dept_stats['Avg Salary (INR)'] = dept_stats['Avg Salary (INR)'].round(0).astype(int)
dept_stats['Avg Work Pressure'] = dept_stats['Avg Work Pressure'].round(2)
dept_stats['Avg Manager Support'] = dept_stats['Avg Manager Support'].round(2)
dept_stats['Avg Job Satisfaction'] = dept_stats['Avg Job Satisfaction'].round(2)
dept_stats['Avg Work-Life Balance'] = dept_stats['Avg Work-Life Balance'].round(2)

st.dataframe(dept_stats, use_container_width=True)

# Download the department stats
csv = dept_stats.to_csv(index=False)
st.download_button(
    label="Download Department Comparison as CSV",
    data=csv,
    file_name='department_comparison.csv',
    mime='text/csv',
)

# Download the filtered data
st.sidebar.header("Export Data")
csv_filtered = filtered_df.to_csv(index=False)
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=csv_filtered,
    file_name='department_analysis.csv',
    mime='text/csv',
)
