import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_processing import load_data, preprocess_data
st.set_page_config(
    page_title="Correlation Analysis - Corporate Stress Dashboard",
    page_icon="🔄",
    layout="wide"
)
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.title("Correlation Analysis")
st.markdown("Explore relationships between different variables and their impact on stress levels.")
try:
    df = load_data("attached_assets/corporate_stress_dataset.csv")
    df = preprocess_data(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
numerical_cols = [
    'Age', 'Experience_Years', 'Monthly_Salary_INR', 'Working_Hours_per_Week',
    'Commute_Time_Hours', 'Stress_Level', 'Sleep_Hours', 'Physical_Activity_Hours_per_Week',
    'Manager_Support_Level', 'Work_Pressure_Level', 'Annual_Leaves_Taken',
    'Work_Life_Balance', 'Family_Support_Level', 'Job_Satisfaction',
    'Performance_Rating', 'Team_Size'
]
st.sidebar.header("Analysis Options")
department_options = ["All"] + sorted(df["Department"].unique().tolist())
selected_department = st.sidebar.selectbox("Department", department_options)
filtered_df = df.copy()
if selected_department != "All":
    filtered_df = filtered_df[filtered_df['Department'] == selected_department]
st.markdown(f"### Analyzing correlations across {len(filtered_df)} employee records")
st.header("Correlation Matrix")
corr_df = filtered_df[numerical_cols].corr()
fig = px.imshow(
    corr_df,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    aspect="auto",
    title="Correlation Matrix of Numerical Variables"
)
fig.update_layout(height=700)
st.plotly_chart(fig, use_container_width=True)
st.header("Relationship Explorer")
col1, col2 = st.columns(2)

with col1:
    x_var = st.selectbox("Select X-axis variable", options=numerical_cols, index=numerical_cols.index('Working_Hours_per_Week'))

with col2:
    y_var = st.selectbox("Select Y-axis variable", options=numerical_cols, index=numerical_cols.index('Stress_Level'))
color_var = st.selectbox(
    "Color by",
    options=["Stress_Level", "Department", "Gender", "Job_Role", "Remote_Work"],
    index=0
)
fig = px.scatter(
    filtered_df,
    x=x_var,
    y=y_var,
    color=color_var,
    opacity=0.7,
    size="Stress_Level" if color_var != "Stress_Level" else None,
    hover_data=["ID", "Job_Role", "Department"],
    title=f"Relationship between {x_var} and {y_var}"
)
if color_var == "Stress_Level" or color_var == "Remote_Work":
    fig.update_traces(marker=dict(size=10))
    x = filtered_df[x_var]
    y = filtered_df[y_var]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=sorted(x),
        y=p(sorted(x)),
        mode='lines',
        name='Trend',
        line=dict(color='black', dash='dash')
    ))

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)
corr_value = filtered_df[x_var].corr(filtered_df[y_var])
st.markdown(f"**Correlation coefficient between {x_var} and {y_var}:** {corr_value:.4f}")
if abs(corr_value) < 0.3:
    strength = "weak"
    color = "blue"
elif abs(corr_value) < 0.7:
    strength = "moderate"
    color = "orange"
else:
    strength = "strong"
    color = "red"

direction = "positive" if corr_value > 0 else "negative"
st.markdown(f"This indicates a <span style='color:{color}'>{strength} {direction}</span> correlation.", unsafe_allow_html=True)
st.header("Multi-variable Relationships")
facet_var = st.selectbox(
    "Facet by (categorize)",
    options=["Department", "Gender", "Job_Role", "Company_Size", "Remote_Work", "Health_Issues", "Burnout_Symptoms"],
    index=0
)
fig = px.scatter(
    filtered_df,
    x=x_var,
    y=y_var,
    color="Stress_Level",
    facet_col=facet_var if filtered_df[facet_var].nunique() <= 4 else None,
    facet_row=None if filtered_df[facet_var].nunique() <= 4 else facet_var,
    opacity=0.7,
    title=f"Relationship between {x_var} and {y_var} by {facet_var}"
)
if filtered_df[facet_var].nunique() <= 8:  
    fig.update_layout(height=max(600, filtered_df[facet_var].nunique() * 200))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Too many categories in {facet_var} for faceting. Please choose a variable with fewer categories.")
st.header("Stress Level Correlations")
stress_corr = filtered_df[numerical_cols].corr()['Stress_Level'].sort_values(ascending=False)
stress_corr = stress_corr.drop('Stress_Level')
top_pos = stress_corr.head(6)
top_neg = stress_corr.tail(6).sort_values()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Factors Increasing Stress")
    fig = px.bar(
        x=top_pos.values,
        y=top_pos.index,
        orientation='h',
        color=top_pos.values,
        color_continuous_scale='Reds',
        title="Top Positive Correlations with Stress"
    )
    fig.update_layout(xaxis_title="Correlation", yaxis_title="Factor", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Factors Reducing Stress")
    fig = px.bar(
        x=top_neg.values,
        y=top_neg.index,
        orientation='h',
        color=top_neg.values,
        color_continuous_scale='Blues_r',
        title="Top Negative Correlations with Stress"
    )
    fig.update_layout(xaxis_title="Correlation", yaxis_title="Factor", height=400)
    st.plotly_chart(fig, use_container_width=True)
st.header("Top Correlated Pairs")
corr_abs = filtered_df[numerical_cols].corr().abs()
np.fill_diagonal(corr_abs.values, 0)
corr_pairs = []
for i in range(len(corr_abs.columns)):
    for j in range(i):
        corr_pairs.append((corr_abs.columns[i], corr_abs.columns[j], corr_abs.iloc[i, j]))
corr_pairs.sort(key=lambda x: x[2], reverse=True)
st.subheader("Highly Correlated Variable Pairs")
top_pairs_df = pd.DataFrame(corr_pairs[:10], columns=['Variable 1', 'Variable 2', 'Correlation'])
top_pairs_df['Correlation'] = top_pairs_df['Correlation'].round(4)

st.dataframe(top_pairs_df, use_container_width=True)
if st.button("Visualize Top Correlation Pair"):
    top_var1, top_var2, _ = corr_pairs[0]
    
    fig = px.scatter(
        filtered_df,
        x=top_var1,
        y=top_var2,
        color="Stress_Level",
        opacity=0.7,
        title=f"Relationship between {top_var1} and {top_var2}"
    )
    x = filtered_df[top_var1]
    y = filtered_df[top_var2]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=sorted(x),
        y=p(sorted(x)),
        mode='lines',
        name='Trend',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
st.sidebar.header("Export Data")
corr_csv = corr_df.to_csv()
st.sidebar.download_button(
    label="Download Correlation Matrix as CSV",
    data=corr_csv,
    file_name='correlation_matrix.csv',
    mime='text/csv',
)
