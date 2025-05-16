import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils.data_processing import load_data, preprocess_data
st.set_page_config(
    page_title="Predictive Insights - Corporate Stress Dashboard",
    page_icon="🔮",
    layout="wide"
)
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.title("Predictive Insights")
st.markdown("Discover key predictors of stress levels and simulate different scenarios.")
try:
    df = load_data("attached_assets/corporate_stress_dataset.csv")
    df = preprocess_data(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
model_df = df.copy()
categorical_cols = ['Gender', 'Marital_Status', 'Job_Role', 'Health_Issues', 
                  'Company_Size', 'Department', 'Burnout_Symptoms', 'Location']
model_df = model_df.drop('ID', axis=1)
model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)
bool_cols = ['Remote_Work', 'Mental_Health_Leave_Taken', 'Training_Opportunities',
           'Gender_Bias_Experienced', 'Discrimination_Experienced']
for col in bool_cols:
    model_df[col] = model_df[col].astype(int)
X = model_df.drop('Stress_Level', axis=1)
y = model_df['Stress_Level']
feature_names = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.header("Stress Level Prediction Model")
st.markdown("A machine learning model to identify key factors predicting stress levels.")

with st.spinner("Training predictive model..."):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.4f}")
col2.metric("R² Score", f"{r2:.4f}")
st.header("Stress Level Simulator")
sim_working_hours = st.slider("Working Hours per Week", 30, 90, 40)
sim_work_pressure = st.slider("Work Pressure Level", 0, 10, 5)
sim_manager_support = st.slider("Manager Support Level", 0, 10, 5)
sim_work_life_balance = st.slider("Work-Life Balance", 0, 10, 5)
sim_sleep_hours = st.slider("Sleep Hours", 4.0, 9.0, 7.0)

if st.button("Predict Stress Level"):
    sample_row = X.iloc[0].copy()
    sample_row['Working_Hours_per_Week'] = sim_working_hours
    sample_row['Work_Pressure_Level'] = sim_work_pressure
    sample_row['Manager_Support_Level'] = sim_manager_support
    sample_row['Work_Life_Balance'] = sim_work_life_balance
    sample_row['Sleep_Hours'] = sim_sleep_hours

    predicted_stress = model.predict([sample_row])[0]

    st.markdown("### Predicted Stress Level")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_stress,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Stress Level"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 5], 'color': "green"},
                {'range': [5, 10], 'color': "red"},
            ],
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    if predicted_stress < 5:
        recommendations = [
            "Maintain current work-life balance practices",
            "Continue regular physical activity and sleep patterns",
            "Stay engaged in social and relaxation activities"
        ]
    else:
        recommendations = [
            "Reduce working hours if possible",
            "Improve work-life balance and relaxation routines",
            "Consider taking periodic breaks or seeking professional support"
        ]
    
    st.markdown("#### Recommendations:")
    for rec in recommendations:
        st.markdown(f"- {rec}")
