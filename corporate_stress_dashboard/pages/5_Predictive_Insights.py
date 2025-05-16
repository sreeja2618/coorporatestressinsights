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

# Set page configuration
st.set_page_config(
    page_title="Predictive Insights - Corporate Stress Dashboard",
    page_icon="🔮",
    layout="wide"
)

# Apply custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.title("Predictive Insights")
st.markdown("Discover key predictors of stress levels and simulate different scenarios.")

# Load data
try:
    df = load_data("attached_assets/corporate_stress_dataset.csv")
    df = preprocess_data(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Convert categorical variables to numerical for modeling
model_df = df.copy()

# Encoding categorical variables
categorical_cols = ['Gender', 'Marital_Status', 'Job_Role', 'Health_Issues', 
                  'Company_Size', 'Department', 'Burnout_Symptoms', 'Location']

# Remove ID column
model_df = model_df.drop('ID', axis=1)

# One-hot encode categorical variables
model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

# Convert boolean columns to integers
bool_cols = ['Remote_Work', 'Mental_Health_Leave_Taken', 'Training_Opportunities',
           'Gender_Bias_Experienced', 'Discrimination_Experienced']
for col in bool_cols:
    model_df[col] = model_df[col].astype(int)

# Prepare data for modeling
X = model_df.drop('Stress_Level', axis=1)
y = model_df['Stress_Level']

# Keep track of feature names
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
st.header("Stress Level Prediction Model")
st.markdown("A machine learning model to identify key factors predicting stress levels.")

with st.spinner("Training predictive model..."):
    # Train model with lower n_estimators for demonstration
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

# Display model performance
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.4f}")
col2.metric("R² Score", f"{r2:.4f}")

# Feature importance
st.subheader("Key Factors Predicting Stress Levels")

# Get feature importances
importance = model.feature_importances_
indices = np.argsort(importance)[-15:]  # Get indices of top 15 features

# Create feature importance chart
fig = px.bar(
    x=importance[indices],
    y=[feature_names[i] for i in indices],
    orientation='h',
    color=importance[indices],
    color_continuous_scale='YlOrRd',
    title="Top 15 Factors Influencing Stress Levels"
)
fig.update_layout(xaxis_title="Importance", yaxis_title="Factor", height=600)
st.plotly_chart(fig, use_container_width=True)

# Stress Level Simulator
st.header("Stress Level Simulator")
st.markdown("""
    Use this simulator to understand how changing different factors might affect stress levels.
    Adjust the sliders to see the predicted impact on stress level.
""")

# Create two columns for simulator inputs
col1, col2 = st.columns(2)

# Define input ranges for significant numerical factors
with col1:
    st.subheader("Work Factors")
    sim_working_hours = st.slider("Working Hours per Week", 30, 90, 40)
    sim_work_pressure = st.slider("Work Pressure Level", 0, 10, 5)
    sim_manager_support = st.slider("Manager Support Level", 0, 10, 5)
    sim_work_life_balance = st.slider("Work-Life Balance", 0, 10, 5)
    sim_job_satisfaction = st.slider("Job Satisfaction", 0, 10, 5)

with col2:
    st.subheader("Personal Factors")
    sim_sleep_hours = st.slider("Sleep Hours", 4.0, 9.0, 7.0)
    sim_physical_activity = st.slider("Physical Activity Hours per Week", 0.0, 10.0, 3.0)
    sim_family_support = st.slider("Family Support Level", 0, 10, 5)
    sim_remote_work = st.selectbox("Remote Work", ["Yes", "No"]) == "Yes"
    sim_annual_leaves = st.slider("Annual Leaves Taken", 0, 30, 15)

# Create a feature vector for prediction using a sample row as template
if st.button("Predict Stress Level"):
    # Get a sample row to use as a template for prediction
    sample_row = X.iloc[0].copy()
    
    # Update values based on user inputs
    for feature in feature_names:
        if 'Working_Hours_per_Week' in feature:
            sample_row[feature] = sim_working_hours
        elif 'Work_Pressure_Level' in feature:
            sample_row[feature] = sim_work_pressure
        elif 'Manager_Support_Level' in feature:
            sample_row[feature] = sim_manager_support
        elif 'Work_Life_Balance' in feature:
            sample_row[feature] = sim_work_life_balance
        elif 'Job_Satisfaction' in feature:
            sample_row[feature] = sim_job_satisfaction
        elif 'Sleep_Hours' in feature:
            sample_row[feature] = sim_sleep_hours
        elif 'Physical_Activity_Hours_per_Week' in feature:
            sample_row[feature] = sim_physical_activity
        elif 'Family_Support_Level' in feature:
            sample_row[feature] = sim_family_support
        elif 'Remote_Work' in feature:
            sample_row[feature] = int(sim_remote_work)
        elif 'Annual_Leaves_Taken' in feature:
            sample_row[feature] = sim_annual_leaves
    
    # Make prediction
    predicted_stress = model.predict([sample_row])[0]
    
    # Display prediction
    st.markdown("### Predicted Stress Level")
    
    # Create a gauge chart for the predicted stress level
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_stress,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Stress Level"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "green"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': predicted_stress
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpret the result
    if predicted_stress < 3:
        stress_level = "low"
        recommendations = [
            "Maintain current work-life balance practices",
            "Continue regular physical activity and sleep patterns",
            "Sustain positive management practices"
        ]
    elif predicted_stress < 7:
        stress_level = "moderate"
        recommendations = [
            "Consider slight adjustments to working hours or workload",
            "Ensure adequate sleep and physical activity",
            "Utilize available leave days periodically",
            "Maintain open communication with management"
        ]
    else:
        stress_level = "high"
        recommendations = [
            "Significantly reduce working hours if possible",
            "Prioritize work-life balance improvement",
            "Increase management support and communication",
            "Focus on improving sleep quality and physical activity",
            "Consider using mental health resources or leave if needed"
        ]
    
    st.markdown(f"#### This scenario indicates a **{stress_level}** stress level.")
    
    st.markdown("#### Recommendations:")
    for rec in recommendations:
        st.markdown(f"- {rec}")

# Department Stress Prediction
st.header("Department Stress Prediction")
st.markdown("Predict average stress levels for different departments with varying conditions.")

# Department selection
selected_dept = st.selectbox(
    "Select Department",
    options=sorted(df['Department'].unique().tolist())
)

# Create scenario columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Current Scenario")
    # Get current department averages
    dept_df = df[df['Department'] == selected_dept]
    
    current_hours = dept_df['Working_Hours_per_Week'].mean()
    current_pressure = dept_df['Work_Pressure_Level'].mean()
    current_manager = dept_df['Manager_Support_Level'].mean()
    current_balance = dept_df['Work_Life_Balance'].mean()
    
    st.metric("Avg Working Hours", f"{current_hours:.1f}")
    st.metric("Avg Work Pressure", f"{current_pressure:.1f}")
    st.metric("Avg Manager Support", f"{current_manager:.1f}")
    st.metric("Avg Work-Life Balance", f"{current_balance:.1f}")
    
    current_stress = dept_df['Stress_Level'].mean()
    st.metric("Current Avg Stress", f"{current_stress:.2f}")

with col2:
    st.subheader("Improved Scenario")
    # Decrease hours and pressure, increase support and balance
    improved_hours = max(current_hours - 10, 35)
    improved_pressure = max(current_pressure - 2, 1)
    improved_manager = min(current_manager + 2, 10)
    improved_balance = min(current_balance + 2, 10)
    
    st.metric("Avg Working Hours", f"{improved_hours:.1f}", f"{improved_hours - current_hours:.1f}")
    st.metric("Avg Work Pressure", f"{improved_pressure:.1f}", f"{improved_pressure - current_pressure:.1f}")
    st.metric("Avg Manager Support", f"{improved_manager:.1f}", f"{improved_manager - current_manager:.1f}")
    st.metric("Avg Work-Life Balance", f"{improved_balance:.1f}", f"{improved_balance - current_balance:.1f}")
    
    # Placeholder for prediction (will calculate below)
    improved_stress_placeholder = st.empty()

with col3:
    st.subheader("Worsened Scenario")
    # Increase hours and pressure, decrease support and balance
    worsened_hours = min(current_hours + 10, 90)
    worsened_pressure = min(current_pressure + 2, 10)
    worsened_manager = max(current_manager - 2, 0)
    worsened_balance = max(current_balance - 2, 0)
    
    st.metric("Avg Working Hours", f"{worsened_hours:.1f}", f"{worsened_hours - current_hours:.1f}")
    st.metric("Avg Work Pressure", f"{worsened_pressure:.1f}", f"{worsened_pressure - current_pressure:.1f}")
    st.metric("Avg Manager Support", f"{worsened_manager:.1f}", f"{worsened_manager - current_manager:.1f}")
    st.metric("Avg Work-Life Balance", f"{worsened_balance:.1f}", f"{worsened_balance - current_balance:.1f}")
    
    # Placeholder for prediction (will calculate below)
    worsened_stress_placeholder = st.empty()

# Calculate scenario predictions
if st.button("Calculate Scenario Predictions"):
    # Prepare department data for prediction
    dept_model_df = model_df[X.columns].copy()
    dept_X = dept_model_df.copy()
    
    # Create improved scenario data
    improved_X = dept_X.copy()
    for i in range(len(improved_X)):
        improved_X.loc[i, 'Working_Hours_per_Week'] = improved_hours
        improved_X.loc[i, 'Work_Pressure_Level'] = improved_pressure
        improved_X.loc[i, 'Manager_Support_Level'] = improved_manager
        improved_X.loc[i, 'Work_Life_Balance'] = improved_balance
    
    # Create worsened scenario data
    worsened_X = dept_X.copy()
    for i in range(len(worsened_X)):
        worsened_X.loc[i, 'Working_Hours_per_Week'] = worsened_hours
        worsened_X.loc[i, 'Work_Pressure_Level'] = worsened_pressure
        worsened_X.loc[i, 'Manager_Support_Level'] = worsened_manager
        worsened_X.loc[i, 'Work_Life_Balance'] = worsened_balance
    
    # Make predictions for each scenario
    improved_preds = model.predict(improved_X)
    worsened_preds = model.predict(worsened_X)
    
    # Calculate average predictions
    improved_stress = np.mean(improved_preds)
    worsened_stress = np.mean(worsened_preds)
    
    # Update placeholders with predictions
    improved_stress_placeholder.metric(
        "Predicted Avg Stress", 
        f"{improved_stress:.2f}", 
        f"{improved_stress - current_stress:.2f}"
    )
    
    worsened_stress_placeholder.metric(
        "Predicted Avg Stress", 
        f"{worsened_stress:.2f}", 
        f"{worsened_stress - current_stress:.2f}"
    )
    
    # Create comparison chart
    scenarios = ['Current', 'Improved', 'Worsened']
    stress_values = [current_stress, improved_stress, worsened_stress]
    
    fig = px.bar(
        x=scenarios,
        y=stress_values,
        color=stress_values,
        color_continuous_scale='RdYlGn_r',
        title=f"Stress Level Scenarios for {selected_dept} Department"
    )
    fig.update_layout(xaxis_title="Scenario", yaxis_title="Average Stress Level", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate potential impact
    potential_reduction = current_stress - improved_stress
    st.markdown(f"#### Potential stress reduction: {potential_reduction:.2f} points ({(potential_reduction/current_stress)*100:.1f}%)")
    
    if potential_reduction > 2:
        impact = "significant"
    elif potential_reduction > 1:
        impact = "moderate"
    else:
        impact = "minor"
    
    st.markdown(f"This represents a **{impact}** potential improvement in employee wellbeing.")
