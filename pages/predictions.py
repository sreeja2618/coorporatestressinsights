import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_processing import load_data, preprocess_data
from utils.navbar import create_navbar

def show():
    """Display the predictive insights page content"""
    # Apply custom CSS
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Title
    st.title("Predictive Insights")
    st.markdown("Explore predictive models for understanding and forecasting stress levels.")
    
    # Load data
    try:
        df = load_data("attached_assets/corporate_stress_dataset.csv")
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Basic model information
    st.header("Stress Level Prediction Model")
    
    st.info("""
        This page demonstrates a simple machine learning model that predicts employee stress levels
        based on various factors. The model uses a Random Forest algorithm, which can capture 
        complex relationships between different factors.
    """)
    
    # Features selection
    st.subheader("Feature Selection")
    
    # Get numerical columns for potential features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('Stress_Level')  # Remove target variable
    
    # Let user select features
    selected_features = st.multiselect(
        "Select features to include in the model",
        options=numerical_cols,
        default=['Work_Pressure_Level', 'Working_Hours_per_Week', 'Sleep_Hours', 'Manager_Support_Level', 'Job_Satisfaction']
    )
    
    if not selected_features:
        st.warning("Please select at least one feature to build the model.")
        st.stop()
    
    # Build and evaluate model
    st.subheader("Model Performance")
    
    # Prepare data
    X = df[selected_features]
    y = df['Stress_Level']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mse:.4f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
    col3.metric("R² Score", f"{r2:.4f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Predicting Stress Level'
    )
    
    st.plotly_chart(fig, use_container_width=True)