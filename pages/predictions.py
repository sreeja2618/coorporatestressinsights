import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
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

    # Model configuration with tabs
    tab1, tab2, tab3 = st.tabs(["Feature Selection", "Model Configuration", "Stress Predictor"])

    with tab1:
        st.subheader("Feature Selection")

        # Get numerical columns for potential features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols.remove('Stress_Level')  # Remove target variable

        # Split features by category
        work_features = ['Work_Pressure_Level', 'Working_Hours_per_Week', 'Manager_Support_Level', 
                         'Job_Satisfaction', 'Work_Life_Balance']
        personal_features = ['Age', 'Experience_Years', 'Monthly_Salary_INR', 'Sleep_Hours', 
                             'Exercise_Hours_Per_Week', 'Family_Support']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Work-Related Features**")
            work_selected = st.multiselect(
                "Select work-related features:",
                options=work_features,
                default=work_features
            )

        with col2:
            st.markdown("**Personal Features**")
            personal_selected = st.multiselect(
                "Select personal features:",
                options=personal_features,
                default=['Sleep_Hours', 'Monthly_Salary_INR']
            )

        # Combine selected features
        selected_features = work_selected + personal_selected

        if not selected_features:
            st.warning("Please select at least one feature to build the model.")
            st.stop()

        # Display feature correlation with stress
        st.markdown("### Feature Correlation with Stress")
        feature_corr = df[selected_features + ['Stress_Level']].corr()['Stress_Level'].drop('Stress_Level').sort_values(ascending=False)

        fig = px.bar(
            x=feature_corr.values,
            y=feature_corr.index,
            orientation='h',
            color=feature_corr.values,
            color_continuous_scale='RdBu_r',
            labels={'x': 'Correlation with Stress Level', 'y': 'Feature'},
            title='Feature Correlation with Stress Level'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            model_type = st.selectbox(
                "Select Machine Learning Model",
                options=["Random Forest", "Linear Regression", "Gradient Boosting"]
            )

            test_size = st.slider(
                "Test Data Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            ) / 100

        with col2:
            if model_type == "Random Forest":
                n_trees = st.slider("Number of Trees", 50, 500, 100, 50)
                max_depth = st.slider("Maximum Tree Depth", 3, 20, 10, 1)
            elif model_type == "Gradient Boosting":
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                n_trees = st.slider("Number of Trees", 50, 500, 100, 50)
            # Linear Regression has no hyperparameters to tune

        # Prepare data
        X = df[selected_features]
        y = df['Stress_Level']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Select and train model based on user choice
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
        elif model_type == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=n_trees, learning_rate=learning_rate, random_state=42)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Display metrics with more visual appeal
        st.markdown("### Model Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Squared Error", f"{mse:.4f}")
        col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
        col3.metric("R² Score", f"{r2:.4f}")

        # Plot actual vs predicted
        st.markdown("### Actual vs Predicted Stress Levels")

        compare_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })

        fig = px.scatter(
            compare_df, x='Actual', y='Predicted',
            trendline='ols',
            labels={'Actual': 'Actual Stress Level', 'Predicted': 'Predicted Stress Level'},
            title='Model Prediction Accuracy'
        )

        # Add a perfect prediction line
        fig.add_trace(
            go.Scatter(
                x=[min(y_test), max(y_test)],
                y=[min(y_test), max(y_test)],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.markdown("### Feature Importance")

        # Different models have different ways to get feature importance
        if model_type in ["Random Forest", "Gradient Boosting"]:
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
                color='Importance',
                color_continuous_scale='viridis',
                title='Feature Importance in Predicting Stress Level'
            )

            st.plotly_chart(fig, use_container_width=True)
        elif model_type == "Linear Regression":
            # For linear regression, we use coefficients
            coefficients = model.coef_
            feature_coeffs = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': coefficients
            }).sort_values('Coefficient', ascending=False)

            fig = px.bar(
                feature_coeffs,
                x='Coefficient',
                y='Feature',
                orientation='h',
                color='Coefficient',
                color_continuous_scale='RdBu',
                title='Feature Coefficients in Linear Regression Model'
            )

            st.plotly_chart(fig, use_container_width=True)
        
        # Add ROC curve analysis section
        st.markdown("### ROC Curve Analysis")
        st.markdown("""
            The ROC (Receiver Operating Characteristic) curve is a graphical visualization used to assess 
            the performance of a binary classification model by plotting the true positive rate against 
            the false positive rate at various threshold settings.
        """)
        
        # Create binary classification data for ROC curve
        # Convert stress level to binary target for ROC curve (high stress vs not)
        stress_threshold = st.slider(
            "Stress Threshold for Binary Classification",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
            help="Stress levels above this threshold are considered 'high stress'"
        )
        
        # Create binary target
        y_binary = (df['Stress_Level'] >= stress_threshold).astype(int)
        
        # Train binary classification models
        X_binary = df[selected_features]
        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
            X_binary, y_binary, test_size=test_size, random_state=42
        )
        
        # Train three different classification models
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        lr_classifier = LogisticRegression(random_state=42)
        gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Fit the models
        rf_classifier.fit(X_train_bin, y_train_bin)
        lr_classifier.fit(X_train_bin, y_train_bin)
        gb_classifier.fit(X_train_bin, y_train_bin)
        
        # Get probabilities for positive class
        rf_probs = rf_classifier.predict_proba(X_test_bin)[:, 1]
        lr_probs = lr_classifier.predict_proba(X_test_bin)[:, 1]
        gb_probs = gb_classifier.predict_proba(X_test_bin)[:, 1]
        
        # Compute ROC curve and area under the curve
        rf_fpr, rf_tpr, _ = roc_curve(y_test_bin, rf_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test_bin, lr_probs)
        gb_fpr, gb_tpr, _ = roc_curve(y_test_bin, gb_probs)
        
        rf_auc = auc(rf_fpr, rf_tpr)
        lr_auc = auc(lr_fpr, lr_tpr)
        gb_auc = auc(gb_fpr, gb_tpr)
        
        # Create ROC curve visualization using Plotly
        fig = go.Figure()
        
        # Add traces for each model
        fig.add_trace(go.Scatter(
            x=rf_fpr, y=rf_tpr,
            name=f'Random Forest (AUC = {rf_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=lr_fpr, y=lr_tpr,
            name=f'Logistic Regression (AUC = {lr_auc:.3f})',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=gb_fpr, y=gb_tpr,
            name=f'Gradient Boosting (AUC = {gb_auc:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Add diagonal line representing random guessing
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Guessing',
            line=dict(color='grey', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'ROC Curve Comparison (Threshold = {stress_threshold})',
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate'),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of the ROC curve
        st.info("""
            **Understanding the ROC Curve:**
            - The diagonal line represents random guessing (AUC = 0.5)
            - A model with perfect classification would have a curve that passes through the top-left corner (AUC = 1.0)
            - Higher AUC values indicate better model performance
            - Different models can be compared by their AUC scores
            - The threshold slider allows you to define what constitutes "high stress" for binary classification
        """)

    with tab3:
        st.subheader("Stress Level Predictor")
        st.markdown("""
            Use this interactive tool to predict stress levels based on different factors.
            Adjust the sliders below to see how changes in various factors affect predicted stress levels.
        """)

        # Create interactive sliders for each feature
        feature_values = {}

        # Organize features into two columns
        col1, col2 = st.columns(2)

        # Split features between columns
        half_point = len(selected_features) // 2 + len(selected_features) % 2
        left_features = selected_features[:half_point]
        right_features = selected_features[half_point:]

        # Create sliders for left column
        with col1:
            for feature in left_features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())

                # Determine step size based on the range
                range_size = max_val - min_val
                if range_size > 100:
                    step = 100
                elif range_size > 10:
                    step = 1
                elif range_size > 1:
                    step = 0.1
                else:
                    step = 0.01

                # Format the label to be more readable
                readable_label = feature.replace('_', ' ').title()

                feature_values[feature] = st.slider(
                    readable_label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=float(step)
                )

        # Create sliders for right column
        with col2:
            for feature in right_features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())

                # Determine step size based on the range
                range_size = max_val - min_val
                if range_size > 100:
                    step = 100
                elif range_size > 10:
                    step = 1
                elif range_size > 1:
                    step = 0.1
                else:
                    step = 0.01

                # Format the label to be more readable
                readable_label = feature.replace('_', ' ').title()

                feature_values[feature] = st.slider(
                    readable_label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=float(step)
                )

        # Create a dataframe with user-selected values
        user_input = pd.DataFrame([feature_values])

        # Make prediction
        prediction = model.predict(user_input)[0]

        # Determine stress category
        if prediction < 3:
            category = "Low Stress"
            color = "#4CAF50"  # Green
        elif prediction < 7:
            category = "Moderate Stress"
            color = "#FF9800"  # Orange
        else:
            category = "High Stress"
            color = "#F44336"  # Red

        # Display prediction with custom styling
        st.markdown("""
            <style>
            .prediction-box {
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .prediction-value {
                font-size: 48px;
                font-weight: bold;
                margin: 10px 0;
            }
            .prediction-category {
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 10px;
            }
            </style>
            """, unsafe_allow_html=True)

        prediction_html = f"""
            <div class="prediction-box" style="background-color: {color}20; border: 2px solid {color};">
                <p>Predicted Stress Level</p>
                <div class="prediction-value" style="color: {color};">{prediction:.2f}/10</div>
                <div class="prediction-category" style="color: {color};">{category}</div>
            </div>
        """
        st.markdown(prediction_html, unsafe_allow_html=True)

        # Recommendations based on prediction
        st.subheader("Recommendations")

        if prediction >= 7:
            st.markdown("""
                ### High Stress Level Alert

                Immediate actions recommended:
                - **Schedule an urgent meeting** with your supervisor to discuss workload reallocation
                - **Take immediate steps** to reduce overtime hours
                - **Book a consultation** with workplace counselor or stress management expert
                - **Consider using available mental health leave**
                - **Implement strict work-life boundaries** including no work emails after hours
            """)
        elif prediction >= 5:
            st.markdown("""
                ### Elevated Stress Level Warning

                Proactive steps to consider:
                - **Review and restructure** your daily work schedule
                - **Start a stress journal** to identify major stressors
                - **Practice mindfulness** for 15 minutes during work breaks
                - **Set up regular check-ins** with your manager
                - **Join workplace wellness programs** if available
            """)
        elif prediction >= 3:
            st.markdown("""
                ### Moderate Stress Level Notice

                Preventive measures to maintain:
                - **Optimize your work environment** for better focus
                - **Schedule regular breaks** during intense work periods
                - **Build a support network** among colleagues
                - **Practice time management** techniques
                - **Maintain regular exercise** routine
            """)
        else:
            st.markdown("""
                ### Healthy Stress Level

                Sustaining practices:
                - **Document your effective strategies** for future reference
                - **Mentor others** in stress management
                - **Continue balanced workload** practices
                - **Maintain your current exercise and sleep routine**
                - **Schedule regular career development** discussions
            """)

        # Offer a way to analyze what factors would most reduce stress
        st.subheader("Stress Reduction Analysis")

        # Create what-if analysis
        st.markdown("See which factors would have the biggest impact on reducing your predicted stress level:")

        # Calculate the impact of improving each factor
        impact_data = []

        for feature in selected_features:
            # Create a copy of the user input
            modified_input = user_input.copy()

            # Determine a "better" value for this feature
            if feature in ['Sleep_Hours', 'Exercise_Hours_Per_Week', 'Work_Life_Balance', 'Job_Satisfaction', 'Manager_Support_Level', 'Family_Support']:
                # For these factors, higher is better (generally)
                current = modified_input[feature].values[0]
                max_val = float(df[feature].max())
                improved_value = min(current + (max_val - current) * 0.2, max_val)  # 20% improvement towards max
                modified_input[feature] = improved_value
            elif feature in ['Work_Pressure_Level', 'Working_Hours_per_Week']:
                # For these factors, lower is better (generally)
                current = modified_input[feature].values[0]
                min_val = float(df[feature].min())
                improved_value = max(current - (current - min_val) * 0.2, min_val)  # 20% improvement towards min
                modified_input[feature] = improved_value
            else:
                # For other factors, we'll skip as the direction isn't clear
                continue

            # Predict with the modified input
            new_prediction = model.predict(modified_input)[0]

            # Calculate improvement
            improvement = prediction - new_prediction

            if improvement > 0:  # Only include factors that would reduce stress
                impact_data.append({
                    'Factor': feature.replace('_', ' ').title(),
                    'Potential Stress Reduction': improvement
                })

        # Create a dataframe from the impact data
        if impact_data:
            impact_df = pd.DataFrame(impact_data).sort_values('Potential Stress Reduction', ascending=False)

            fig = px.bar(
                impact_df,
                x='Potential Stress Reduction',
                y='Factor',
                orientation='h',
                color='Potential Stress Reduction',
                color_continuous_scale='Greens',
                title='Factors with Greatest Potential for Stress Reduction'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant stress reduction factors identified based on current settings.")