# Corporate Stress Analysis Dashboard

A comprehensive data analytics dashboard for corporate stress analysis built with Streamlit. This dashboard provides insights into workplace stress levels across different demographics, departments, and work factors.

## Features

- **Main Dashboard**: Overview of stress levels with key metrics and distribution
- **Demographics Analysis**: Analyze stress patterns by age, gender, marital status, and job roles
- **Stress Factors Analysis**: Identify key factors influencing stress levels
- **Department Analysis**: Compare stress across departments and identify high-risk areas
- **Correlation Analysis**: Understand relationships between various factors and stress
- **Predictive Insights**: Machine learning model to predict stress levels based on various factors

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install streamlit pandas numpy plotly matplotlib seaborn scikit-learn
   ```
3. Run the application:
   ```
   streamlit run main.py
   ```

## Project Structure

- `main.py`: Main application file with the home dashboard
- `pages/`: Contains the different analysis pages
  - `1_Demographics_Analysis.py`: Analysis by demographics
  - `2_Stress_Factors.py`: Analysis of work and personal factors
  - `3_Department_Analysis.py`: Department-specific analysis
  - `4_Correlation_Analysis.py`: Correlation studies
  - `5_Predictive_Insights.py`: ML-based prediction and insights
- `utils/`: Helper functions
  - `data_processing.py`: Data loading and preprocessing
  - `visualization.py`: Visualization functions
- `assets/`: Styling resources
- `attached_assets/`: Contains the dataset

## Dashboard Sections

### Main Dashboard
Provides an overview of stress levels in the organization with key metrics, stress distribution, and top factors correlated with stress.

### Demographics Analysis
Analyzes how stress varies across different demographic groups such as age, gender, experience, and job roles.

### Stress Factors Analysis
Explores various workplace and personal factors influencing stress levels, including working hours, management support, and work-life balance.

### Department Analysis
Compares stress levels across departments and identifies trends or patterns within specific departments.

### Correlation Analysis
Shows correlations between different factors and stress levels, helping identify relationships and potential causation.

### Predictive Insights
Uses machine learning to predict stress levels based on various factors and simulate scenarios for stress reduction.

## Data

The dashboard uses a corporate stress dataset with various metrics including:
- Demographics (age, gender, experience)
- Work factors (hours, pressure, management support)
- Personal factors (sleep, physical activity, family support)
- Organization factors (department, team size, training)

## Credits

Created as a data analytics project for corporate wellness and employee experience improvement.