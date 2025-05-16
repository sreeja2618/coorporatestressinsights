import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load and return the corporate stress dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the corporate stress dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataset
    """
    
    processed_df = df.copy()
    bool_cols = ['Remote_Work', 'Mental_Health_Leave_Taken', 'Training_Opportunities',
                'Gender_Bias_Experienced', 'Discrimination_Experienced']
    
    for col in bool_cols:
        processed_df[col] = processed_df[col].astype('bool')
    num_cols = ['Age', 'Experience_Years', 'Monthly_Salary_INR', 'Working_Hours_per_Week',
                'Commute_Time_Hours', 'Stress_Level', 'Sleep_Hours', 'Physical_Activity_Hours_per_Week',
                'Manager_Support_Level', 'Work_Pressure_Level', 'Annual_Leaves_Taken',
                'Work_Life_Balance', 'Family_Support_Level', 'Job_Satisfaction',
                'Performance_Rating', 'Team_Size']
    
    for col in num_cols:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    if processed_df.isnull().sum().sum() > 0:
        for col in num_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
        cat_cols = ['Gender', 'Marital_Status', 'Job_Role', 'Health_Issues', 
                    'Company_Size', 'Department', 'Burnout_Symptoms', 'Location']
        
        for col in cat_cols:
            if processed_df[col].isnull().sum() > 0:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    return processed_df

def calculate_stress_metrics(df):
    """
    Calculate key stress metrics from the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataset
        
    Returns:
    --------
    dict
        Dictionary of stress metrics
    """
    metrics = {}
    metrics['avg_stress'] = df['Stress_Level'].mean()
    high_stress_count = (df['Stress_Level'] >= 7).sum()
    metrics['high_stress_percentage'] = (high_stress_count / len(df)) * 100
    low_stress_count = (df['Stress_Level'] <= 3).sum()
    metrics['low_stress_percentage'] = (low_stress_count / len(df)) * 100
    burnout_count = df[df['Burnout_Symptoms'].isin(['Yes', 'Occasional'])].shape[0]
    metrics['burnout_percentage'] = (burnout_count / len(df)) * 100
    metrics['gender_stress'] = df.groupby('Gender')['Stress_Level'].mean().to_dict()
    metrics['department_stress'] = df.groupby('Department')['Stress_Level'].mean().to_dict()
    metrics['working_hours_corr'] = df['Working_Hours_per_Week'].corr(df['Stress_Level'])
    metrics['manager_support_corr'] = df['Manager_Support_Level'].corr(df['Stress_Level'])
    
    return metrics

def get_stress_level_category(stress_level):
    """
    Convert numerical stress level to category
    
    Parameters:
    -----------
    stress_level : float
        Numerical stress level (0-10)
        
    Returns:
    --------
    str
        Stress level category
    """
    if stress_level <= 3:
        return "Low"
    elif stress_level <= 6:
        return "Moderate"
    else:
        return "High"
