import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

def create_stress_distribution(df):
    """
    Create a visualization of stress level distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataset
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization figure
    """
  
    df_viz = df.copy()
    
    
    bins = [-1, 3, 6, 10]
    labels = ['Low (0-3)', 'Moderate (4-6)', 'High (7-10)']
    df_viz['Stress_Category'] = pd.cut(df_viz['Stress_Level'], bins=bins, labels=labels)
    
    
    stress_counts = df_viz['Stress_Category'].value_counts().reset_index()
    stress_counts.columns = ['Stress_Category', 'Count']
    
    
    total = stress_counts['Count'].sum()
    stress_counts['Percentage'] = (stress_counts['Count'] / total) * 100
    
    
    stress_counts['Stress_Category'] = pd.Categorical(
        stress_counts['Stress_Category'],
        categories=labels,
        ordered=True
    )
    stress_counts = stress_counts.sort_values('Stress_Category')
    
   
    color_map = {
        'Low (0-3)': '#4CAF50',     
        'Moderate (4-6)': '#FFC107', 
        'High (7-10)': '#F44336'     
    }
    
    
    fig = px.bar(
        stress_counts,
        x='Stress_Category',
        y='Count',
        color='Stress_Category',
        color_discrete_map=color_map,
        text=stress_counts['Percentage'].apply(lambda x: f'{x:.1f}%')
    )
    
   
    fig.update_layout(
        title='Distribution of Stress Levels',
        xaxis_title='Stress Level Category',
        yaxis_title='Number of Employees',
        showlegend=False
    )
    
    return fig

def create_stress_by_factor_chart(df, factor):
    """
    Create a visualization of stress level by a specific factor
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataset
    factor : str
        Factor to analyze
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization figure
    """
   
    if factor in ['Age', 'Experience_Years', 'Monthly_Salary_INR', 'Working_Hours_per_Week', 
                'Sleep_Hours', 'Physical_Activity_Hours_per_Week']:
        
        fig = px.scatter(
            df,
            x=factor,
            y='Stress_Level',
            color='Stress_Level',
            color_continuous_scale='RdYlGn_r',
            opacity=0.7,
            title=f'Stress Level by {factor}'
        )
        
        
        x = df[factor]
        y = df['Stress_Level']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        
        fig.add_trace(go.Scatter(
            x=sorted(x),
            y=p(sorted(x)),
            mode='lines',
            name='Trend',
            line=dict(color='black', dash='dash')
        ))
        
    elif factor in ['Manager_Support_Level', 'Work_Pressure_Level', 'Work_Life_Balance', 'Job_Satisfaction']:
        
        fig = px.box(
            df,
            x=factor,
            y='Stress_Level',
            color=factor,
            color_continuous_scale='RdYlGn_r',
            title=f'Stress Level by {factor}'
        )
        
    else:
        
        fig = px.scatter(
            df,
            x=factor,
            y='Stress_Level',
            color='Stress_Level',
            color_continuous_scale='RdYlGn_r',
            opacity=0.7,
            title=f'Stress Level by {factor}'
        )
    
    
    fig.update_layout(
        xaxis_title=factor,
        yaxis_title='Stress Level'
    )
    
    return fig

def create_department_comparison(df):
    """
    Create a visualization comparing stress levels across departments
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataset
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization figure
    """
   
    dept_stress = df.groupby('Department')['Stress_Level'].mean().reset_index()
    dept_stress = dept_stress.sort_values(by='Stress_Level', ascending=False)
    
    
    fig = px.bar(
        dept_stress,
        x='Department',
        y='Stress_Level',
        color='Stress_Level',
        color_continuous_scale='RdYlGn_r',
        title='Average Stress Level by Department'
    )
    
   
    fig.update_layout(
        xaxis_title='Department',
        yaxis_title='Average Stress Level'
    )
    
    return fig

def create_heatmap(df, x_var, y_var, z_var='Stress_Level'):
    """
    Create a heatmap visualization
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataset
    x_var : str
        Variable for x-axis
    y_var : str
        Variable for y-axis
    z_var : str
        Variable for color (default: 'Stress_Level')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization figure
    """
    
    if df[x_var].dtype == 'object' or df[x_var].dtype == 'bool' or df[x_var].nunique() < 11:
        
        x_categories = df[x_var].unique()
    else:
        
        x_min, x_max = df[x_var].min(), df[x_var].max()
        bin_range = x_max - x_min
        if bin_range <= 10:
            
            bins = np.arange(int(x_min), int(x_max) + 2)
        else:
            
            bins = 10
        
        df[f'{x_var}_binned'] = pd.cut(df[x_var], bins=bins)
        x_var = f'{x_var}_binned'
        x_categories = df[x_var].cat.categories.tolist()
    
    if df[y_var].dtype == 'object' or df[y_var].dtype == 'bool' or df[y_var].nunique() < 11:
        
        y_categories = df[y_var].unique()
    else:
        
        y_min, y_max = df[y_var].min(), df[y_var].max()
        bin_range = y_max - y_min
        if bin_range <= 10:
            
            bins = np.arange(int(y_min), int(y_max) + 2)
        else:
            
            bins = 10
        
        df[f'{y_var}_binned'] = pd.cut(df[y_var], bins=bins)
        y_var = f'{y_var}_binned'
        y_categories = df[y_var].cat.categories.tolist()
    
    
    heatmap_data = df.groupby([x_var, y_var])[z_var].mean().reset_index()
    
   
    pivot_data = heatmap_data.pivot(index=y_var, columns=x_var, values=z_var)
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x=x_var, y=y_var, color=z_var),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='RdYlGn_r',
        title=f'Heatmap of {z_var} by {x_var} and {y_var}'
    )
    
   
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title=y_var
    )
    
    return fig
