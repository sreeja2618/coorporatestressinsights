import streamlit as st
import os

def create_navbar():
    """
    Create a horizontal navbar for all pages
    """
    # Custom CSS for the navbar
    navbar_css = """
    <style>
    .navbar {
        display: flex;
        justify-content: space-around;
        padding: 10px;
        background-color: #1E3A8A;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .navbar a {
        color: white;
        text-decoration: none;
        padding: 8px 16px;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .navbar a:hover {
        background-color: #2563EB;
    }
    .navbar a.active {
        background-color: #3B82F6;
    }
    </style>
    """
    st.markdown(navbar_css, unsafe_allow_html=True)
    
    # Define all pages with their paths
    pages = {
        "Home": "",
        "Demographics": "1_Demographics_Analysis",
        "Stress Factors": "2_Stress_Factors",
        "Departments": "3_Department_Analysis",
        "Correlations": "4_Correlation_Analysis",
        "Predictions": "5_Predictive_Insights"
    }
    
    # Get current page
    try:
        # Get the script path from the session state
        current_file = os.path.basename(__file__)
        if current_file == "navbar.py":
            # We're in the navbar module, get the caller's path
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_file = os.path.basename(caller_frame.f_code.co_filename)
            current_page = caller_file
        else:
            # Fallback to this file
            current_page = current_file
    except:
        # If all else fails, assume we're on the main page
        current_page = "main.py"
    
    # Build HTML for navbar
    navbar_html = '<div class="navbar">'
    for page_name, page_path in pages.items():
        is_active = current_page == page_path
        active_class = 'class="active"' if is_active else ''
        
        if page_name == "Home" and current_page == "main.py":
            active_class = 'class="active"'
            
        if page_path:
            navbar_html += f'<a href="/{page_path}" {active_class}>{page_name}</a>'
        else:
            navbar_html += f'<a href="/" {active_class}>{page_name}</a>'
            
    navbar_html += '</div>'
    
    st.markdown(navbar_html, unsafe_allow_html=True)