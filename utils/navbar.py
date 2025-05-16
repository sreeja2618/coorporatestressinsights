import streamlit as st

def create_navbar():
    """
    Create a horizontal navbar for all pages using Streamlit's built-in components
    instead of HTML links which don't work well with Streamlit's routing
    """
    # Custom CSS for the navbar
    navbar_css = """
    <style>
    .navbar-container {
        display: flex;
        justify-content: space-around;
        padding: 10px;
        background-color: #1E3A8A;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .nav-item {
        color: white;
        text-align: center;
        padding: 8px 16px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
    }
    .nav-item:hover {
        background-color: #2563EB;
    }
    .nav-item.active {
        background-color: #3B82F6;
    }
    </style>
    """
    st.markdown(navbar_css, unsafe_allow_html=True)
    cols = st.columns(6)
    page_names = ["Home", "Demographics", "Stress Factors", "Departments", "Correlations", "Predictions"]
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    for i, page_name in enumerate(page_names):
        with cols[i]:
            is_active = st.session_state.page == page_name
            if st.button(page_name, key=f"nav_{page_name}", 
                        use_container_width=True,
                        type="primary" if is_active else "secondary"):
                st.session_state.page = page_name
                st.rerun()
    return st.session_state.page