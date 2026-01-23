import pandas as pd
import streamlit as st

@st.cache_data
def load_csv(file):
    """
    Reads a CSV file into a pandas DataFrame.
    Assumes standard CSV format with comma delimiter.
    """
    try:
        # User uploaded file or local path
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None
