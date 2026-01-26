"""
Data loading and decoding utilities.

This module handles loading data from Parquet files, loading codebooks,
and decoding data using codebook mappings.
"""

import pandas as pd
import yaml
import streamlit as st
from typing import Dict


@st.cache_data
def load_codebook(codebook_path: str) -> Dict:
    """
    Load and parse YAML codebook.

    Args:
        codebook_path: Path to the codebook YAML file

    Returns:
        Parsed codebook dictionary
    """
    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = yaml.safe_load(f)
    return codebook


@st.cache_data(ttl=1800, max_entries=3)
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load Parquet data with caching.

    Args:
        data_path: Path to the Parquet file

    Returns:
        DataFrame with loaded data
    """
    df = pd.read_parquet(data_path, engine="pyarrow")
    return df


@st.cache_data
def decode_data(df: pd.DataFrame, codebook: Dict) -> pd.DataFrame:
    """
    Apply codebook transformations to decode data.

    Replaces coded values with their human-readable equivalents
    based on the codebook mappings.

    Args:
        df: DataFrame with coded values
        codebook: Codebook dictionary with code mappings

    Returns:
        DataFrame with decoded values
    """
    df_decode = df.copy()

    for col in codebook.keys():
        if col in df_decode.columns:
            # Apply code mapping if it exists
            if "codes" in codebook[col]:
                df_decode[col] = df_decode[col].replace(codebook[col]["codes"])

    return df_decode


def get_chinese_columns(codebook: Dict) -> Dict[str, str]:
    """
    Extract Chinese column name mappings from codebook.

    Args:
        codebook: Codebook dictionary

    Returns:
        Dictionary mapping column keys to Chinese names
    """
    chinese_columns = {}

    for col in codebook.keys():
        if "name" in codebook[col]:
            chinese_columns[col] = codebook[col]["name"]
        else:
            # Fallback to column key if no name is defined
            chinese_columns[col] = col

    return chinese_columns
