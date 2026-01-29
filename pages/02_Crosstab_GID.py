import streamlit as st
import pandas as pd
import os
import glob

from src.core.config_manager import (
    get_base_paths,
    resolve_data_path,
    detect_dataset_type,
    get_codebook_section,
)
from src.core.data_loader import (
    load_codebook,
    load_data,
    decode_data,
    get_chinese_columns,
)
from src.core.pivot_engine import compute_pivot_tables, calculate_growth_rates
from src.core.ui_components import (
    render_pivot_selector,
    render_filter_sidebar,
    render_visual_settings,
    render_pivot_tabs,
    render_growth_analysis,
)

# ========================= INITIALIZE PATHS =========================
BASE_DIR, DATA_DIR, CONFIG_DIR = get_base_paths()
CODEBOOK_PATH = os.path.join(CONFIG_DIR, "codebook.yaml")


# ========================= DATA SELECTION =========================

# Get list of available data files
# file_list = glob.glob(os.path.join(DATA_DIR, "**", "*.parquet"), recursive=True)

# if not file_list:
#     st.error("❌ 找不到任何資料檔案。請檢查 data 目錄。")
#     st.stop()

# raw_data_path = st.selectbox("選擇資料集", file_list)
# DATA_PATH = resolve_data_path(raw_data_path, BASE_DIR)
DATA_PATH = r"C:\Users\10523\Desktop\Dong\99_GitHub\housing-research\data\260127\稅籍_coded_202601271258.parquet"

# ========================= LOAD DATA & CODEBOOK =========================
try:
    # Detect dataset type from filename
    dataset_type = detect_dataset_type(DATA_PATH)

    # Load codebook
    codebook = load_codebook(CODEBOOK_PATH)
    codebook_sel = get_codebook_section(codebook, dataset_type)

    # Get Chinese column mappings
    chinese_columns = get_chinese_columns(codebook_sel)

    # Load and decode data
    df = load_data(DATA_PATH)
    df_decode = decode_data(df, codebook_sel)

except Exception as e:
    st.error(f"❌ 載入資料時發生錯誤：{str(e)}")
    st.stop()

st.dataframe(df[~df["COUNTY"].isna()])
st.dataframe(df_decode)
