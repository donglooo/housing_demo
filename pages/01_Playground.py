"""
互動區 - Interactive Pivot Table Analysis Page

This page provides an interactive interface for exploring data through
customizable pivot tables with filtering and year-over-year analysis.
"""

import streamlit as st
import glob
import os

# Import core modules
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


# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Playground",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ========================= INITIALIZE PATHS =========================
BASE_DIR, DATA_DIR, CONFIG_DIR = get_base_paths()
CODEBOOK_PATH = os.path.join(CONFIG_DIR, "codebook.yaml")


# ========================= DATA SELECTION =========================

# Get list of available data files
file_list = glob.glob(os.path.join(DATA_DIR, "**", "*.parquet"), recursive=True)

if not file_list:
    st.error("❌ 找不到任何資料檔案。請檢查 data 目錄。")
    st.stop()

raw_data_path = st.selectbox("選擇資料集", file_list)
DATA_PATH = resolve_data_path(raw_data_path, BASE_DIR)


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


# ========================= UI CONTROLS =========================

# Render pivot dimension selectors
pivot_row, pivot_col, pivot_sum = render_pivot_selector(chinese_columns)

# Render filter sidebar
render_filter_sidebar(df_decode, chinese_columns)

# Render visual settings
axis = render_visual_settings()


# ========================= VALIDATION =========================
# Check if user selected same dimension for row and column
if pivot_row == pivot_col:
    st.info("🔼 請選擇不同的交叉維度")
    st.stop()


# ========================= QUERY EXECUTION =========================
if st.button("查詢", type="primary"):
    try:
        # Prepare filters for caching (hashable tuple)
        current_filter_items = []
        for col in df_decode.columns[1:-1]:
            if col in st.session_state and st.session_state[col]:
                current_filter_items.append((col, tuple(sorted(st.session_state[col]))))
        current_filter_items = tuple(sorted(current_filter_items))

        # Compute pivot tables
        unique_years, results, row_totals_year, col_totals_year, all_totals_year = (
            compute_pivot_tables(
                df_decode,
                pivot_row,
                pivot_col,
                pivot_sum,
                current_filter_items,
                codebook_sel,
            )
        )

        # Check if we have any results
        if not results or all(v is None for v in results.values()):
            st.warning("⚠️ 沒有符合篩選條件的資料。請調整篩選條件。")
            st.stop()

        # Render pivot table tabs
        render_pivot_tabs(unique_years, results, axis)

        # Calculate growth rates
        overall_growth_df, row_growth_df, col_growth_df = calculate_growth_rates(
            row_totals_year, col_totals_year, all_totals_year
        )

        # Render growth analysis
        render_growth_analysis(
            overall_growth_df, row_growth_df, col_growth_df, pivot_row, pivot_col
        )

        # st.success("✅ 分析完成！")

    except Exception as e:
        st.error(f"❌ 計算時發生錯誤：{str(e)}")
        import traceback

        with st.expander("詳細錯誤訊息"):
            st.code(traceback.format_exc())
