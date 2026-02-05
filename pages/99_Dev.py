import streamlit as st
import glob
import os
import pandas as pd
import sqlite3

# Import core modules
from src.core.config_manager import (
    get_base_paths,
    resolve_data_path,
    detect_dataset_type,
    get_codebook_section,
    EXCLUDED_METRIC_COLS,
    EXCLUDED_METRIC_PREFIXES,
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

sql = """
SELECT * FROM df_decode
"""

if st.button("Update Data"):
    conn = sqlite3.connect("temp.db")
    df_decode.to_sql("df_decode", conn, if_exists="replace", index=False)


conn = sqlite3.connect("temp.db")
df = pd.read_sql(sql, conn)

gids = df["GID"].unique()
gids_table_query = """
SELECT 
    GID
    , CASE WHEN MAX(DATA_YR) IS NOT NULL THEN 'DATA_YR' ELSE NULL END AS has_data_yr
    , CASE WHEN MAX(COUNTY) IS NOT NULL THEN 'COUNTY' ELSE NULL END AS has_county
    , CASE WHEN MAX(TOWN) IS NOT NULL THEN 'TOWN' ELSE NULL END AS has_town
    , CASE WHEN MAX(AREA_ALL_SUM_PING_GROUP) IS NOT NULL THEN 'AREA_ALL_SUM_PING_GROUP' ELSE NULL END AS has_area
    , CASE WHEN MAX(HOU_DEPR_YEAR_OLD_GROUP) IS NOT NULL THEN 'HOU_DEPR_YEAR_OLD_GROUP' ELSE NULL END AS has_depr
    , CASE WHEN MAX(STRUC_MAX_GROUP) IS NOT NULL THEN 'STRUC_MAX_GROUP' ELSE NULL END AS has_struc
    , CASE WHEN MAX(HOU_FLOOR_MAX_GROUP) IS NOT NULL THEN 'HOU_FLOOR_MAX_GROUP' ELSE NULL END AS has_floor
FROM df_decode
GROUP BY GID
"""

gids_table = pd.read_sql(gids_table_query, conn)
st.dataframe(gids_table)
st.sidebar.selectbox("選擇 GID", gids, key="gid")
filtered_df = df[df["GID"] == st.session_state.gid]

# 遮蔽的組數
total_count = filtered_df.shape[0]
disabled_count = filtered_df[filtered_df["DATA_STATUS"] == "遮蔽"].shape[0]
st.write(f"遮蔽的組數: {disabled_count}")
st.write(f"總組數: {total_count}")
st.write(f"遮蔽比例: {disabled_count / total_count * 100:.2%}")

# 選擇欄位做分析
pivot_row = st.sidebar.multiselect("選擇行", filtered_df.columns)
pivot_col = st.sidebar.multiselect("選擇列", filtered_df.columns)
pivot_sum = st.sidebar.multiselect("選擇總和", filtered_df.columns)

st.dataframe(filtered_df[filtered_df["DATA_STATUS"] == "正常"])
st.dataframe(filtered_df[filtered_df["DATA_STATUS"] == "遮蔽"])

if st.sidebar.button("查詢"):
    pivot_table = pd.pivot_table(
        filtered_df, values=pivot_sum, index=pivot_row, columns=pivot_col
    )
    st.dataframe(pivot_table)
