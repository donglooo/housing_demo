import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

st.set_page_config(page_title="DEV", layout="wide")

st.title("DEV")

# Load data
df = pd.read_parquet("data/260204/稅籍_coded_202602041056.parquet")

# Show data
st.write(df.head(1000))

# 先建立 Database
create_db = st.button("建立 Database")

if create_db:
    conn = sqlite3.connect("data/260204/housing_research.db")
    df.to_sql("tax", conn, if_exists="replace", index=False)
    conn.close()
    st.write("Database建立成功")

# 選取資料
default_query = """
    SELECT 
        GID
        , CASE WHEN MAX(DATA_YR) IS NOT NULL THEN '年份' ELSE NULL END AS has_data_yr
        , CASE WHEN MAX(COUNTY) IS NOT NULL THEN '縣市' ELSE NULL END AS has_county
        , CASE WHEN MAX(TOWN) IS NOT NULL THEN '鄉鎮' ELSE NULL END AS has_town
        , CASE WHEN MAX(AREA_ALL_SUM_PING_GROUP) IS NOT NULL THEN '面積' ELSE NULL END AS has_area
        , CASE WHEN MAX(HOU_DEPR_YEAR_OLD_GROUP) IS NOT NULL THEN '屋齡' ELSE NULL END AS has_age
        , CASE WHEN MAX(STRUC_MAX_GROUP) IS NOT NULL THEN '結構' ELSE NULL END AS has_struc
        , CASE WHEN MAX(HOU_FLOOR_MAX_GROUP) IS NOT NULL THEN '樓層' ELSE NULL END AS has_floor
        , COUNT(*) AS cnt
    FROM tax
    GROUP BY GID
    ORDER BY GID
"""
query = st.text_area("SQL Query", height=200, value=default_query)

if query:
    conn = sqlite3.connect("data/260204/housing_research.db")
    df = pd.read_sql(query, conn)
    conn.close()
    st.write(df)

    edited_df = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key="mapping_editor"
    )

# 先將 yaml 檔改為一張 table 建立
