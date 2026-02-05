"""
資料源管理 - Manage Data Sources

This page allows administrators to enable or disable (shelf) specific data files
from the Playground selection list.
"""

import streamlit as st
import os
import pandas as pd
from src.core.data_registry_manager import get_all_data_files, load_disabled_files, toggle_file_status

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Manage Data",
    page_icon="🗄️",
    layout="wide",
)

st.title("資料源管理")
st.markdown("在此管理 **Playground** 中可選用的資料檔案。被關閉 (Disable) 的檔案將不會出現在資料選單中，但檔案本身在硬碟上仍安全保留。")

# ========================= LIST FILES =========================
all_files = get_all_data_files()
disabled_set = load_disabled_files()

if not all_files:
    st.warning("找不到任何資料檔案 (data/*.parquet)。")
    st.stop()

# Pre-calculate metadata for sorting
file_data = []
import datetime

for fpath in all_files:
    try:
        mtime = os.path.getmtime(fpath)
        dt_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    except:
        mtime = 0
        dt_str = "Unknown"
        
    file_data.append({
        "path": fpath,
        "name": os.path.basename(fpath),
        "mtime": mtime,
        "dt_str": dt_str
    })

# Sorting Controls
c_sort, _ = st.columns([1, 1])
sort_opt = c_sort.radio(
    "", 
    ["修改時間 (新→舊)", "檔案名稱 (A→Z)", "檔案名稱 (Z→A)"], 
    horizontal=True
)

if sort_opt == "修改時間 (新→舊)":
    file_data.sort(key=lambda x: x["mtime"], reverse=True)
elif sort_opt == "檔案名稱 (A→Z)":
    file_data.sort(key=lambda x: x["name"])
else: # Z->A
    file_data.sort(key=lambda x: x["name"], reverse=True)

st.markdown("### 檔案清單")

# Create a container/table layout
header_cols = st.columns([4, 2, 2, 1])
header_cols[0].markdown("**檔案名稱**")
header_cols[1].markdown("**路徑**")
header_cols[2].markdown("**最後修改時間**")
header_cols[3].markdown("**狀態**")
st.markdown("---")

for item in file_data:
    fname = item["name"]
    fpath = item["path"]
    dt_str = item["dt_str"]
    
    rel_path = os.path.relpath(fpath, start=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    is_disabled = fname in disabled_set
    is_enabled = not is_disabled
    
    cols = st.columns([4, 2, 2, 1])
    cols[0].markdown(f"`{fname}`")
    cols[1].caption(rel_path)
    cols[2].caption(dt_str)
    
    # Toggle switch
    new_status = cols[3].toggle("啟用", value=is_enabled, key=f"toggle_{fname}", label_visibility="collapsed")
    
    # Check if status changed
    if new_status != is_enabled:
        toggle_file_status(fname, new_status)
        if new_status:
            st.toast(f"✅ 已啟用: {fname}")
        else:
            st.toast(f"🚫 已停用: {fname}")
        # Rerun to update state properly if needed, although toggle visual updates instantly.
        # But we need backend to save. (Already called toggle_file_status inside loop? No, that would fire on every render if condition met?)
        # Logic issue: If I just reload, `is_enabled` will match file state. 
        # `new_status` is the WIDGET state.
        # If I change widget, `new_status` differs from `is_enabled` (loaded from file).
        # So I call `toggle_file_status`.
        # Then I should rerun to sync `is_enabled` with the file system for next render.
        import time
        time.sleep(0.5) # Give slight delay for UX
        st.rerun()

st.markdown("---")
st.markdown("*修改狀態後，請回到 Playground 刷新頁面即可看到變更。*")
