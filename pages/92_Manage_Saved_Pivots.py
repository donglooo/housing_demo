"""
管理儲存設定 - Manage Saved Pivots

This page provides a spreadsheet-like interface to batch edit saved pivot table configurations.
"""

import streamlit as st
import pandas as pd
import os

from src.core.saved_pivots_manager import load_saved_pivots, save_all_pivots
from src.core.exporter import export_all_pivots_to_excel

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="管理儲存設定",
    page_icon="⚙️",
    layout="wide",
)

st.title("管理儲存設定")

# ========================= HELPERS =========================
def load_data():
    pivots = load_saved_pivots()
    # Add index to preserve original order or ID
    for i, p in enumerate(pivots):
        p["_id"] = i
    return pivots

# ========================= EXPORT LOGIC =========================
if "export_data" not in st.session_state:
    st.session_state["export_data"] = None

def generate_export():
    pivots = load_saved_pivots()
    if not pivots:
        st.warning("沒有可匯出的設定。")
        return
    try:
        data = export_all_pivots_to_excel(pivots)
        st.session_state["export_data"] = data
        st.toast("✅ 報表產生成功！請點擊下載。", icon="📥")
    except Exception as e:
        st.error(f"匯出失敗: {e}")

# Sidebar Actions
st.sidebar.markdown("### 批次操作")
if st.sidebar.button("產生 Excel 報表"):
    with st.spinner("正在執行查詢與產生報表... (可能需要幾秒鐘)"):
        generate_export()

if st.session_state["export_data"]:
    st.sidebar.download_button(
        label="下載 Excel 報表",
        data=st.session_state["export_data"],
        file_name=f"housing_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ========================= LOAD CONFIGS =========================
if "editor_key" not in st.session_state:
    st.session_state["editor_key"] = 0

try:
    saved_pivots_list = load_data()
except Exception as e:
    st.error(f"無法載入設定檔: {e}")
    saved_pivots_list = []

if not saved_pivots_list:
    st.info("目前沒有儲存的分析設定。請先到 Playground 儲存一些分析結果。")
else:
    st.markdown("""
    **操作說明**：
    1. 直接在下方表格中修改 **章節**、**名稱** 或 **說明**。
    2. 若要刪除，請選取該行並按 Delete 鍵 (或點選刪除列)。
    3. 修改完畢後，**務必點擊「💾 儲存變更」按鈕** 才會生效。
    
    ⚠️ **注意**：不支援在表格中直接「新增」項目（因為缺乏資料源與設定邏輯）。
    """)
    
    # Developer toggle
    c_dev, _ = st.columns([1, 4])
    advanced_mode = c_dev.toggle("進階開發", help="開啟此模式可編輯篩選器、維度等底層設定 (Raw Data)。請小心操作。")

    # Convert to DataFrame
    df = pd.DataFrame(saved_pivots_list)
    
    # Ensure columns exist even if empty
    required_cols = ["chapter", "name", "description", "unit"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Define complex columns that need serialization in Dev mode
    complex_cols = ["filters", "pivot_row", "pivot_col", "pivot_tab", "pivot_sum"]

    if advanced_mode:
        import ast
        # Serialize complex columns for editing
        for col in complex_cols:
             if col in df.columns:
                 # Clean NaN before repr
                 df[col] = df[col].apply(lambda x: repr(x) if pd.notna(x) else "None")
        
        # In dev mode, show most columns, but reorder to put ID/Keys first
        # Hide _id still? No, ID is needed to map rows but user shouldn't edit it? 
        # User wants to manage whole file.
        # Let's show everything roughly in order.
        cols_order = ["chapter", "name", "unit", "description", "data_source"] + complex_cols + ["focus_tab", "timestamp", "_id"]
        # Union with existing cols
        final_cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
        df = df[final_cols]
        
        # Basic config for dev
        col_config = {
            "_id": None, 
            "chapter": st.column_config.TextColumn("chapter", width="small"),
            "name": st.column_config.TextColumn("name", width="medium"),
            "unit": st.column_config.TextColumn("unit", width="small"),
            "description": st.column_config.TextColumn("description", width="large"),
            "data_source": st.column_config.TextColumn("data_source", width="large"),
        }
    else:            
        # Normal Mode
        # Reorder columns for display (Hidden columns still exist in df)
        # We want Chapter, Name, Unit, Description first
        display_cols = ["chapter", "name", "unit", "description", "focus_tab", "timestamp", "_id"]
        # Add other columns to end
        other_cols = [c for c in df.columns if c not in display_cols]
        df = df[display_cols + other_cols]
        
        col_config = {
            "_id": None, # Hide ID
            "chapter": st.column_config.TextColumn("章節", width="small", help="e.g. 4-1-1"),
            "name": st.column_config.TextColumn("名稱", width="medium"),
            "unit": st.column_config.SelectboxColumn("單位", options=["宅", "戶", "人", "坪", "年"], width="small"),
            "description": st.column_config.TextColumn("說明", width="large"),
            "focus_tab": st.column_config.TextColumn("聚焦分組", disabled=True, width="small"),
            "timestamp": st.column_config.TextColumn("建立時間", disabled=True, width="medium"),
            # Hide complex columns
            "data_source": None,
            "pivot_tab": None,
            "pivot_row": None,
            "pivot_col": None,
            "pivot_sum": None,
            "filters": None,
        }

    # Display Data Editor
    edited_df = st.data_editor(
        df,
        column_config=col_config,
        hide_index=True,
        width="stretch",
        num_rows="dynamic", # Allow add/delete
        key=f"pivot_editor_{st.session_state['editor_key']}"
    )

    if st.button("💾 儲存變更 (Save Changes)", type="primary"):
        # Convert back to list of dicts
        
        # 1. Filter out newly added rows that lack essential data (like _id or data_source)
        # Assuming original rows have valid _id. New rows won't have _id (NaN).
        # Actually editor returns new rows with NaN in _id column unless we handle it?
        # Let's check. Yes, if user adds a row, _id will be None/NaN.
        
        valid_rows = []
        dropped_count = 0
        error_msg = None
        
        import ast

        for _, row in edited_df.iterrows():
            record = row.to_dict()
            
            # Check if this is a valid existing record via _id or data_source check?
            # In Dev mode, user might ADD a row and paste a source. We should support that if possible.
            # If "data_source" is filled, it's potentially valid.
            
            ds = record.get("data_source")
            if pd.isna(ds) or ds == "" or ds == "None":
                 # If normal mode, we drop. If dev mode and user didn't fill it, strip.
                 dropped_count += 1
                 continue

            # Parse complex columns back if in Dev Mode (even if logic check says advanced_mode, the DF structure depends on it)
            # But wait, st.button triggers rerun. advanced_mode state is preserved? Yes.
            
            if advanced_mode:
                try:
                    for col in complex_cols:
                        if col in record and isinstance(record[col], str):
                            # Try parse
                            # Use literal_eval to be safe
                            val_str = record[col]
                            if val_str and val_str != "None":
                                record[col] = ast.literal_eval(val_str)
                            else:
                                record[col] = None
                except Exception as e:
                    error_msg = f"Parsing error for row {record.get('name')}: {e}"
                    break


            # Clean up temporary columns
            if "_id" in record:
                del record["_id"]
                
            # Clean up NaN values (Editor might introduce NaN for empty text)
            for k, v in record.items():
                if pd.isna(v):
                    record[k] = None if k == "focus_tab" else "" # specific defaults?
            
            valid_rows.append(record)
        
        if error_msg:
             st.error(f"❌ 儲存失敗: {error_msg}")
        elif valid_rows:
            save_all_pivots(valid_rows)
            st.success("✅ 更新成功！")
            if dropped_count > 0:
                st.warning(f"⚠️ 已忽略 {dropped_count} 筆資料 (缺少 data_source)。")
            
            # Refresh to update editor state (resetting ID)
            st.session_state["editor_key"] += 1
            import time
            time.sleep(0.5)
            st.rerun()
        else:
            if dropped_count > 0:
                 st.error("⚠️ 無有效資料可儲存 (所有列皆為無效新增)。請確保 data_source 不為空。")
            else:
                 # Empty list means user deleted everything
                 save_all_pivots([])
                 st.success("✅ 已清空所有設定。")
                 st.session_state["editor_key"] += 1
                 st.rerun()
