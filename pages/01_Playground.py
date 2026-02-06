"""
互動區 - Interactive Pivot Table Analysis Page

This page provides an interactive interface for exploring data through
customizable pivot tables with filtering and year-over-year analysis.
"""

import streamlit as st
import glob
import os
import ast
import pandas as pd

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
    # render_growth_analysis,
)
from src.core.saved_pivots_manager import (
    load_saved_pivots,
    save_pivot_config,
    delete_pivot_config,
)
from src.core.data_registry_manager import get_available_files


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


# ========================= SAVED CONFIGURATIONS =========================
st.sidebar.header("快速讀取")
saved_configs = load_saved_pivots()

if saved_configs:

    def format_config_option(idx):
        c = saved_configs[idx]
        chapter = c.get("chapter", "").strip()
        name = c.get("name", "未命名")
        if chapter:
            return f"[{chapter}] {name}"
        return name

    selected_config_idx = st.sidebar.selectbox(
        "選擇設定",
        options=range(len(saved_configs)),
        format_func=format_config_option,
        key="saved_config_selector",
        index=None,
        placeholder="請選擇...",
    )

    c_load, c_del = st.sidebar.columns([3, 1])

    if c_load.button("讀取 (Load)", type="primary", use_container_width=True):
        if selected_config_idx is not None:
            # Get config by index
            config = saved_configs[selected_config_idx]
            if config:
                # 1. Update Data Source
                # We need to ensure the data source exists.
                # Since file_list is defined later, we might need to move file_list scanning up or just check existence.
                # Actually, file_list logic is below. Let's move file_list scanning up or rely on config value.
                # Better to move file_list scanning up to allow validation.

                # We will set session state, and rely on the page flow to handle it.
                st.session_state["data_source_selector"] = config["data_source"]

                # 2. Update Pivot Selectors
                st.session_state["pivot_tab"] = config["pivot_tab"]
                st.session_state["pivot_row"] = config["pivot_row"]
                st.session_state["pivot_col"] = config["pivot_col"]
                st.session_state["pivot_sum"] = config["pivot_sum"]

                # 3. Update Filters
                if "filters" in config:
                    filter_val = config["filters"]
                    if isinstance(filter_val, str):
                        try:
                            filter_val = ast.literal_eval(filter_val)
                        except Exception:
                            filter_val = {}

                    if isinstance(filter_val, dict):
                        for col, vals in filter_val.items():
                            st.session_state[col] = vals

                # 4. Handle Focus Tab
                if config.get("focus_tab"):
                    ptab = config["pivot_tab"]
                    st.session_state[ptab] = [config["focus_tab"]]

                # 5. Trigger Auto Run
                st.session_state["auto_run_query"] = True

                st.success(f"已載入：{config['name']}")
                st.rerun()

    if c_del.button("❌", help="刪除此設定"):
        if selected_config_idx is not None:
            delete_pivot_config(selected_config_idx)
            st.rerun()
else:
    st.sidebar.caption("尚無儲存的設定")

st.sidebar.markdown("---")


# ========================= DATA SELECTION =========================

# Get list of available data files (Filtered by Registry)
file_list = get_available_files()

if not file_list:
    st.error("❌ 找不到任何可用的資料檔案。請檢查 data 目錄或資料管理設定。")
    st.stop()

# Ensure session state has default if not set
if "data_source_selector" not in st.session_state:
    st.session_state["data_source_selector"] = file_list[0] if file_list else None

# Check if current value is valid
if st.session_state["data_source_selector"] not in file_list:
    st.session_state["data_source_selector"] = file_list[0]

raw_data_path = st.selectbox("選擇資料集", file_list, key="data_source_selector")
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
pivot_tab, pivot_row, pivot_col, pivot_sum = render_pivot_selector(chinese_columns)

# Render filter sidebar
render_filter_sidebar(df_decode, chinese_columns)

# Render visual settings
axis = render_visual_settings()


# ========================= VALIDATION =========================
# Check if user selected same dimension for row and column
# Check if user selected same dimension for row and column (intersect)
# pivot_row and pivot_col are now lists
if set(pivot_row).intersection(set(pivot_col)):
    st.info("🔼 請選擇不同的交叉維度（列與欄不能有重複）")
    st.stop()


# ========================= QUERY EXECUTION =========================
# Check if auto-run is requested (from Saved Analysis page)
auto_run = st.session_state.get("auto_run_query", False)
if auto_run:
    st.session_state["auto_run_query"] = False  # Reset immediately

# Check triggers
if st.button("查詢", type="primary") or auto_run:
    try:
        # Prepare filters for caching (hashable tuple)
        current_filter_items = []
        for col in df_decode.columns:
            if col in st.session_state and st.session_state[col]:
                current_filter_items.append((col, tuple(sorted(st.session_state[col]))))
        current_filter_items = tuple(sorted(current_filter_items))

        # Compute pivot tables
        (
            unique_tabs,
            results,
            row_totals,
            col_totals,
            all_totals,
            masked_df,
            ref_totals,
            ref_df_by_tab,
            filtered_df_by_tab,
        ) = compute_pivot_tables(
            df_decode,
            pivot_tab,
            pivot_row,
            pivot_col,
            pivot_sum,
            current_filter_items,
            codebook_sel,
        )

        # Store results in session_state for display and saving
        st.session_state["last_results"] = {
            "unique_tabs": unique_tabs,
            "results": results,
            "row_totals": row_totals,
            "col_totals": col_totals,
            "all_totals": all_totals,
            "masked_df": masked_df,
            "ref_totals": ref_totals,
            "ref_df_by_tab": ref_df_by_tab,
            "filtered_df_by_tab": filtered_df_by_tab,
            "pivot_tab": pivot_tab,
            "pivot_row": pivot_row,
            "pivot_col": pivot_col,
            "pivot_sum": pivot_sum,
            "filters": {
                k: v for k, v in dict(current_filter_items).items()
            },  # Convert back to dict for storage
            "raw_data_path": raw_data_path,  # Store this too
        }

    except Exception as e:
        st.error(f"❌ 計算時發生錯誤：{str(e)}")
        import traceback

        with st.expander("詳細錯誤訊息"):
            st.code(traceback.format_exc())

# ========================= DISPLAY RESULTS =========================
if "last_results" in st.session_state:
    res_data = st.session_state["last_results"]

    # Check if we have any results
    if not res_data["results"] or all(v is None for v in res_data["results"].values()):
        st.warning("⚠️ 沒有符合篩選條件的資料。")
    else:
        # Render pivot table tabs
        render_pivot_tabs(
            res_data["unique_tabs"],
            res_data["results"],
            axis,  # Axis is from current UI, allows dynamic adjusting without requery!
            masked_df=res_data["masked_df"],
            pivot_tab_col=res_data["pivot_tab"],
            chinese_columns=chinese_columns,
            ref_totals=res_data["ref_totals"],
            ref_df_by_tab=res_data["ref_df_by_tab"],
            filtered_df_by_tab=res_data.get("filtered_df_by_tab", {}),
        )

        # ========================= SAVE CONFIGURATION =========================
        st.write("---")

        # --- QUICK SAVE ---
        with st.expander("快速儲存", expanded=True):
            st.caption("自動產生名稱與說明")
            with st.form("quick_save_form"):
                qc1, qc2, qc3 = st.columns([1, 2, 1])
                qs_chapter = qc1.text_input("章節", placeholder="e.g. 4-1-1")
                # Default focus tab to first one if available
                def_focus = (
                    res_data["unique_tabs"][0] if res_data["unique_tabs"] else "(無)"
                )
                qs_focus_tab = qc2.selectbox(
                    "選擇分組", options=res_data["unique_tabs"], index=0
                )
                qs_unit = qc3.selectbox("單位", ["宅", "戶", "人", "坪", "年"], index=0)

                qs_submitted = st.form_submit_button("快速儲存")

                if qs_submitted and qs_chapter:
                    # 1. Generate Name
                    # Format: [Year] 年 [Context] [Row]分佈 - 按[Row]與[Col]區分
                    year = qs_focus_tab

                    # Determine Context based on filters
                    context = "住宅"
                    filters = res_data["filters"]
                    if "OWNED_TYPE" in filters:
                        # filters["OWNED_TYPE"] is tuple
                        vals = filters["OWNED_TYPE"]
                        if "單獨持有" in vals:
                            context = "所有權單獨持有" + context
                        elif "多人持有" in vals:
                            context = "所有權多人持有" + context
                        # Add other known filters if needed

                    # Get dimension names from Codebook
                    # pivot_row is list, take first for name?
                    # codebook_sel is available in local scope
                    def get_dim_name(dim):
                        return codebook_sel.get(dim, {}).get("name", dim)

                    row_name = (
                        get_dim_name(res_data["pivot_row"][0])
                        if res_data["pivot_row"]
                        else "總計"
                    )
                    col_name = (
                        get_dim_name(res_data["pivot_col"][0])
                        if res_data["pivot_col"]
                        else "總計"
                    )

                    auto_name = f"{year} 年{context}{row_name}分佈 - 按{row_name}與{col_name}區分"

                    # 2. Generate Description (Masking Info)
                    auto_desc = ""
                    if "masked_df" in res_data and res_data["masked_df"] is not None:
                        # masked_df contains the actual rows that are masked.
                        # We want the count of these groups (rows).
                        masked_count = len(res_data["masked_df"])
                        if masked_count > 0:
                            # Try to get reference total if possible
                            # ref_totals might serve as a proxy for "Total households before filtering" if defined that way?
                            # User text: （而未經篩選及遮蔽的本分組總計應為: 8,322,725 宅）
                            # We can try to use all_totals sum?
                            total_count = "N/A"
                            if (
                                "all_totals" in res_data
                                and res_data["all_totals"] is not None
                            ):
                                # Sum of all totals (Grand Total)
                                # all_totals is a Series of totals for each tab? Or single?
                                # In pivot_engine, all_totals is usually Series indexed by metrics.
                                # Let's assume CNT metric.
                                try:
                                    if "CNT" in res_data["all_totals"]:
                                        total_cnt = res_data["all_totals"]["CNT"]
                                        total_count = f"{int(total_cnt):,}"
                                except:
                                    pass

                            auto_desc = f"此分組包含 {masked_count} 組被遮蔽資料（樣本數小於 3 筆）。"
                            # Note: The "Reference Total" part is complex to get exactly right without specific logic.
                            # We omit the exact reference total sentence unless we are sure what it refers to.
                            # Just masking info is safer.

                    # 3. Save
                    new_config = {
                        "name": auto_name,
                        "chapter": qs_chapter,
                        "description": auto_desc,
                        "unit": qs_unit,
                        "data_source": res_data["raw_data_path"],
                        "pivot_tab": res_data["pivot_tab"],
                        "pivot_row": res_data["pivot_row"],
                        "pivot_col": res_data["pivot_col"],
                        "pivot_sum": res_data["pivot_sum"],
                        "filters": res_data["filters"],
                        "focus_tab": str(qs_focus_tab),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }

                    save_pivot_config(new_config)
                    st.success(
                        f"✅ 已快速儲存：\n名稱：{auto_name}\n說明：{auto_desc or '(無)'}\n單位：{qs_unit}"
                    )

        with st.expander("儲存表格 (自訂名稱)"):
            with st.form("save_config_form"):
                # Adjusted layout for unit
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                config_name = c1.text_input(
                    "設定名稱", placeholder="e.g. 112年台北市分析"
                )
                folder_code = c2.text_input("章節", placeholder="e.g. 4-1-1")
                focus_tab = c3.selectbox(
                    "預設聚焦分組", ["(無)"] + [str(t) for t in res_data["unique_tabs"]]
                )
                unit_val = c4.selectbox("單位", ["宅", "戶", "人", "坪", "年"], index=0)

                config_desc = st.text_area(
                    "說明", placeholder="請輸入關於此分析的說明..."
                )

                submitted = st.form_submit_button("儲存 (Save)")
                if submitted and config_name:
                    # Construct config object
                    new_config = {
                        "name": config_name,
                        "chapter": folder_code,
                        "description": config_desc,
                        "unit": unit_val,
                        "data_source": res_data["raw_data_path"],
                        "pivot_tab": res_data["pivot_tab"],
                        "pivot_row": res_data["pivot_row"],
                        "pivot_col": res_data["pivot_col"],
                        "pivot_sum": res_data["pivot_sum"],
                        "filters": res_data["filters"],
                        "focus_tab": focus_tab if focus_tab != "(無)" else None,
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }

                    save_pivot_config(new_config)
                    st.success(f"已儲存設定：{config_name} (章節: {folder_code})")

                    # Keep results visible, do not rerun necessarily or just show success
                    # st.rerun() # Optional

        # Calculate growth rates with current data
        overall_growth_df, row_growth_df, col_growth_df = calculate_growth_rates(
            res_data["row_totals"],
            res_data["col_totals"],
            res_data["all_totals"],
            pivot_tab_name=res_data["pivot_tab"],
        )

        # Render growth analysis (commented out as in original)
        # render_growth_analysis(...)
