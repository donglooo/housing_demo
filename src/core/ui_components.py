"""
Reusable Streamlit UI components.

This module provides reusable UI components for building
pivot table interfaces in Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional

from src.core.config_manager import (
    EXCLUDED_METRIC_PREFIXES,
    EXCLUDED_METRIC_COLS,
)


@st.cache_data
def get_unique_values(df: pd.DataFrame) -> Dict[str, List]:
    """
    Get unique values for all columns in the DataFrame with caching.

    This avoids recalculating .unique() on large DataFrames on every
    script rerun, which is a major performance bottleneck.

    Args:
        df: Input DataFrame

    Returns:
        Dict mapping column names to sorted list of unique values
    """
    unique_vals = {}
    for col in df.columns:
        # Filter out NaN/None values and sort
        unique_vals[col] = sorted([x for x in df[col].unique() if pd.notna(x)], key=str)
    return unique_vals


def render_pivot_selector(chinese_columns: Dict[str, str]) -> tuple:
    """
    Render row/column/sum selector controls.

    Args:
        chinese_columns: Dict mapping column keys to Chinese names

    Returns:
        Tuple of (pivot_row, pivot_col, pivot_sum) selected values
    """
    pivot_tab_col, pivot_row_col, pivot_col_col, pivot_sum_col = st.columns(4)

    opts = list(chinese_columns.keys())

    def get_label(key):
        return chinese_columns.get(key, key)

    with pivot_tab_col:
        # Initialize default if not set
        if "pivot_tab" not in st.session_state and opts:
            st.session_state["pivot_tab"] = opts[0]

        p_tab = st.selectbox(
            "分組依據(Tab)", opts, format_func=get_label, key="pivot_tab"
        )

    with pivot_row_col:
        # Initialize default if not set
        if "pivot_row" not in st.session_state:
            st.session_state["pivot_row"] = [opts[1]] if len(opts) > 1 else [opts[0]]

        p_row = st.multiselect(
            "列維度(Row)",
            opts,
            # No default arg, use session_state exclusively
            format_func=get_label,
            key="pivot_row",
            max_selections=2,
        )
        if not p_row:
            p_row = [opts[1]] if len(opts) > 1 else [opts[0]]

    with pivot_col_col:
        # Initialize default if not set
        if "pivot_col" not in st.session_state:
            st.session_state["pivot_col"] = []

        p_col = st.multiselect(
            "欄維度(Column)",
            opts,
            # No default arg
            format_func=get_label,
            key="pivot_col",
            max_selections=2,
        )
        if not p_col:
            p_col = []

    with pivot_sum_col:
        if "pivot_sum" not in st.session_state:
            st.session_state["pivot_sum"] = "CNT"

        p_sum = st.selectbox("計算欄", ["CNT"], key="pivot_sum")

    return p_tab, p_row, p_col, p_sum


def render_filter_sidebar(
    df_decode: pd.DataFrame, chinese_columns: Dict[str, str]
) -> None:
    """
    Render filter multiselect controls in sidebar.

    Args:
        df_decode: Decoded DataFrame
        chinese_columns: Dict mapping column keys to Chinese names
    """
    st.sidebar.header("篩選器")

    def get_label(key):
        return chinese_columns.get(key, key)

    # Get cached unique values to prevent reloading entire column data
    unique_values = get_unique_values(df_decode)

    # Define columns/prefixes to exclude from filters
    # These are metrics (SUM, CNT, AVG) or identifiers (GID) or internal status (DATA_STATUS)
    # now imported from config_manager

    filters = [
        col
        for col in df_decode.columns
        if not col.startswith(EXCLUDED_METRIC_PREFIXES)
        and col not in EXCLUDED_METRIC_COLS
    ]

    for col in filters:
        opts = unique_values.get(col, [])
        st.sidebar.multiselect(get_label(col), opts, key=col)


def render_visual_settings() -> Optional[int]:
    """
    Render visualization settings (axis selection).

    Returns:
        Axis value for gradient: None (全表), 0 (直向), or 1 (橫向)
    """
    st.sidebar.header("Visual Settings")

    check_visual_axis = st.sidebar.radio(
        "視覺化比較軸向", ["全表", "直向", "橫向"], index=0, horizontal=True
    )

    axis_options = {
        "全表": None,  # 適合：找全域最大/最小值
        "直向": 0,  # 適合：比較同一月份各縣市的表現
        "橫向": 1,  # 適合：比較同一縣市不同坪數的分布
    }

    return axis_options[check_visual_axis]


def render_pivot_tabs(
    unique_keys: List[any],
    results: Dict[any, Dict],
    axis: Optional[int],
    masked_df: Optional[pd.DataFrame] = None,
    pivot_tab_col: Optional[str] = None,
    chinese_columns: Optional[Dict] = None,
    ref_totals: Optional[Dict] = None,
    ref_df_by_tab: Optional[Dict] = None,
    filtered_df_by_tab: Optional[Dict] = None,
) -> None:
    """
    Render pivot table tabs with styling.

    Args:
        unique_keys: Sorted list of tab keys (group values)
        results: Dict mapping key to pivot results
        axis: Gradient axis (None, 0, or 1)
        masked_df: Optional DataFrame containing masked data
        pivot_tab_col: Column name identifying the tab dimension
        chinese_columns: Dict of column key -> Chinese name
        ref_totals: Optional dict of tab -> total count
        ref_df_by_tab: Optional dict of tab -> DataFrame containing reference data for that tab
    """
    tabs = st.tabs([str(k) for k in unique_keys])

    for i, tab_key in enumerate(unique_keys):
        with tabs[i]:
            # --- Check and Display Masked Data for this Tab ---
            if (
                masked_df is not None
                and pivot_tab_col is not None
                and not masked_df.empty
            ):
                # Filter masked rows relevant to this tab
                # Handle type mismatches carefully (convert to string if needed, or rely on pandas)
                try:
                    # Try direct comparison first
                    local_masked = masked_df[masked_df[pivot_tab_col] == tab_key]
                except Exception:
                    # Fallback to string comparison
                    local_masked = masked_df[
                        masked_df[pivot_tab_col].astype(str) == str(tab_key)
                    ]

                if not local_masked.empty:
                    msg = f"ℹ️ 此分組包含 {len(local_masked)} 組被遮蔽資料（樣本數小於 3 筆）。"

                    # Display calculation details in container
                    if ref_totals:
                        with st.expander("🔍 查看參考總計計算明細", expanded=False):
                            st.write("**參考總計字典 (ref_totals):**")
                            st.json(ref_totals)

                            st.write(
                                f"**當前分組鍵 (tab_key):** `{tab_key}` (型別: `{type(tab_key).__name__}`)"
                            )

                            # Try exact or str match for key
                            val = ref_totals.get(tab_key)

                            st.write("---")
                            st.write("**查找過程:**")
                            if val is not None:
                                st.success(
                                    f"✅ 直接匹配成功: `ref_totals[{tab_key}]` = {val:,.0f}"
                                )
                            else:
                                st.warning(
                                    f"⚠️ 直接匹配失敗: `ref_totals.get({tab_key})` 返回 None"
                                )
                                st.write("**嘗試字串匹配:**")

                                # Try str match scan
                                match_found = False
                                for k, v in ref_totals.items():
                                    is_match = str(k) == str(tab_key)
                                    if is_match:
                                        val = v
                                        match_found = True
                                        st.success(
                                            f"✅ 字串匹配成功: `str({k})` == `str({tab_key})` → 值 = {v:,.0f}"
                                        )
                                        break
                                    else:
                                        st.write(
                                            f"   ❌ `str({k})` != `str({tab_key})`"
                                        )

                                if not match_found:
                                    st.error("❌ 未找到匹配的鍵值")

                            if val is not None:
                                st.write(f"**最終結果:** {val:,.0f} 宅")

                        if val is not None:
                            msg += (
                                f"（而未經篩選及遮蔽的本分組總計應為: {val:,.0f} 宅）"
                            )

                    st.info(msg)
                    with st.expander(
                        f"查看共有 {len(local_masked)} 組被遮蔽的資料細節"
                    ):
                        mask_details = []
                        col_map = chinese_columns if chinese_columns else {}

                        for idx, row in local_masked.iterrows():
                            dims = []
                            for col in local_masked.columns:
                                if (
                                    col not in EXCLUDED_METRIC_COLS
                                    and not str(col).startswith(
                                        EXCLUDED_METRIC_PREFIXES
                                    )
                                    and pd.notna(row[col])
                                ):
                                    col_name = col_map.get(col, col)
                                    val = row[col]
                                    dims.append(f"{col_name}: {val}")

                            if dims:
                                mask_details.append(" | ".join(dims))

                        if mask_details:
                            st.write(mask_details)
                        else:
                            st.write("無法識別特定的維度組合")
            # --------------------------------------------------

            res = results.get(tab_key)

            if res is None:
                st.warning(f"No data for {tab_key} with current filters.")
                continue

            pivot_table = res["pivot"]
            pivot_table_row = res["row_pct"]
            pivot_table_col = res["col_pct"]
            pivot_table_total = res["total_pct"]

            # Create sub-tabs for different views
            sub_tab0, sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(
                ["MIX", "Pivot Table", "Row(%)", "Col(%)", "Total(%)"]
            )

            # --- Hybrid View Logic ---
            with sub_tab0:
                # 1. Prepare Base DataFrames
                df_val = pivot_table.copy()
                df_pct = pivot_table_row.copy()

                # 2. Interleave columns
                # Move '全國' (Total) to front if exists
                cols = [c for c in df_val.columns if c != "全國"]

                hybrid_df = pd.DataFrame(index=df_val.index)

                # Add Total column first
                if "全國" in df_val.columns:
                    hybrid_df["全國"] = df_val["全國"]

                # Interleave others
                format_map = {}
                # Add Total format
                if "全國" in df_val.columns:
                    format_map["全國"] = "{:,.0f}"

                # Insert Average Columns if available
                # results.get(tab_key) -> 'avg_data'
                avg_data = res.get("avg_data")
                if avg_data:
                    rows_avg = avg_data["rows_avg"]
                    total_avg = avg_data["total_avg"]

                    # avg_data rows_avg index should match pivot table index (excluding Total Row "全國" usually, unless integrated)
                    # total_row_key is "全國" or ("全國", ...)
                    total_row_key = hybrid_df.index[0]

                    for avg_key in rows_avg.columns:
                        # Create a series for the column
                        # 1. Fill with rows_avg data
                        # Align to hybrid_df index (which includes Total Row)
                        s = pd.Series(index=hybrid_df.index, dtype=float)

                        # Assign row values (aligning by index)
                        # rows_avg index should match non-total rows of hybrid_df
                        s.update(rows_avg[avg_key])

                        # Assign Total Row value
                        s.loc[total_row_key] = total_avg.get(avg_key, 0)

                        hybrid_df[avg_key] = s
                        format_map[avg_key] = "{:,.2f}"

                pct_cols = []
                for c in cols:
                    # Value Column
                    val_col_name = f"{c}"
                    hybrid_df[val_col_name] = df_val[c]
                    format_map[val_col_name] = "{:,.0f}"

                    # Pct Column
                    pct_col_name = f"{c}(%)"
                    hybrid_df[pct_col_name] = df_pct[c]
                    format_map[pct_col_name] = "{:.2%}"
                    pct_cols.append(pct_col_name)

                # 3. Display
                # Define gradient subset
                gradient_rows = [idx for idx in pivot_table.index if idx != "全國"]

                # Apply white background to NaN cells first, then gradient to non-NaN
                styled = hybrid_df.style.format(format_map)

                # Apply gradient
                styled = styled.background_gradient(
                    subset=(gradient_rows, pct_cols), cmap="Blues", axis=axis, vmin=0
                )

                # Override NaN cells with white background
                def highlight_nan(val):
                    if pd.isna(val):
                        return "background-color: white"
                    return ""

                styled = styled.applymap(
                    highlight_nan, subset=(gradient_rows, pct_cols)
                )

                st.dataframe(
                    styled,
                    height=int((len(hybrid_df) * 35) + 37),
                )

            # Calculate dynamic height
            dynamic_height = int((len(pivot_table) * 35) + 37)

            # Exclude totals from gradient
            gradient_columns = [col for col in pivot_table.columns if col != "全國"]
            # gradient_rows already defined above

            with sub_tab1:
                # Apply gradient and override NaN with white
                styled = pivot_table.style.background_gradient(
                    subset=(gradient_rows, gradient_columns),
                    cmap="Blues",
                    axis=axis,
                    vmin=0,
                ).format("{:,.0f}")

                def highlight_nan(val):
                    if pd.isna(val):
                        return "background-color: white"
                    return ""

                styled = styled.applymap(
                    highlight_nan, subset=(gradient_rows, gradient_columns)
                )

                st.dataframe(
                    styled,
                    height=dynamic_height,
                )

            with sub_tab2:
                # Apply gradient and override NaN with white
                styled = pivot_table_row.style.background_gradient(
                    subset=(gradient_rows, gradient_columns),
                    cmap="Blues",
                    axis=axis,
                    vmin=0,
                ).format("{:.2%}")

                def highlight_nan(val):
                    if pd.isna(val):
                        return "background-color: white"
                    return ""

                styled = styled.applymap(
                    highlight_nan, subset=(gradient_rows, gradient_columns)
                )

                st.dataframe(
                    styled,
                    height=dynamic_height,
                )

            with sub_tab3:
                # Apply gradient and override NaN with white
                styled = pivot_table_col.style.background_gradient(
                    subset=(gradient_rows, gradient_columns),
                    cmap="Blues",
                    axis=axis,
                    vmin=0,
                ).format("{:.2%}")

                def highlight_nan(val):
                    if pd.isna(val):
                        return "background-color: white"
                    return ""

                styled = styled.applymap(
                    highlight_nan, subset=(gradient_rows, gradient_columns)
                )

                st.dataframe(
                    styled,
                    height=dynamic_height,
                )

            with sub_tab4:
                # Apply gradient and override NaN with white
                styled = pivot_table_total.style.background_gradient(
                    subset=(gradient_rows, gradient_columns),
                    cmap="Blues",
                    axis=axis,
                    vmin=0,
                ).format("{:.2%}")

                def highlight_nan(val):
                    if pd.isna(val):
                        return "background-color: white"
                    return ""

                styled = styled.applymap(
                    highlight_nan, subset=(gradient_rows, gradient_columns)
                )

                st.dataframe(
                    styled,
                    height=dynamic_height,
                )

            # --- Display Source Data for this Tab ---
            if filtered_df_by_tab is not None and tab_key in filtered_df_by_tab:
                with st.expander(
                    "查看此分組的原始明細資料（交叉表數據來源）", expanded=False
                ):
                    tab_filtered_df = filtered_df_by_tab[tab_key]

                    if not tab_filtered_df.empty:
                        st.write(f"**當前分組 ({tab_key}) 的篩選後明細:**")
                        st.caption(
                            "此為經過篩選條件後，用於生成上方交叉表的完整原始資料"
                        )
                        # Show sample or all data
                        if len(tab_filtered_df) > 1000:
                            st.warning(
                                f"資料筆數較多 ({len(tab_filtered_df):,} 筆)，僅顯示前 1000 筆"
                            )
                            st.dataframe(tab_filtered_df.head(1000), width="stretch")
                        else:
                            st.dataframe(tab_filtered_df, width="stretch")
                        st.write(f"**總筆數:** {len(tab_filtered_df):,} 筆")
                        if "CNT" in tab_filtered_df.columns:
                            total_cnt = tab_filtered_df["CNT"].sum()
                            st.write(f"**CNT 總和:** {total_cnt:,.0f}")
                    else:
                        st.info(f"此分組 ({tab_key}) 無明細資料")

                    # Show ref_df as technical detail
                    if ref_df_by_tab is not None and tab_key in ref_df_by_tab:
                        with st.expander("ref_df（參考總計用）", expanded=False):
                            tab_ref_df = ref_df_by_tab[tab_key]
                            st.caption(
                                "此為計算「參考總計」時使用的數據（pivot維度強制為Null的總計列）"
                            )
                            if not tab_ref_df.empty:
                                st.dataframe(tab_ref_df, width="stretch")
                                st.write(f"**筆數:** {len(tab_ref_df)} 筆")
                            else:
                                st.info("無 ref_df 資料")


# def render_growth_analysis(
#     overall_growth_df: Optional[pd.DataFrame],
#     row_growth_df: Optional[pd.DataFrame],
#     col_growth_df: Optional[pd.DataFrame],
#     pivot_row: str,
#     pivot_col: str,
# ) -> None:
#     """
#     Render growth rate analysis tabs.

#     Args:
#         overall_growth_df: Overall growth DataFrame
#         row_growth_df: Row dimension growth DataFrame
#         col_growth_df: Column dimension growth DataFrame
#         pivot_row: Row dimension name
#         pivot_col: Column dimension name
#     """
#     st.markdown("### 年增率分析")

#     growth_tabs = st.tabs(["總體", "列(Row)維度", "欄(Col)維度"])

#     with growth_tabs[0]:
#         if overall_growth_df is not None:
#             col_metric1, col_metric2 = st.columns(2)

#             avg_growth = overall_growth_df["YEARLY_GROWTH_PCT"].mean()
#             latest_growth = overall_growth_df["YEARLY_GROWTH_PCT"].iloc[-1]

#             col_metric1.metric("平均年增率", f"{avg_growth:.2f}%")
#             col_metric2.metric("最新年增率", f"{latest_growth:.2f}%")

#             st.dataframe(
#                 overall_growth_df[
#                     ["DATA_YR", "TOTAL", "YEARLY_GROWTH_PCT"]
#                 ].style.format({"TOTAL": "{:,.0f}", "YEARLY_GROWTH_PCT": "{:,.2f}%"})
#             )
#         else:
#             st.info("無足夠資料計算年增率")

#     with growth_tabs[1]:
#         if row_growth_df is not None:
#             st.write(f"列維度 ({pivot_row}) 年增率 (%):")
#             st.dataframe(
#                 row_growth_df.style.format("{:,.2f}%").background_gradient(
#                     cmap="RdYlBu", vmin=-10, vmax=10
#                 )
#             )
#         else:
#             st.info("無足夠資料計算年增率")

#     with growth_tabs[2]:
#         if col_growth_df is not None:
#             st.write(f"欄維度 ({pivot_col}) 年增率 (%):")
#             st.dataframe(
#                 col_growth_df.style.format("{:,.2f}%").background_gradient(
#                     cmap="RdYlBu", vmin=-10, vmax=10
#                 )
#             )
#         else:
#             st.info("無足夠資料計算年增率")
