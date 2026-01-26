"""
Reusable Streamlit UI components.

This module provides reusable UI components for building
pivot table interfaces in Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


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
        unique_vals[col] = sorted([x for x in df[col].unique() if pd.notna(x)])
    return unique_vals


def render_pivot_selector(chinese_columns: Dict[str, str]) -> tuple:
    """
    Render row/column/sum selector controls.

    Args:
        chinese_columns: Dict mapping column keys to Chinese names

    Returns:
        Tuple of (pivot_row, pivot_col, pivot_sum) selected values
    """
    pivot_row_col, pivot_col_col, pivot_sum_col = st.columns(3)

    opts = list(chinese_columns.keys())

    def get_label(key):
        return chinese_columns.get(key, key)

    with pivot_row_col:
        p_row = st.selectbox(
            "列維度(Row)", opts, format_func=get_label, key="pivot_row"
        )

    with pivot_col_col:
        p_col = st.selectbox(
            "欄維度(Column)", opts, format_func=get_label, key="pivot_col"
        )

    with pivot_sum_col:
        p_sum = st.selectbox("計算欄", ["CNT"], key="pivot_sum")

    return p_row, p_col, p_sum


def render_filter_sidebar(
    df_decode: pd.DataFrame, chinese_columns: Dict[str, str]
) -> None:
    """
    Render filter multiselect controls in sidebar.

    Args:
        df_decode: Decoded DataFrame
        chinese_columns: Dict mapping column keys to Chinese names
    """
    st.sidebar.header("Filters")

    def get_label(key):
        return chinese_columns.get(key, key)

    # Get cached unique values to prevent reloading entire column data
    unique_values = get_unique_values(df_decode)

    for col in df_decode.columns[1:-1]:
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
    unique_years: List[int], results: Dict[int, Dict], axis: Optional[int]
) -> None:
    """
    Render yearly pivot table tabs with styling.

    Args:
        unique_years: Sorted list of years
        results: Dict mapping year to pivot results
        axis: Gradient axis (None, 0, or 1)
    """
    tabs = st.tabs([str(yr) for yr in unique_years])

    for i, data_yr in enumerate(unique_years):
        with tabs[i]:
            res = results.get(data_yr)

            if res is None:
                st.warning(f"No data for {data_yr} with current filters.")
                continue

            pivot_table = res["pivot"]
            pivot_table_row = res["row_pct"]
            pivot_table_col = res["col_pct"]
            pivot_table_total = res["total_pct"]

            # Create sub-tabs for different views
            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(
                ["Pivot Table", "Row(%)", "Col(%)", "Total(%)"]
            )

            # Calculate dynamic height
            dynamic_height = int((len(pivot_table) * 35) + 37)

            # Exclude totals from gradient
            gradient_columns = [col for col in pivot_table.columns if col != "全國"]
            gradient_rows = [idx for idx in pivot_table.index if idx != "全國"]

            with sub_tab1:
                st.dataframe(
                    pivot_table.style.background_gradient(
                        subset=(gradient_rows, gradient_columns),
                        cmap="Blues",
                        axis=axis,
                    ).format("{:,.0f}"),
                    height=dynamic_height,
                )

            with sub_tab2:
                st.dataframe(
                    pivot_table_row.style.background_gradient(
                        subset=(gradient_rows, gradient_columns),
                        cmap="Blues",
                        axis=axis,
                    ).format("{:.2%}"),
                    height=dynamic_height,
                )

            with sub_tab3:
                st.dataframe(
                    pivot_table_col.style.background_gradient(
                        subset=(gradient_rows, gradient_columns),
                        cmap="Blues",
                        axis=axis,
                    ).format("{:.2%}"),
                    height=dynamic_height,
                )

            with sub_tab4:
                st.dataframe(
                    pivot_table_total.style.background_gradient(
                        subset=(gradient_rows, gradient_columns),
                        cmap="Blues",
                        axis=axis,
                    ).format("{:.2%}"),
                    height=dynamic_height,
                )


def render_growth_analysis(
    overall_growth_df: Optional[pd.DataFrame],
    row_growth_df: Optional[pd.DataFrame],
    col_growth_df: Optional[pd.DataFrame],
    pivot_row: str,
    pivot_col: str,
) -> None:
    """
    Render growth rate analysis tabs.

    Args:
        overall_growth_df: Overall growth DataFrame
        row_growth_df: Row dimension growth DataFrame
        col_growth_df: Column dimension growth DataFrame
        pivot_row: Row dimension name
        pivot_col: Column dimension name
    """
    st.markdown("### 年增率分析")

    growth_tabs = st.tabs(["總體", "列(Row)維度", "欄(Col)維度"])

    with growth_tabs[0]:
        if overall_growth_df is not None:
            col_metric1, col_metric2 = st.columns(2)

            avg_growth = overall_growth_df["YEARLY_GROWTH_PCT"].mean()
            latest_growth = overall_growth_df["YEARLY_GROWTH_PCT"].iloc[-1]

            col_metric1.metric("平均年增率", f"{avg_growth:.2f}%")
            col_metric2.metric("最新年增率", f"{latest_growth:.2f}%")

            st.dataframe(
                overall_growth_df[
                    ["DATA_YR", "TOTAL", "YEARLY_GROWTH_PCT"]
                ].style.format({"TOTAL": "{:,.0f}", "YEARLY_GROWTH_PCT": "{:,.2f}%"})
            )
        else:
            st.info("無足夠資料計算年增率")

    with growth_tabs[1]:
        if row_growth_df is not None:
            st.write(f"列維度 ({pivot_row}) 年增率 (%):")
            st.dataframe(
                row_growth_df.style.format("{:,.2f}%").background_gradient(
                    cmap="RdYlBu", vmin=-10, vmax=10
                )
            )
        else:
            st.info("無足夠資料計算年增率")

    with growth_tabs[2]:
        if col_growth_df is not None:
            st.write(f"欄維度 ({pivot_col}) 年增率 (%):")
            st.dataframe(
                col_growth_df.style.format("{:,.2f}%").background_gradient(
                    cmap="RdYlBu", vmin=-10, vmax=10
                )
            )
        else:
            st.info("無足夠資料計算年增率")
