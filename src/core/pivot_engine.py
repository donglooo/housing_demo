"""
Pivot table computation and analysis engine.

This module handles all pivot table computations, filtering, sorting,
percentage calculations, and growth rate analysis.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional


@st.cache_data
def compute_pivot_tables(
    df_decode: pd.DataFrame,
    pivot_row: str,
    pivot_col: str,
    pivot_sum: str,
    filter_items: Tuple[Tuple[str, Tuple], ...],
    codebook: Dict,
) -> Tuple[List[int], Dict[int, Dict], List[Dict], List[Dict], List[List]]:
    """
    Compute all yearly pivot tables and summary statistics.

    Args:
        df_decode: Decoded DataFrame
        pivot_row: Row dimension column name
        pivot_col: Column dimension column name
        pivot_sum: Value column to sum
        filter_items: Tuple of (col, tuple(sorted_values)) for hashability
        codebook: Codebook for sorting logic

    Returns:
        Tuple of:
        - unique_years: Sorted list of years
        - results: Dict mapping year to pivot results
        - row_totals_year: List of dicts with row totals per year
        - col_totals_year: List of dicts with column totals per year
        - all_totals_year: List of [year, total] pairs
    """
    # 1. 進行全域篩選
    active_filters = {k: v for k, v in filter_items}

    df_all_years = apply_filters(df_decode, active_filters, pivot_row, pivot_col)

    # Filter out null years
    unique_years = sorted(
        [int(y) for y in df_all_years["DATA_YR"].unique() if not pd.isna(y)],
        reverse=True,
    )

    super_pivot = df_all_years.pivot_table(
        index=["DATA_YR", pivot_row], columns=pivot_col, values=pivot_sum, aggfunc="sum"
    ).fillna(0)

    # 2. 進行年別篩選
    results = {}
    col_totals_year = []
    row_totals_year = []
    all_totals_year = []

    # Convert tuple back to dict for easy lookup
    # active_filters = {k: v for k, v in filter_items}

    for data_yr in unique_years:
        try:
            pivot_table = super_pivot.xs(data_yr, level="DATA_YR").copy()
        except KeyError:
            results[data_yr] = None
            continue
        # Apply filters for this year
        df_year = apply_filters(
            df_decode, active_filters, pivot_row, pivot_col, data_yr
        )

        if df_year.empty:
            results[data_yr] = None
            continue

        # Create pivot table
        pivot_table = df_year.pivot_table(
            index=pivot_row, columns=pivot_col, values=pivot_sum, aggfunc="sum"
        )

        # Sort pivot table using codebook
        pivot_table = sort_pivot_table(pivot_table, pivot_row, pivot_col, codebook)

        # Fill NaN with 0
        pivot_table = pivot_table.fillna(0)

        # Add totals
        pivot_table = add_totals(pivot_table)

        # Calculate percentages
        pivot_table_row, pivot_table_col, pivot_table_total = calculate_percentages(
            pivot_table
        )

        results[data_yr] = {
            "pivot": pivot_table,
            "row_pct": pivot_table_row,
            "col_pct": pivot_table_col,
            "total_pct": pivot_table_total,
        }

        # Store totals for year-over-year comparison
        row_totals_year.append({"DATA_YR": data_yr, **pivot_table["全國"].to_dict()})
        col_totals_year.append(
            {"DATA_YR": data_yr, **pivot_table.loc["全國"].to_dict()}
        )
        all_totals_year.append([data_yr, pivot_table.loc["全國", "全國"]])

    return unique_years, results, row_totals_year, col_totals_year, all_totals_year


def apply_filters(
    df: pd.DataFrame,
    filters: Dict[str, List],
    pivot_row: str,
    pivot_col: str,
    data_yr: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply filter logic to DataFrame.

    Args:
        df: Source DataFrame
        filters: Dict of column -> selected values
        pivot_row: Row dimension (must not be null)
        pivot_col: Column dimension (must not be null)
        data_yr: Year to filter (if provided)

    Returns:
        Filtered DataFrame
    """
    mask = pd.Series(True, index=df.index)  # 建立初始全為True之遮罩
    target_columns = df.columns[1:-1]  # 定義處裡欄位範圍
    unused_cols = []
    # df_filtered = df.copy()

    # Apply user-selected filters
    for col in target_columns:
        if col in filters:
            mask &= df[col].isin(filters[col])
        elif col == pivot_row or col == pivot_col:
            # Pivot dimensions must not be null
            mask &= ~df[col].isna()
        else:
            # Other columns must be null (not selected)
            unused_cols.append(col)

    if unused_cols:
        mask &= df[unused_cols].isna().all(axis=1)

    # Filter by year if provided
    if data_yr is not None:
        mask &= df["DATA_YR"] == data_yr

    return df.loc[mask].copy()


def sort_pivot_table(
    pivot: pd.DataFrame, row_col: str, col_col: str, codebook: Dict
) -> pd.DataFrame:
    """
    Sort pivot table rows and columns using codebook ordering.

    Args:
        pivot: Pivot table to sort
        row_col: Row dimension name (for codebook lookup)
        col_col: Column dimension name (for codebook lookup)
        codebook: Codebook with ordering information

    Returns:
        Sorted pivot table
    """
    try:
        # Sort rows
        if row_col in codebook and "codes" in codebook[row_col]:
            codes = codebook[row_col]["codes"]
            sorted_keys = sorted(
                [k for k in codes.keys() if isinstance(k, int) or str(k).isdigit()]
            )
            sorted_labels = [codes[k] for k in sorted_keys]
            # Remove duplicates while preserving order
            sorted_labels = list(dict.fromkeys(sorted_labels))

            current_labels = pivot.index.tolist()
            final_row_order = [
                lbl for lbl in sorted_labels if lbl in current_labels
            ] + [lbl for lbl in current_labels if lbl not in sorted_labels]
            pivot = pivot.reindex(index=final_row_order)

        # Sort columns
        if col_col in codebook and "codes" in codebook[col_col]:
            codes = codebook[col_col]["codes"]
            sorted_keys = sorted(
                [k for k in codes.keys() if isinstance(k, int) or str(k).isdigit()]
            )
            sorted_labels = [codes[k] for k in sorted_keys]
            # Remove duplicates while preserving order
            sorted_labels = list(dict.fromkeys(sorted_labels))

            current_labels = pivot.columns.tolist()
            final_col_order = [
                lbl for lbl in sorted_labels if lbl in current_labels
            ] + [lbl for lbl in current_labels if lbl not in sorted_labels]
            pivot = pivot.reindex(columns=final_col_order)

    except Exception:
        # If sorting fails, keep original order
        pass

    return pivot


def add_totals(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Add row and column totals to pivot table.

    Args:
        pivot: Pivot table

    Returns:
        Pivot table with "全國" totals added
    """
    pivot_with_totals = pivot.copy()

    # Add row total (全國)
    pivot_with_totals.loc["全國"] = pivot_with_totals.sum()

    # Add column total (全國)
    pivot_with_totals.loc[:, "全國"] = pivot_with_totals.sum(axis=1)

    return pivot_with_totals


def calculate_percentages(
    pivot: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate row, column, and total percentages.

    Args:
        pivot: Pivot table with totals (must have "全國" row and column)

    Returns:
        Tuple of (row_pct, col_pct, total_pct) DataFrames
    """
    # Row percentages (divide by row total)
    pivot_row_pct = pivot.div(pivot["全國"], axis=0)

    # Column percentages (divide by column total)
    pivot_col_pct = pivot.div(pivot.loc["全國"], axis=1)

    # Total percentage (divide by grand total)
    grand_total = pivot.loc["全國", "全國"]
    pivot_total_pct = pivot / grand_total

    return pivot_row_pct, pivot_col_pct, pivot_total_pct


def calculate_growth_rates(
    row_totals_year: List[Dict],
    col_totals_year: List[Dict],
    all_totals_year: List[List],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Calculate year-over-year growth rates.

    Args:
        row_totals_year: List of row totals by year
        col_totals_year: List of column totals by year
        all_totals_year: List of [year, total] pairs

    Returns:
        Tuple of (overall_growth_df, row_growth_df, col_growth_df)
    """
    # Overall growth
    overall_growth_df = None
    if all_totals_year:
        overall_growth_df = pd.DataFrame(all_totals_year, columns=["DATA_YR", "TOTAL"])
        overall_growth_df["DATA_YR"] = pd.to_numeric(overall_growth_df["DATA_YR"])
        overall_growth_df["TOTAL"] = pd.to_numeric(overall_growth_df["TOTAL"])
        overall_growth_df = overall_growth_df.sort_values(by="DATA_YR")
        overall_growth_df["YEARLY_GROWTH"] = overall_growth_df["TOTAL"].pct_change()
        overall_growth_df["YEARLY_GROWTH"] = overall_growth_df["YEARLY_GROWTH"].fillna(
            0
        )
        overall_growth_df["YEARLY_GROWTH_PCT"] = (
            overall_growth_df["YEARLY_GROWTH"] * 100
        )

    # Row dimension growth
    row_growth_df = None
    if row_totals_year:
        row_df = pd.DataFrame(row_totals_year)
        row_df["DATA_YR"] = pd.to_numeric(row_df["DATA_YR"])
        row_df = row_df.sort_values(by="DATA_YR").set_index("DATA_YR")
        row_growth_df = row_df.pct_change() * 100
        row_growth_df = row_growth_df.iloc[1:]  # Remove first row (NaN)

    # Column dimension growth
    col_growth_df = None
    if col_totals_year:
        col_df = pd.DataFrame(col_totals_year)
        col_df["DATA_YR"] = pd.to_numeric(col_df["DATA_YR"])
        col_df = col_df.sort_values(by="DATA_YR").set_index("DATA_YR")
        col_growth_df = col_df.pct_change() * 100
        col_growth_df = col_growth_df.iloc[1:]  # Remove first row (NaN)

    return overall_growth_df, row_growth_df, col_growth_df
