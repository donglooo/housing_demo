"""
Pivot table computation and analysis engine.

This module handles all pivot table computations, filtering, sorting,
percentage calculations, and growth rate analysis.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional

from src.core.config_manager import EXCLUDED_METRIC_PREFIXES, EXCLUDED_METRIC_COLS


@st.cache_data
def compute_pivot_tables(
    df_decode: pd.DataFrame,
    pivot_tab: str,
    pivot_row: str,
    pivot_col: str,
    pivot_sum: str,
    filter_items: Tuple[Tuple[str, Tuple], ...],
    codebook: Dict,
) -> Tuple[List[int], Dict[int, Dict], List[Dict], List[Dict], List[List], bool]:
    """
    Compute all pivot tables and summary statistics for each tab (group).

    Args:
        df_decode: Decoded DataFrame
        pivot_tab: Column to group tabs by (e.g. DATA_YR, COUNTY)
        pivot_row: Row dimension column name
        pivot_col: Column dimension column name
        pivot_sum: Value column to sum
        filter_items: Tuple of (col, tuple(sorted_values)) for hashability
        codebook: Codebook for sorting logic

    Returns:
        Tuple of:
        - unique_tabs: Sorted list of tab keys
        - results: Dict mapping tab key to pivot results
        - row_totals_tab: List of dicts with row totals per tab
        - col_totals_tab: List of dicts with column totals per tab
        - all_totals_tab: List of [tab_key, total] pairs
        - masked_df: DataFrame containing masked data rows
    """
    # 1. 進行全域篩選
    active_filters = {k: v for k, v in filter_items}

    df_all = apply_filters(df_decode, active_filters, pivot_tab, pivot_row, pivot_col, codebook=codebook)

    # Check for masked data
    masked_df = pd.DataFrame()
    if "DATA_STATUS" in df_all.columns:
        masked_df = df_all[df_all["DATA_STATUS"] == "遮蔽"].copy()

    # Determine unique tab values and sort them
    raw_tabs = [t for t in df_all[pivot_tab].unique() if not pd.isna(t)]

    col_info = codebook.get(pivot_tab, {})
    codes = col_info.get("codes", {})

    if codes:
        # Use codebook order (Label -> Code)
        label_to_code = {v: k for k, v in codes.items()}
        # Sort by code value
        unique_tabs = sorted(raw_tabs, key=lambda x: label_to_code.get(x, float("inf")))
    elif pivot_tab == "DATA_YR":
        # Year default reverse sort
        unique_tabs = sorted(raw_tabs, reverse=True)
    else:
        # Default smart sort (numeric if possible, else string)
        try:
             unique_tabs = sorted(
                raw_tabs,
                key=lambda x: float(x) if str(x).replace(".", "").isdigit() else str(x),
                reverse=False
            )
        except Exception:
            unique_tabs = sorted(raw_tabs)

    super_pivot = df_all.pivot_table(
        index=[pivot_tab, pivot_row], columns=pivot_col, values=pivot_sum, aggfunc="sum"
    ).fillna(0)

    # 2. 進行分頁篩選
    results = {}
    col_totals_tab = []
    row_totals_tab = []
    all_totals_tab = []

    for tab_val in unique_tabs:
        try:
            pivot_table = super_pivot.xs(tab_val, level=pivot_tab).copy()
        except KeyError:
            results[tab_val] = None
            continue

        # Create pivot table (already have it from xs, but re-applying specific filters for safety/consistency if needed?)
        # Actually super_pivot handles the aggregation.
        # But wait, super_pivot computed based on df_all which applied generic filters.
        # So we can just use the slice.

        # Ensure we filter properly if apply_filters does something dynamic?
        # apply_filters is static based on active_filters.
        # So super_pivot is correct.

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

        results[tab_val] = {
            "pivot": pivot_table,
            "row_pct": pivot_table_row,
            "col_pct": pivot_table_col,
            "total_pct": pivot_table_total,
        }

        # Store totals for comparison (e.g. YOY, or just trend)
        row_totals_tab.append({pivot_tab: tab_val, **pivot_table["全國"].to_dict()})
        col_totals_tab.append({pivot_tab: tab_val, **pivot_table.loc["全國"].to_dict()})
        all_totals_tab.append([tab_val, pivot_table.loc["全國", "全國"]])

    return unique_tabs, results, row_totals_tab, col_totals_tab, all_totals_tab, masked_df


def apply_filters(
    df: pd.DataFrame,
    filters: Dict[str, List],
    pivot_tab: str,
    pivot_row: str,
    pivot_col: str,
    tab_value: Optional[any] = None,
    codebook: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Apply filter logic to DataFrame.

    Args:
        df: Source DataFrame
        filters: Dict of column -> selected values
        pivot_tab: Tab dimension column name
        pivot_row: Row dimension column name
        pivot_col: Col dimension column name
        tab_value: Optional value to filter pivot_tab by
        codebook: Codebook dictionary for checking group_type

    Returns:
        Filtered DataFrame
    """

    mask = pd.Series(True, index=df.index)  # 建立初始全為True之遮罩
    
    # Define columns/prefixes to exclude from dimension checks (metrics/IDs)
    # Imported from config_manager
    
    # We should iterate over columns that are POTENTIAL DIMENSIONS.
    # Exclude metrics.
    
    target_columns = [
        col for col in df.columns 
        if not col.startswith(EXCLUDED_METRIC_PREFIXES) 
        and col not in EXCLUDED_METRIC_COLS
    ]
    
    unused_cols_to_null = [] # Columns that must be null if not selected

    # Apply user-selected filters
    for col in target_columns:
        if col in filters:
            mask &= df[col].isin(filters[col])
        if col == pivot_tab:
             # Pivot tab dimension must not be null
            mask &= ~df[col].isna()
        elif col == pivot_row or col == pivot_col:
            # Pivot row/col dimensions must not be null
            mask &= ~df[col].isna()
        else:
            # Smart Probe Logic:
            # Check if an aggregate row (where col is NaN) exists for the current context.
            # If (mask & col.isna()).any() is True, it means the DB provides a pre-calculated aggregate.
            # We should use it (Enforce NaN) to avoid double counting.
            # If False, it means no aggregate exists (e.g. selecting a Child node in Rollup without selecting Parent),
            # so we must aggregate manually (Enforce NotNa).
            
            # Note: We must check existence against the CURRENT mask (so far).
            # But order matters? Ideally check against "If we enforced nothing on this col".
            
            probe_mask = mask & df[col].isna()
            if probe_mask.any():
                unused_cols_to_null.append(col)
            else:
                # No aggregate row found -> Sum the leaf nodes
                mask &= df[col].notna()

    if unused_cols_to_null:
        # All columns in unused_cols_to_null must be NaN for the row to be included
        mask &= df[unused_cols_to_null].isna().all(axis=1)

    # Filter by specific tab value if provided
    if tab_value is not None:
        mask &= df[pivot_tab] == tab_value

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
    # Calculate totals
    row_sum = pivot.sum(axis=0)
    col_sum = pivot.sum(axis=1)
    grand_total = row_sum.sum()

    # Reconstruct DataFrame with "全國" at the top/left
    
    # 1. Add "全國" to row_sum (for the top row)
    row_sum["全國"] = grand_total
    
    # 2. Create new index with "全國" first
    new_index = ["全國"] + [idx for idx in pivot.index]
    new_columns = ["全國"] + [col for col in pivot.columns]
    
    # 3. Create new DF
    pivot_with_totals = pd.DataFrame(index=new_index, columns=new_columns)
    
    # 4. Fill values
    pivot_with_totals.loc["全國", "全國"] = grand_total
    pivot_with_totals.loc["全國", pivot.columns] = row_sum[pivot.columns]
    pivot_with_totals.loc[pivot.index, "全國"] = col_sum
    pivot_with_totals.loc[pivot.index, pivot.columns] = pivot.values

    # Ensure numeric types
    pivot_with_totals = pivot_with_totals.apply(pd.to_numeric, errors='ignore')

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
    row_totals_tab: List[Dict],
    col_totals_tab: List[Dict],
    all_totals_tab: List[List],
    pivot_tab_name: str = "DATA_YR",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Calculate growth rates between tabs (only valid for numeric/time series).

    Args:
        row_totals_tab: List of row totals by tab
        col_totals_tab: List of column totals by tab
        all_totals_tab: List of [tab_key, total] pairs
        pivot_tab_name: Name of the tab grouping column. Defaults to "DATA_YR".

    Returns:
        Tuple of (overall_growth_df, row_growth_df, col_growth_df)
    """

    # If pivot_tab is not numeric-like (e.g. DATA_YR), we might skip growth calc or just return differences?
    # For now, let's try to enforce numeric conversion. If it fails, return None.

    is_time_series = True
    # check first element if possible
    if all_totals_tab:
        first_val = all_totals_tab[0][0]
        try:
            float(first_val)
        except (ValueError, TypeError):
            # Not a number, so growth rate (pct_change) doesn't imply time progression usually.
            # But maybe user wants to compare categories? Pct change between categories is weird.
            is_time_series = False

    if not is_time_series and pivot_tab_name != "DATA_YR":
        return None, None, None

    # Overall growth
    overall_growth_df = None
    if all_totals_tab:
        overall_growth_df = pd.DataFrame(
            all_totals_tab, columns=[pivot_tab_name, "TOTAL"]
        )
        try:
            overall_growth_df[pivot_tab_name] = pd.to_numeric(
                overall_growth_df[pivot_tab_name]
            )
            overall_growth_df["TOTAL"] = pd.to_numeric(overall_growth_df["TOTAL"])
            overall_growth_df = overall_growth_df.sort_values(by=pivot_tab_name)
            overall_growth_df["YEARLY_GROWTH"] = overall_growth_df["TOTAL"].pct_change()
            overall_growth_df["YEARLY_GROWTH"] = overall_growth_df[
                "YEARLY_GROWTH"
            ].fillna(0)
            overall_growth_df["YEARLY_GROWTH_PCT"] = (
                overall_growth_df["YEARLY_GROWTH"] * 100
            )
        except Exception:
            overall_growth_df = None

    # Row dimension growth
    row_growth_df = None
    if row_totals_tab:
        try:
            row_df = pd.DataFrame(row_totals_tab)
            # Ensure pivot_tab column is numeric for sorting
            row_df[pivot_tab_name] = pd.to_numeric(row_df[pivot_tab_name])
            row_df = row_df.sort_values(by=pivot_tab_name).set_index(pivot_tab_name)
            row_growth_df = row_df.pct_change() * 100
            row_growth_df = row_growth_df.iloc[1:]  # Remove first row (NaN)
        except Exception:
            row_growth_df = None

    # Column dimension growth
    col_growth_df = None
    if col_totals_tab:
        try:
            col_df = pd.DataFrame(col_totals_tab)
            col_df[pivot_tab_name] = pd.to_numeric(col_df[pivot_tab_name])
            col_df = col_df.sort_values(by=pivot_tab_name).set_index(pivot_tab_name)
            col_growth_df = col_df.pct_change() * 100
            col_growth_df = col_growth_df.iloc[1:]  # Remove first row (NaN)
        except Exception:
            col_growth_df = None

    return overall_growth_df, row_growth_df, col_growth_df
