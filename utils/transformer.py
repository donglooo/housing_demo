import pandas as pd
import yaml
import numpy as np

def load_codebook(config_path="config/codebook.yaml"):
    """Loads the YAML codebook configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_codebook(df: pd.DataFrame, codebook: dict) -> pd.DataFrame:
    """
    Applies the mapping logic derived from the codebook to the dataframe.
    Replaces integer codes with string labels.
    """
    df_transformed = df.copy()
    mappings = codebook.get("mappings", {})
    
    for col, settings in mappings.items():
        if col in df_transformed.columns:
            codes_map = settings.get("codes", {})
            
            def mapper(val):
                # Handle Grouping Sets (Totals)
                # If val is NaN/None, it is likely a CUBE Total row.
                if pd.isna(val):
                    return "總計" # Label for Total
                
                # Try map
                if val in codes_map:
                    return codes_map[val]
                try:
                    if int(val) in codes_map:
                        return codes_map[int(val)]
                except:
                    pass
                return val

            df_transformed[col] = df_transformed[col].apply(mapper)
            
    return df_transformed

def get_dimension_options(df: pd.DataFrame) -> list:
    """Returns list of potential dimension columns (categorical)."""
    # Simple heuristic: columns ending in _N, _GROUP, _YR, or just all except CNT/Metrics
    # For now, return all except 'CNT' or numeric metrics
    return [c for c in df.columns if c not in ['CNT']]

def slice_cube(df: pd.DataFrame, row_dim: str, col_dim: str, fixed_dims: dict = None) -> pd.DataFrame:
    """
    Slices the data to get the specific 2D table.
    Logic Update: Per user requirement, we must filter all dimensions to be "Non-NULL" (Leaf nodes)
    and then aggregate up. We do NOT use the pre-calculated CUBE totals (NaN rows) to avoid double counting.
    """
    temp_df = df.copy()
    
    # Identify all dimensions available in DF
    all_dims = [c for c in df.columns if c != 'CNT']
    
    # 1. Global Filter: Exclude CUBE Totals (Labelled as "總計")
    # We must ensure we are working with atomic data.
    for dim in all_dims:
        # Check if the column has "總計" values (which represent CUBE NaNs)
        # We assume "總計" is the label for CUBE Totals injected by apply_codebook
        # We do NOT filter out "Unspecified" (usually code 99/98).
        if "總計" in temp_df[dim].values:
            temp_df = temp_df[temp_df[dim] != "總計"]
            
    # 2. Apply Specific Filters (Dimensions)
    if fixed_dims:
        for dim, val in fixed_dims.items():
            if dim in temp_df.columns:
                temp_df = temp_df[temp_df[dim] == val]

    # 3. Pivot
    # Now we have atomic data, we sum it up.
    pivot = temp_df.pivot_table(index=row_dim, columns=col_dim, values='CNT', aggfunc='sum', fill_value=0)
    
    return pivot

def generate_stats_tables(pivot_df: pd.DataFrame):
    """
    Generates the 4 statistical tables from the base count pivot.
    1. Count
    2. Row Percentage
    3. Column Percentage
    4. Total Percentage
    """
    # 1. Count (Original)
    # Ensure margins (Totals) are computed if they are not already in simpler view
    # But CUBE data might include "總計" as a row/col.
    # If "總計" is in index/columns, let's treat it as data for now, 
    # but strictly speaking, standard stats tables compute separate margins.
    
    # Clean up: If "總計" exists in rows/cols, remove it to recalculate purely? 
    # Or keep it? The user wants standard tables. Standard tables usually *have* margins.
    # If we rely on CUBE provided "總計", the math is already done.
    # If we recalculate, we ensure consistency. 
    # Let's strip "總計" if present to avoid double summing, then add standard margins?
    # Actually, maintaining CUBE integrity is safer. 
    # But for "Row %", we need the Row Total. 
    # If "總計" is in the pivot, we can use it.
    
    tables = {}
    tables['Count'] = pivot_df
    
    total_val = pivot_df.values.sum()
    
    # 2. Total %
    tables['Total_Pct'] = (pivot_df / total_val * 100).round(2)
    
    # 3. Row %
    # Div by row sum
    tables['Row_Pct'] = pivot_df.div(pivot_df.sum(axis=1), axis=0).mul(100).round(2)
    
    # 4. Col %
    tables['Col_Pct'] = pivot_df.div(pivot_df.sum(axis=0), axis=1).mul(100).round(2)
    
    return tables
