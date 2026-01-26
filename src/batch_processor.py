import yaml
import pandas as pd
import os
import streamlit as st
import numpy as np
from utils.transformer import load_codebook, apply_codebook, generate_stats_tables

# Cache data loading to avoid reloading for every task using same source
@st.cache_data
def load_and_prep_data(csv_path):
    """
    Loads CSV and applies codebook mapping. 
    Assumes CSV path is relative to project root or absolute.
    """
    if not os.path.isabs(csv_path):
        pass # rely on pd to find it relative to CWD
            
    if not os.path.exists(csv_path):
        # Graceful fallback or error
        raise FileNotFoundError(f"Source file not found: {csv_path}")
        
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    codebook = load_codebook()
    df_labeled = apply_codebook(df, codebook)
    
    return df_labeled

def load_tables_config(config_path="config/tables.yaml"):
    """Loads the tables definition YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def process_batch_presets_for_export(config_data: dict) -> tuple:
    """
    Processes presets and returns (data_dict, toc_metadata) for ExcelExporter.
    
    Returns:
        data_dict: { "SheetName": [ { 'df': df, 'fmt': '0' }, ... ] }
        toc_metadata: [ {'sheet_name':..., 'description':..., 'index':...} ]
    """
    data_dict = {}
    toc_metadata = []
    
    presets = config_data.get('presets', [])
    if not presets:
        return {}, []
    
    # We assume we are processing ONE preset or merging all?
    # The new requirement is "batch generate report". usually ONE report per preset or ONE Combined?
    # Let's assume the user selects ONE preset in the UI, but this function takes the config.
    # To support the previous UI flow, we iterate all presets passed in config.
    # But usually config has ALL presets. 
    # Function should probably handle just the tasks passed to it?
    # Let's assume config_data IS the filtered structure or we process all.
    
    for preset in presets:
        preset_name = preset.get('name', 'Report')
        tasks = preset.get('tasks', [])
        
        for i_task, task in enumerate(tasks, 1):
            task_name = task.get('name', f'Table_{i_task}')
            desc = task.get('description', '')
            csv_src = task.get('csv_source')
            rows = task.get('rows', []) 
            cols = task.get('cols', []) 
            exclude = task.get('exclude_values', [])
            agg_val = task.get('values', '筆數') 
            aggfunc = task.get('aggfunc', 'sum')
            
            # --- 1. Load Data ---
            try:
                df = load_and_prep_data(csv_src)
            except Exception as e:
                print(f"Error loading {csv_src}: {e}")
                continue
            
            # --- 2. Cube Slicing Logic (Strict Match to 99_dev.py) ---
            df_working = df.copy()
            
            # Identify Dimensions: All cols except DATA_YR (0) and Metric (last)
            # Assumption: Structure is [DATA_YR, Dim1, Dim2, ..., Metric]
            # This is standard for the user's "Cube" CSVs.
            all_dims = df_working.columns[1:-1].tolist()
            
            target_dims = set(rows + cols)
            
            # A. Filter Target Dims: Must NOT be NaN + Exclusions
            for dim in target_dims:
                if dim in df_working.columns:
                    # 1. Not NaN
                    df_working = df_working.dropna(subset=[dim])
                    # 2. Exclusions (If specified)
                    if exclude:
                         # Exclude if value is in list
                         df_working = df_working[~df_working[dim].isin(exclude)]
            
            # B. Filter Unused Dims: Must BE NaN (Total) to prevent double counting
            for dim in all_dims:
                if dim not in target_dims and dim in df_working.columns:
                     df_working = df_working[df_working[dim].isna()]
            
            if df_working.empty:
                continue

            # --- 3. Pivot ---
            # Determine Value Column
            val_col = None
            if agg_val in df_working.columns:
                val_col = agg_val
            elif 'CNT' in df_working.columns: # Default fallback
                val_col = 'CNT'
            # If no value col, pivot_table with aggfunc='size' works but we need a col to count?
            
            try:
                if val_col:
                    pivot = df_working.pivot_table(index=rows, columns=cols, values=val_col, aggfunc=aggfunc)
                else:
                    pivot = df_working.pivot_table(index=rows, columns=cols, aggfunc='size')
                
                # Fill NaNs with 0
                pivot = pivot.fillna(0)
                
                # --- 4. Grand Totals ---
                # Add Row Total
                pivot['總計'] = pivot.sum(axis=1)
                # Add Col Total
                pivot.loc['總計'] = pivot.sum(axis=0)
                
                # --- 5. Generate Stats Tables ---
                # (Raw, Row%, Col%, Total%)
                stats_dfs = generate_stats_tables(pivot)
                
                # --- 6. Package for ExcelExporter ---
                # We need a list of tables.
                # Standard format from stats_dfs: {"Count": df, "RowPct": df, ...}
                # User wants a readable report. Usually:
                # Table 1: Count (Primary)
                # Table 2: Row % ?
                # Let's include Count and Row % as standard, or all?
                # Let's include all 4 stacked.
                
                tables_list = []
                
                # Map internal keys to display logic/format
                # Count
                if "Count" in stats_dfs:
                    tables_list.append({'df': stats_dfs["Count"], 'fmt': '#,##0'})
                    
                # Percentages
                for k in ["RowPct", "ColPct", "TotalPct"]:
                    if k in stats_dfs:
                        # Optional: Add a title row or just stack them? 
                        # ExcelExporter stacks them with spacing.
                        # Maybe rename index to indicate what it is?
                        # For now, just stack.
                        tables_list.append({'df': stats_dfs[k], 'fmt': '0.00%'})
                
                # Sheet Name Logic
                # clean_name = task_name.replace(":", "").replace("/", "_")[:31]
                # Actually, duplicate names handle?
                clean_name = task_name[:31] 
                
                # Store
                data_dict[clean_name] = tables_list
                
                # Metadata
                toc_metadata.append({
                    'sheet_name': clean_name,
                    'description': desc,
                    'index': task_name
                })
                
            except Exception as e:
                print(f"Pivot failed for {task_name}: {e}")
                
    return data_dict, toc_metadata


