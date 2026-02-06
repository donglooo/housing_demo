"""
Excel Exporter for Saved Pivot Tables.

This module handles generating a consolidated Excel file from a list of saved configurations.
It re-runs the pivot queries to get fresh data and organizes them into sheets with a Table of Contents.
"""

import pandas as pd
import io
import streamlit as st
from typing import List, Dict
import datetime

# Import pivot engine to re-run queries
from src.core.data_loader import load_data, load_codebook
from src.core.pivot_engine import compute_pivot_tables
from src.core.data_loader import decode_data
from src.core.config_manager import CODEBOOK_PATH, get_codebook_section

def export_all_pivots_to_excel(saved_configs: List[Dict]) -> bytes:
    """
    Export all saved pivot configurations to a single Excel file.
    
    Returns:
        bytes: The Excel file content.
    """
    output = io.BytesIO()
    
    # Sort configs by Chapter for logical order
    # Handle missing chapters or different formats if needed
    def get_sort_key(cfg):
        chap = cfg.get("chapter", "")
        # Try to sort numerically if possible (e.g. 4-1-1 -> [4, 1, 1])
        try:
            return [int(x) for x in chap.split("-") if x.isdigit()]
        except:
            return chap
            
    sorted_configs = sorted(saved_configs, key=get_sort_key)
    
    # Load Codebook once (assumed constant for all)
    # We might need to load data per config if they use different sources
    # Optimization: Cache data loaders if multiple configs use same source
    data_cache = {}
    codebook = load_codebook(CODEBOOK_PATH)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- 1. Table of Contents (TOC) ---
        toc_sheet_name = "目錄"
        # Create TOC sheet
        toc_data = []
        for cfg in sorted_configs:
            chap = cfg.get("chapter", "").strip()
            name = cfg.get("name", "未命名")
            unit = cfg.get("unit", "")
            unit_str = f"({unit})" if unit else ""
            
            # Sheet name: Chapter (sanitized)
            # Excel sheet names max 31 chars, no special chars : \ / ? * [ ]
            sheet_name = str(chap) if chap else f"Report_{sorted_configs.index(cfg)}"
            sheet_name = "".join([c for c in sheet_name if c not in r"[]:*?/\\"])[:31]
            
            # Store for linking
            cfg["_sheet_name"] = sheet_name 
            
            toc_data.append({
                "章節": chap,
                "標題": f"{name} {unit_str}",
                "連結": f"internal:'{sheet_name}'!A1"
            })
            
        df_toc = pd.DataFrame(toc_data)
        df_toc.to_excel(writer, sheet_name=toc_sheet_name, index=False)
        
        # Apply formatting to TOC
        toc_sheet = writer.sheets[toc_sheet_name]
        # Set column widths
        toc_sheet.set_column('A:A', 15) # Chapter
        toc_sheet.set_column('B:B', 60) # Title
        
        # Write hyperlinks manually for the "連結" column?
        # Pandas writes URLs automatically if they look like URLs, but internal links might need help or the 'internal:' prefix works directly in some versions?
        # Actually pandas writes strings. Let's use xlsxwriter to write links if needed.
        # But 'internal:' syntax is for helpful reference, actual link needs `write_url`.
        # Simplified: Just write the sheet name. User requested "embedded links".
        # We can iterate and overwrite the cell with a URL.
        
        link_format = workbook.add_format({'font_color': 'blue', 'underline': 1})
        for i, row in df_toc.iterrows():
            # Row 0 is header. Data starts at row 1.
            # Link to the Sheet
            sheet_n = sorted_configs[i]["_sheet_name"]
            toc_sheet.write_url(i + 1, 1, f"internal:'{sheet_n}'!A1", string=row["標題"], cell_format=link_format)


        # --- 2. Report Sheets ---
        for cfg in sorted_configs:
            try:
                # Extract Params
                data_path = cfg.get("data_source")
                if not data_path:
                    continue
                    
                # Load Data (Cached)
                if data_path not in data_cache:
                    df = load_data(data_path)
                    # Infer codebook section? Default 'ownership' or 'usage'?
                    # We can try to guess from filepath or config?
                    # The saved config stores raw data path but not the "dataset_type" key explicitly used in Playground.
                    # However, usually the path contains "所有權" or "使用權".
                    if "所有權" in data_path:
                        d_type = "ownership"
                    elif "使用權" in data_path:
                        d_type = "usage"
                    else:
                        d_type = "ownership" # Default
                        
                    cb_sec = get_codebook_section(codebook, d_type)
                    df_decode = decode_data(df, cb_sec)
                    data_cache[data_path] = (df_decode, cb_sec)
                
                df_decode, codebook_sel = data_cache[data_path]
                
                # Re-run Query
                # Reconstruct filter items
                filters = cfg.get("filters", {})
                filter_items = []
                for k, v in filters.items():
                    # v must be tuple/list
                    if isinstance(v, (list, tuple)):
                        filter_items.append((k, tuple(sorted(v))))
                filter_items = tuple(sorted(filter_items))
                
                # Create result
                # Note: pivot_row/col might be lists or strings
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
                    cfg.get("pivot_tab"),
                    cfg.get("pivot_row"),
                    cfg.get("pivot_col"),
                    cfg.get("pivot_sum"),
                    filter_items,
                    codebook_sel
                )
                
                # Extract specific focus tab if set, otherwise first
                focus = cfg.get("focus_tab")
                if not focus or focus not in results or results[focus] is None:
                    # Fallback to first available
                    if unique_tabs:
                        focus = unique_tabs[0]
                
                if focus and focus in results and results[focus] is not None:
                    # Get the formatting options? 
                    # Saved config doesn't store 'axis' preference. Default to 0?
                    # We usually dump the raw numbers or percentages?
                    # User asked for "statistical results".
                    # Let's dump the Pivot Table (counts) + Percentages if possible?
                    # Or just the main Pivot Table.
                    # Let's output the Main Pivot with Totals.
                    
                    res_data = results[focus]
                    df_val = res_data["pivot"]
                    df_pct = res_data["row_pct"] # Row percentages
                    
                    # Create Sheet
                    sheet_n = cfg["_sheet_name"]
                    
                    # Write Title Block
                    # Row 0: Title
                    current_sheet = workbook.add_worksheet(sheet_n)
                    
                    title_format = workbook.add_format({'bold': True, 'font_size': 14})
                    unit_str = f" (單位: {cfg.get('unit', '')})" if cfg.get('unit') else ""
                    full_title = f"{cfg.get('name', '')}{unit_str}"
                    current_sheet.write(0, 0, full_title, title_format)
                    
                    # Row 1: Description
                    desc = cfg.get("description", "")
                    if desc:
                        current_sheet.write(1, 0, desc)
                        start_row = 3
                    else:
                        start_row = 2

                    # --- Construct Mix DataFrame (Logic from ui_components.py) ---
                    # 1. Base Setup
                    cols = [c for c in df_val.columns if c != "全國"]
                    hybrid_df = pd.DataFrame(index=df_val.index)
                    
                    # 2. Add '全國' (Total) if exists
                    if "全國" in df_val.columns:
                        hybrid_df["全國"] = df_val["全國"]
                        
                    # 3. Add Averages (if available)
                    avg_data = res_data.get("avg_data")
                    if avg_data:
                        rows_avg = avg_data["rows_avg"]
                        total_avg = avg_data["total_avg"]
                        # Assume first row is Total if "全國" exists in index? 
                        # Or typically the first row is filtered out in UI but here we export everything?
                        # In ui_components line 369, it gets total_row_key = hybrid_df.index[0] and updates it.
                        if not hybrid_df.index.empty:
                            total_row_key = hybrid_df.index[0]
                            for avg_key in rows_avg.columns:
                                s = pd.Series(index=hybrid_df.index, dtype=float)
                                s.update(rows_avg[avg_key])
                                # Update total row if it aligns
                                if total_row_key in s.index:
                                    s.loc[total_row_key] = total_avg.get(avg_key, 0)
                                hybrid_df[avg_key] = s

                    # 4. Interleave Value and Percentage Columns
                    for c in cols:
                        # Value
                        hybrid_df[f"{c}"] = df_val[c]
                        # Percentage
                        if c in df_pct.columns:
                            hybrid_df[f"{c}(%)"] = df_pct[c]
                        else:
                            hybrid_df[f"{c}(%)"] = 0

                    # Write Data
                    hybrid_df.to_excel(writer, sheet_name=sheet_n, startrow=start_row)
                    
                    # --- Formatting ---
                    worksheet = writer.sheets[sheet_n]
                    workbook = writer.book
                    
                    fmt_count = workbook.add_format({'num_format': '#,##0'})
                    fmt_pct = workbook.add_format({'num_format': '0.00%'})
                    fmt_avg = workbook.add_format({'num_format': '#,##0.00'})
                    
                    # Index levels determine where data columns start
                    index_levels = hybrid_df.index.nlevels
                    
                    # Set Index Column Widths
                    worksheet.set_column(0, index_levels - 1, 15)
                    
                    # Apply formats to data columns
                    for i, col_name in enumerate(hybrid_df.columns):
                        excel_col = index_levels + i
                        col_str = str(col_name)
                        
                        # Determine format
                        if "(%)" in col_str:
                            cell_fmt = fmt_pct
                        elif "平均" in col_str or "age" in col_str.lower(): # Simple heuristic for averages
                             cell_fmt = fmt_avg
                        else:
                             cell_fmt = fmt_count
                        
                        # Set width 15 and format
                        worksheet.set_column(excel_col, excel_col, 15, cell_fmt)
                    
            except Exception as e:
                # If error, write an error sheet or skip
                print(f"Error exporting {cfg.get('name')}: {e}")
                pass

    output.seek(0)
    return output.getvalue()
