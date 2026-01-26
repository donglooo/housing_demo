# 示範開發區，之後請你幫我優化至正式流程內
import streamlit as st
import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import pyarrow


st.set_page_config(page_title="Playground", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

#========================= FUNCTION =========================
def apply_pareto(df_pivot, top_n):
    """
    對透視表應用 Pareto 法則：保留 Top N，其餘合併為 'Others'
    """
    # 如果資料行數少於 Top N，不需處理
    if len(df_pivot) <= top_n:
        return df_pivot

    # 1. 確保有 '全國' 欄位可供排序 (如果還沒算，先算暫時的)
    if '全國' not in df_pivot.columns:
        df_pivot['全國'] = df_pivot.sum(axis=1)
    
    # 2. 排序
    df_sorted = df_pivot.sort_values(by='全國', ascending=False)
    
    # 3. 切分 Top N 與 Tail
    df_top = df_sorted.head(top_n)
    df_tail = df_sorted.iloc[top_n:]
    
    # 4. 如果有剩餘資料，合併為 Others
    if not df_tail.empty:
        # Sum the tail, convert to DataFrame and Transpose to match row format
        others_row = df_tail.sum().to_frame().T
        others_row.index = ['其他 (Others)']
        
        # 合併
        df_final = pd.concat([df_top, others_row])
    else:
        df_final = df_top
        
    return df_final

#========================= PATH INITIALIZE =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

#========================= CONFIG =========================
# CODEBOOK_PATH = os.path.join(CONFIG_DIR, 'codebook.yaml')
# Reading combined config from tables.yaml
TABLES_CONFIG_PATH = os.path.join(CONFIG_DIR, 'tables.yaml')

# Load App Config
if os.path.exists(TABLES_CONFIG_PATH):
    with open(TABLES_CONFIG_PATH, 'r', encoding='utf-8') as f:
        app_settings = yaml.safe_load(f)
else:
    app_settings = {}

# config: source data
# Allow relative path in config being resolved to absolute
# select data from dropdown: data\*\*.csv
file_list = glob.glob(os.path.join(DATA_DIR, '**', '*.parquet'))
raw_data_path = st.selectbox("選擇資料集", file_list)

if not os.path.isabs(raw_data_path):
    DATA_PATH = os.path.join(BASE_DIR, raw_data_path)
else:
    DATA_PATH = raw_data_path

# 確認檔案的前綴
reg_pat = r'^(.*)_coded'
try:
    match = re.search(reg_pat, os.path.basename(DATA_PATH))
    if not match:
        st.error(f"檔案名稱格式錯誤: {os.path.basename(DATA_PATH)}")
        st.stop()
    prefix = match.group(1)
except Exception as e:
    st.error(f"處理檔案時發生錯誤: {str(e)}")
    st.stop()

CODEBOOK_MAPPING = {
    '稅電': 'codebook_power.yaml',
    '所有權': 'codebook_own.yaml',
    '稅籍': 'codebook_tax.yaml'
}

# CODEBOOK
# CODEBOOK SELECTION
# codebook_list = glob.glob(os.path.join(CONFIG_DIR, 'codebook*.yaml'))
# codebook_options = [os.path.basename(p) for p in codebook_list]

# if not codebook_options:
#     st.error("No codebook files found in config directory.")
#     st.stop()
status = 0
# selected_codebook_name = st.sidebar.selectbox("Select Codebook", codebook_options)
CODEBOOK_PATH = os.path.join(CONFIG_DIR, CODEBOOK_MAPPING[prefix])
codebook = yaml.safe_load(open(CODEBOOK_PATH, 'r', encoding='utf-8'))

#========================= MAIN =========================
@st.cache_data(ttl=1800, max_entries=3)
def load_data(DATA_PATH):
    df = pd.read_parquet(DATA_PATH, engine='pyarrow')
    return df

@st.cache_data
def get_decoded_data(df, _codebook):
    df_decode = df.copy()
    for col in codebook['mappings'].keys():
        if col in df_decode.columns:
            df_decode[col] = df_decode[col].replace(codebook['mappings'][col]['codes'])
    return df_decode

def fet_chinese_columns(codebook):
    chinese_columns = {}
    for col in codebook['mappings'].keys():
        # chinese_columns[codebook['mappings'][col]['name']] = col
        chinese_columns[col] = codebook['mappings'][col]['name']
    return chinese_columns

def get_label(key):
    return chinese_columns.get(key, key)

#========================= OUTPUT =========================
# Load data
chinese_columns = fet_chinese_columns(codebook)
df = load_data(DATA_PATH)
df_decode = get_decoded_data(df, codebook)
# Defaults from config
def_row = app_settings.get('settings', {}).get('defaults', {}).get('row_dim', None)
def_col = app_settings.get('settings', {}).get('defaults', {}).get('col_dim', None)
def_sum = app_settings.get('settings', {}).get('defaults', {}).get('sum_metric', None)

status = 1
# PIVOT_TABLE
# ROW
# COL
# SUM
pivot_row, pivot_col, pivot_sum = st.columns(3)

# config: row
opts = chinese_columns.keys()
p_row = pivot_row.selectbox("列維度(Row)", opts, format_func=get_label, key="pivot_row") 

# config: col
p_col = pivot_col.selectbox("欄維度(Column)", opts, format_func=get_label, key="pivot_col") 

# config: sum
# opts_sum = chinese_columns.keys()[-1:] # usually just 'CNT' or last col
p_sum = 'CNT'#pivot_sum.selectbox("計算欄", , key="pivot_sum") 

# 製作篩選器（複選）
st.sidebar.header("Filters")
for col in df_decode.columns[1:-1]:
    st.sidebar.multiselect(get_label(col), df_decode[col].unique(), key=col)

# 視覺化設定，表格顏色標記: None, 0, 1
st.sidebar.header("Visual Settings")
check_visual_axis = st.sidebar.radio("視覺化比較軸向", ["全表", "直向", "橫向"], index=0, horizontal=True)
axis_options = {
    "全表": None,   # 適合：找全域最大/最小值
    "直向": 0,         # 適合：比較同一月份各縣市的表現
    "橫向": 1          # 適合：比較同一縣市不同坪數的分布
}
axis = axis_options[check_visual_axis]

if st.session_state['pivot_row'] == st.session_state['pivot_col']:
    st.warning('🔼 請選擇不同的交叉維度')
else:
    status = 2

@st.cache_data
def compute_all_pivots(df_decode, pivot_row, pivot_col, pivot_sum, filter_items, codebook_mappings):
    """
    Computes all yearly pivot tables and summary statistics.
    filter_items: Tuple of (col, tuple(sorted_values)) for hashability.
    """
    unique_years = [int(yr) for yr in df_decode['DATA_YR'].unique() if not pd.isna(yr)]
    unique_years = sorted(unique_years, reverse=True)
    
    results = {}
    col_totals_year = []
    row_totals_year = []
    all_totals_year = []
    
    # Convert tuple back to dict for easy lookup
    active_filters = {k: v for k, v in filter_items}

    for data_yr in unique_years:
        # Start from base decoded DF
        df_year = df_decode.copy()
        
        # Apply Filters
        for col in df_decode.columns[1:-1]:
            if col in active_filters:
                df_year = df_year[df_year[col].isin(active_filters[col])]
            elif col == pivot_row or col == pivot_col:
                df_year = df_year[~df_year[col].isna()]
            else:
                df_year = df_year[df_year[col].isna()]
        
        df_year = df_year[df_year['DATA_YR'] == data_yr].copy()
        
        if df_year.empty:
            results[data_yr] = None
            continue

        pivot_table = df_year.pivot_table(index=pivot_row, columns=pivot_col, values=pivot_sum, aggfunc='sum')

        # Sort Logic
        try:
            # 1. Row Order
            if pivot_row in codebook_mappings:
                codes = codebook_mappings[pivot_row]['codes']
                sorted_keys = sorted([k for k in codes.keys() if isinstance(k, int) or str(k).isdigit()])
                sorted_labels = [codes[k] for k in sorted_keys]
                sorted_labels = list(dict.fromkeys(sorted_labels)) # Dedup
                current_labels = pivot_table.index.tolist()
                final_row_order = [lbl for lbl in sorted_labels if lbl in current_labels] + [lbl for lbl in current_labels if lbl not in sorted_labels]
                pivot_table = pivot_table.reindex(index=final_row_order)

            # 2. Col Order
            if pivot_col in codebook_mappings:
                codes = codebook_mappings[pivot_col]['codes']
                sorted_keys = sorted([k for k in codes.keys() if isinstance(k, int) or str(k).isdigit()])
                sorted_labels = [codes[k] for k in sorted_keys]
                sorted_labels = list(dict.fromkeys(sorted_labels)) # Dedup
                current_labels = pivot_table.columns.tolist()
                final_col_order = [lbl for lbl in sorted_labels if lbl in current_labels] + [lbl for lbl in current_labels if lbl not in sorted_labels]
                pivot_table = pivot_table.reindex(columns=final_col_order)
        except Exception:
            pass # Sort failed, keep original

        # fill zero
        pivot_table = pivot_table.fillna(0)
        # Add Grand Totals
        pivot_table.loc['全國'] = pivot_table.sum()
        pivot_table.loc[:, '全國'] = pivot_table.sum(axis=1)

        # Calculate Percentages
        pivot_table_row = pivot_table.div(pivot_table['全國'], axis=0)
        pivot_table_col = pivot_table.div(pivot_table.loc['全國'], axis=1)
        pivot_table_total = pivot_table / pivot_table.loc['全國', '全國']
        
        results[data_yr] = {
            'pivot': pivot_table,
            'row_pct': pivot_table_row,
            'col_pct': pivot_table_col,
            'total_pct': pivot_table_total
        }

        # For comparison
        row_totals_year.append({'DATA_YR': data_yr, **pivot_table['全國'].to_dict()})
        col_totals_year.append({'DATA_YR': data_yr, **pivot_table.loc['全國'].to_dict()})
        all_totals_year.append([data_yr, pivot_table.loc['全國', '全國']])
        
    return unique_years, results, row_totals_year, col_totals_year, all_totals_year

if status == 2 and st.button('查詢', type='primary'):
    # Prepare filters for caching (hashable tuple)
    current_filter_items = []
    for col in df_decode.columns[1:-1]:
        if col in st.session_state and st.session_state[col]:
            current_filter_items.append((col, tuple(sorted(st.session_state[col]))))
    current_filter_items = tuple(sorted(current_filter_items))

    # Compute
    unique_years, results, row_totals_year, col_totals_year, all_totals_year = compute_all_pivots(
        df_decode, 
        st.session_state['pivot_row'], 
        st.session_state['pivot_col'], 
        st.session_state['pivot_sum'], 
        current_filter_items,
        codebook['mappings']
    )

    # Render
    tabs = st.tabs([str(yr) for yr in unique_years])
    for i, data_yr in enumerate(unique_years):
        with tabs[i]:
            res = results.get(data_yr)
            if res is None:
                st.warning(f"No data for {data_yr} with current filters.")
                continue

            pivot_table = res['pivot']
            pivot_table_row = res['row_pct']
            pivot_table_col = res['col_pct']
            pivot_table_total = res['total_pct']

            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["Pivot Table", "Row(%)", "Col(%)", "Total(%)"])
            dynamic_height = int((len(pivot_table) * 35) + 37)
            # Re-calculate gradients for display
            gradient_columns = [col for col in pivot_table.columns if col != '全國']
            gradient_rows = [idx for idx in pivot_table.index if idx != '全國']
            
            with sub_tab1:
                st.dataframe(pivot_table.style.background_gradient(
                            subset=(gradient_rows, gradient_columns), 
                            cmap='Blues',
                            axis=axis
                        ).format("{:,.0f}"), height=dynamic_height)
            with sub_tab2:
                st.dataframe(pivot_table_row.style.background_gradient(
                            subset=(gradient_rows, gradient_columns), 
                            cmap='Blues',
                            axis=axis
                        ).format("{:.2%}"), height=dynamic_height)
            with sub_tab3:
                st.dataframe(pivot_table_col.style.background_gradient(
                            subset=(gradient_rows, gradient_columns), 
                            cmap='Blues',
                            axis=axis
                        ).format("{:.2%}"), height=dynamic_height)
            with sub_tab4:
                st.dataframe(pivot_table_total.style.background_gradient(
                            subset=(gradient_rows, gradient_columns), 
                            cmap='Blues',
                            axis=axis
                        ).format("{:.2%}"), height=dynamic_height)

    # 比較年增狀況
    st.markdown("### 年增率分析")
    growth_tabs = st.tabs(["總體", "列(Row)維度", "欄(Col)維度"])

    with growth_tabs[0]:
        # 計算平均年增率
        if all_totals_year:
            all_totals_year_df = pd.DataFrame(all_totals_year, columns=['DATA_YR', 'TOTAL'])
            all_totals_year_df['DATA_YR'] = pd.to_numeric(all_totals_year_df['DATA_YR'])
            all_totals_year_df['TOTAL'] = pd.to_numeric(all_totals_year_df['TOTAL'])
            all_totals_year_df = all_totals_year_df.sort_values(by='DATA_YR')
            all_totals_year_df['YEARLY_GROWTH'] = all_totals_year_df['TOTAL'].pct_change()
            all_totals_year_df['YEARLY_GROWTH'] = all_totals_year_df['YEARLY_GROWTH'].fillna(0)
            all_totals_year_df['YEARLY_GROWTH_PCT'] = all_totals_year_df['YEARLY_GROWTH'] * 100
            
            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric("平均年增率", f'{all_totals_year_df["YEARLY_GROWTH_PCT"].mean():.2f}%')
            col_metric2.metric("最新年增率", f'{all_totals_year_df["YEARLY_GROWTH_PCT"].iloc[-1]:.2f}%')
            
            st.dataframe(all_totals_year_df[['DATA_YR', 'TOTAL', 'YEARLY_GROWTH_PCT']].style.format({"TOTAL": "{:,.0f}", "YEARLY_GROWTH_PCT": "{:,.2f}%"}))

    with growth_tabs[1]:
        if row_totals_year:
            row_growth_df = pd.DataFrame(row_totals_year)
            row_growth_df['DATA_YR'] = pd.to_numeric(row_growth_df['DATA_YR'])
            row_growth_df = row_growth_df.sort_values(by='DATA_YR').set_index('DATA_YR')
            # Calculate pct change
            row_pct_df = row_growth_df.pct_change() * 100
            # row_pct_df = row_pct_df.fillna(0) # pct_change first row is NaN
            row_pct_df = row_pct_df.iloc[1:]
            st.write(f"列維度 ({st.session_state['pivot_row']}) 年增率 (%):")
            st.dataframe(row_pct_df.style.format("{:,.2f}%").background_gradient(cmap='RdYlBu', vmin=-10, vmax=10))

    with growth_tabs[2]:
        if col_totals_year:
            col_growth_df = pd.DataFrame(col_totals_year)
            col_growth_df['DATA_YR'] = pd.to_numeric(col_growth_df['DATA_YR'])
            col_growth_df = col_growth_df.sort_values(by='DATA_YR').set_index('DATA_YR')
            # Calculate pct change
            col_pct_df = col_growth_df.pct_change() * 100
            # col_pct_df = col_pct_df.fillna(0)
            col_pct_df = col_pct_df.iloc[1:]
            st.write(f"欄維度 ({st.session_state['pivot_col']}) 年增率 (%):")
            st.dataframe(col_pct_df.style.format("{:,.2f}%").background_gradient(cmap='RdYlBu', vmin=-10, vmax=10))

    # # 視覺化
    # select_trend = st.selectbox("選擇項目:", trend_base_df[st.session_state['pivot_row']].unique())
    # pivot_trend = trend_base_df[trend_base_df[st.session_state['pivot_row']] == select_trend]
    # pivot_trend = pivot_trend.pivot_table(index='DATA_YR', columns=st.session_state['pivot_col'], values=st.session_state['pivot_sum'], aggfunc='sum')
    # st.line_chart(pivot_trend)


    # Save Button
    # 可以設定表的名稱，例如我打 表1-1，然後會自行以 表1-1({PIVOT_ROW x PIVOT_COL}) 儲存名稱，必記下所有條件
    # 包含 DATA_SRC,PIVOT_ROW, PIVOT_COL, PIVOT_SUM, FILTERS
    # 儲存到 tables.yaml
    # 每次使用此頁面時，可以在 side_bar 選擇紀錄，更新，就可以套用設定
