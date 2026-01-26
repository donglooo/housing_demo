import streamlit as st
import pandas as pd
import yaml
import os

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.config_dir = os.path.join(base_dir, 'config')
        self.data_dir = os.path.join(base_dir, 'data')
    
    @st.cache_data(ttl=1800, max_entries=3)
    def load_parquet(_self, file_path):
        """載入 Parquet 檔案"""
        try:
            return pd.read_parquet(file_path, engine='pyarrow')
        except Exception as e:
            st.error(f"載入資料失敗: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)
    def load_codebook(_self, codebook_path):
        """載入 codebook"""
        try:
            with open(codebook_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"載入 codebook 失敗: {str(e)}")
            return None
    
    def decode_data(self, df, codebook):
        """解碼資料"""
        df_decode = df.copy()
        for col in codebook.get('mappings', {}).keys():
            if col in df_decode.columns:
                mapping = codebook['mappings'][col].get('codes', {})
                df_decode[col] = df_decode[col].replace(mapping)
        return df_decode