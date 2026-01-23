# Housing Research Data Processor

這是一個用於處理稅籍資料並產出標準化 Excel 報表的工具。

## 功能

- **資料處理**：上傳經過編碼的 CSV 檔案（如 `稅籍_coded_*.csv`），系統將自動轉換代碼為文字（如將 `11` 轉為 `15坪以下`），並產出格式化的 Excel 報表。
- **設定管理**：透過介面直接修改 `codebook.yaml`，無需更動程式碼即可調整代碼對照表或欄位排序。

## 如何使用

1.  確認已安裝 Python 環境並執行 `pip install -r requirements.txt` (或直接使用提供的執行檔)。
2.  執行程式：
    ```bash
    streamlit run app.py
    ```
3.  開啟瀏覽器（預設為 `http://localhost:8501`）。
4.  在左側選單選擇 **Data Processing**。
5.  上傳 CSV 檔案。
6.  點擊 **Run Processing**。
7.  預覽結果並點擊 **Download Styled Excel** 下載報表。

## 設定修改

若需修改代碼對應或是欄位順序：

1.  在左側選單選擇 **Settings**。
2.  直接編輯 YAML 內容。
3.  點擊 **Save Settings** 即時生效。
