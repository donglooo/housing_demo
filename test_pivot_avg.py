import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

print("Script starting...")
try:
    from src.core.pivot_engine import compute_pivot_tables

    print("Import successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def test_pivot_averages():
    # Create dummy data
    df = pd.DataFrame(
        {
            "DATA_YR": [2021, 2021, 2022, 2022],
            "CITY": ["A", "B", "A", "B"],
            "CNT": [10, 20, 30, 40],
            "SUM_AGE": [100, 400, 600, 1600],
            "CNT_AGE": [10, 20, 30, 40],
        }
    )

    # Mock codebook
    codebook = {"CITY": {"name": "城市", "avg": {"num": "SUM_AGE", "den": "CNT_AGE"}}}

    # Test case:
    pivot_tab = "DATA_YR"
    pivot_rows = ["CITY"]
    pivot_cols = []
    pivot_sum = "CNT"
    filter_items = ()

    print("Running compute_pivot_tables with average config...")
    unique_tabs, results, row_totals, col_totals, all_totals, masked_df, ref_totals = (
        compute_pivot_tables(
            df, pivot_tab, pivot_rows, pivot_cols, pivot_sum, filter_items, codebook
        )
    )

    for tab, res in results.items():
        print(f"\nTab: {tab}")
        pivot = res["pivot"]
        print("Pivot Content:\n", pivot)

        avg_data = res.get("avg_data")
        if avg_data:
            print("Avg Data Found")
            print("Rows Avg:\n", avg_data["rows_avg"])
            print("Total Avg:\n", avg_data["total_avg"])

            # Verify Calculation
            # 2021 A: SumAge 100 / CntAge 10 = 10.0
            # 2021 B: SumAge 400 / CntAge 20 = 20.0
            # Total: 500 / 30 = 16.66

            rows_avg = avg_data["rows_avg"]
            total_avg = avg_data["total_avg"]

            if abs(rows_avg.loc["A", "平均城市"] - 10.0) < 0.001:
                print("PASS: Row A average correct")
            else:
                print(f"FAIL: Row A average {rows_avg.loc['A', '平均城市']} != 10.0")

            if abs(total_avg["平均城市"] - (500 / 30)) < 0.001:
                print("PASS: Total average correct")
            else:
                print(f"FAIL: Total average {total_avg['平均城市']} != {500 / 30}")
        else:
            print("FAIL: No avg_data returned")


if __name__ == "__main__":
    test_pivot_averages()
