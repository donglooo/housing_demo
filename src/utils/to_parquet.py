# Transform Csv to Parquet
import os
import pandas as pd

# INPUT FOLDER DATE: 260128
input_dt = input("Please enter the input folder date (e.g. 260128): ")
input_folder = os.path.join("data", input_dt)

# GET ALL CSV FILES
input_files = os.listdir(input_folder)
input_files = [f for f in input_files if f.endswith(".csv")]

# TRANSFORM CSV TO PARQUET
for input_file in input_files:
    input_path = os.path.join(input_folder, input_file)
    output_path = os.path.join(input_folder, input_file.replace(".csv", ".parquet"))
    df = pd.read_csv(input_path)
    df.to_parquet(output_path)
    print(f"[SUCCESS] Transformed {input_file} to {output_path}")