import os
import ast
import pandas as pd  # if needed for repr?

CONFIG_PATH = (
    r"c:\Users\10523\Desktop\Dong\99_GitHub\housing-research\config\saved_pivots.py"
)
TARGET_PATH = r"c:\Users\10523\Desktop\Dong\99_GitHub\housing-research\data\260204\所有權_coded_202602040949.parquet"


def fix_config():
    if not os.path.exists(CONFIG_PATH):
        print("Config not found.")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    namespace = {}
    exec(content, {}, namespace)
    pivots = namespace.get("saved_pivots", [])

    for p in pivots:
        # Fix Path
        # If it points to the standard dataset, replace with local absolute path
        # Or just checking if it ends with the known filename
        ds = p.get("data_source", "")
        if "所有權_coded_202602040949.parquet" in ds:
            # Just force update to correct local path
            p["data_source"] = TARGET_PATH

        # Fix Filters (from string to dict)
        filters = p.get("filters")
        if isinstance(filters, str):
            try:
                p["filters"] = ast.literal_eval(filters)
                print(f"Fixed filters for {p['chapter']}")
            except:
                print(f"Failed to parse filters for {p['chapter']}")

    # Write back
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("# Saved Pivot Table Configurations\n")
        f.write("# This file is auto-generated. You can also edit it manually.\n\n")
        f.write(f"saved_pivots = {repr(pivots)}\n")

    print("Done fixing config.")


if __name__ == "__main__":
    fix_config()
