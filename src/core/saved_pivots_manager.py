import os
import ast
from typing import List, Dict, Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "saved_pivots.py")

def load_saved_pivots() -> List[Dict]:
    """
    Load saved pivot configurations from config/saved_pivots.py.
    """
    if not os.path.exists(CONFIG_PATH):
        return []
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            # Parse the python file to extract the saved_pivots list
            # We expect the file to contain: saved_pivots = [...]
            
            # Simple fallback if empty
            if not content.strip():
                return []
                
            # Execute the file in a safe namespace to get the variable
            namespace = {}
            exec(content, {}, namespace)
            return namespace.get("saved_pivots", [])
    except Exception as e:
        print(f"Error loading saved pivots: {e}")
        return []

def _save_pivots_to_file(pivots: List[Dict]):
    """
    Helper to write the pivots list to file.
    """
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("# Saved Pivot Table Configurations\n")
        f.write("# This file is auto-generated. You can also edit it manually.\n\n")
        f.write(f"saved_pivots = {repr(pivots)}\n")

def save_pivot_config(new_config: Dict):
    """
    Save a new pivot configuration to config/saved_pivots.py.
    """
    pivots = load_saved_pivots()
    pivots.append(new_config)
    _save_pivots_to_file(pivots)

def update_pivot_config(index: int, updated_config: Dict):
    """
    Update an existing pivot config by index.
    """
    pivots = load_saved_pivots()
    if 0 <= index < len(pivots):
        pivots[index] = updated_config
        _save_pivots_to_file(pivots)

def delete_pivot_config(index: int):
    """
    Delete a pivot config by index.
    """
    pivots = load_saved_pivots()
    if 0 <= index < len(pivots):
        pivots.pop(index)
        _save_pivots_to_file(pivots)

def save_all_pivots(pivots: List[Dict]):
    """
    Overwrite all saved pivots with the new list.
    Useful for batch updates or reordering.
    """
    _save_pivots_to_file(pivots)
