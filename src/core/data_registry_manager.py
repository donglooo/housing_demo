import os
import glob
from typing import List, Dict, Set
from src.core.config_manager import get_base_paths

# Path to the config file
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "data_registry.py")

def load_disabled_files() -> Set[str]:
    """
    Load the set of disabled filenames from config/data_registry.py.
    Returns:
        Set of disabled filenames (just the basename).
    """
    if not os.path.exists(CONFIG_PATH):
        return set()
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            # Parse python file
            namespace = {}
            exec(content, {}, namespace)
            return set(namespace.get("disabled_files", []))
    except Exception as e:
        print(f"Error loading data registry: {e}")
        return set()

def save_disabled_files(disabled_set: Set[str]):
    """
    Save the set of disabled filenames to config/data_registry.py.
    """
    # Convert to sorted list for stable file output
    disabled_list = sorted(list(disabled_set))
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("# Data Registry Configuration\n")
        f.write("# This file tracks which data files are disabled (hidden) from the Playground.\n")
        f.write("# Uses a blacklist approach: All files are enabled by default unless listed here.\n\n")
        f.write(f"disabled_files = {repr(disabled_list)}\n")

def get_all_data_files() -> List[str]:
    """
    Scan the data directory for all parquet files.
    Returns:
        List of absolute paths to all data files.
    """
    _, DATA_DIR, _ = get_base_paths()
    return glob.glob(os.path.join(DATA_DIR, "**", "*.parquet"), recursive=True)

def get_available_files() -> List[str]:
    """
    Get list of data files that are NOT disabled.
    Returns:
        List of absolute paths to enabled data files.
    """
    all_files = get_all_data_files()
    disabled_set = load_disabled_files()
    
    available = []
    for fpath in all_files:
        fname = os.path.basename(fpath)
        if fname not in disabled_set:
            available.append(fpath)
            
    return available

def toggle_file_status(filename: str, enable: bool):
    """
    Update the status of a file.
    Args:
        filename: The basename of the file (e.g., 'data.parquet')
        enable: True to enable (remove from blacklist), False to disable (add to blacklist).
    """
    disabled_set = load_disabled_files()
    
    if enable:
        if filename in disabled_set:
            disabled_set.remove(filename)
    else:
        disabled_set.add(filename)
        
    save_disabled_files(disabled_set)
