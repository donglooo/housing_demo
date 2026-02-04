"""
Configuration and path management utilities.

This module handles configuration paths, dataset type detection,
and codebook section retrieval.
"""

import os
from pathlib import Path
from typing import Dict, Tuple


# Dataset prefix to codebook section mapping
CODEBOOK_MAPPING = {"稅電": "power", "所有權": "ownership", "稅籍": "tax"}

# Columns/Prefixes to exclude from generic filtering logic (Metrics/IDs)
EXCLUDED_METRIC_PREFIXES = ("CNT_", "SUM_", "AVG_", "PCT_")
EXCLUDED_METRIC_COLS = {"DATA_STATUS", "CNT", "GID", ""}


def get_base_paths() -> Tuple[str, str, str]:
    """
    Get the base directory paths for the project.

    Returns:
        Tuple of (BASE_DIR, DATA_DIR, CONFIG_DIR)
    """
    # Assuming this module is in src/core/, go up 2 levels to get BASE_DIR
    current_file = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CONFIG_DIR = os.path.join(BASE_DIR, "config")

    return BASE_DIR, DATA_DIR, CONFIG_DIR


def resolve_data_path(raw_path: str, base_dir: str = None) -> str:
    """
    Convert relative paths to absolute paths.

    Args:
        raw_path: The path to resolve (may be relative or absolute)
        base_dir: Base directory to resolve relative paths against

    Returns:
        Absolute path to the data file
    """
    if os.path.isabs(raw_path):
        return raw_path

    if base_dir is None:
        base_dir, _, _ = get_base_paths()

    return os.path.join(base_dir, raw_path)


def detect_dataset_type(file_path: str) -> str:
    """
    Detect dataset type from filename prefix.

    Args:
        file_path: Path to the data file

    Returns:
        Dataset type key (e.g., 'power', 'ownership', 'tax')

    Raises:
        ValueError: If the prefix is not recognized
    """
    filename = Path(file_path).name
    prefix = filename.split("_coded")[0]

    if prefix not in CODEBOOK_MAPPING:
        raise ValueError(
            f"Unknown dataset prefix: '{prefix}'. "
            f"Expected one of: {list(CODEBOOK_MAPPING.keys())}"
        )

    return CODEBOOK_MAPPING[prefix]


def get_codebook_section(codebook: Dict, dataset_type: str) -> Dict:
    """
    Extract the relevant section from the codebook.

    Args:
        codebook: The full codebook dictionary
        dataset_type: The dataset type key (e.g., 'power', 'ownership', 'tax')

    Returns:
        The codebook section for the specified dataset type

    Raises:
        KeyError: If the dataset type is not found in the codebook
    """
    if dataset_type not in codebook:
        raise KeyError(
            f"Dataset type '{dataset_type}' not found in codebook. "
            f"Available types: {list(codebook.keys())}"
        )

    return codebook[dataset_type]
