"""
utils.py - General Utility Functions

Provides helper routines for directory setup, JSON/CSV serialization, 
and experiment configuration handling.
"""

import os
import json
from typing import Dict, Any


def ensure_directories(base_dir: str = ".") -> None:
    """Ensure all required output directories exist."""
    directories = [
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "results", "raw"),
        os.path.join(base_dir, "results", "resource_profiles"),
        os.path.join(base_dir, "figures"),
        os.path.join(base_dir, "report")
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)


def save_metadata(metadata: Dict[str, Any], filepath: str) -> None:
    """Save experiment metadata to a formatted JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(filepath: str) -> Dict[str, Any]:
    """Load metadata from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
