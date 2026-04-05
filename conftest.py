"""
Root conftest.py
=================
Adds all module directories to sys.path so test imports work correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Add module directories so tests can do `from data_structures.hash_table import ...`
for module_dir in [
    ROOT / "02_algorithms",
    ROOT / "03_design_patterns",
    ROOT / "04_testing",
]:
    path_str = str(module_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
