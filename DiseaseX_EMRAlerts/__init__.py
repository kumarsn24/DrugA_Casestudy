"""DiseaseX_EMRAlerts package shim.

This package ensures imports like `DiseaseX_EMRAlerts.fastapi_app` work even when the
original fastapi_app.py resides at the repository root. It inserts the repository root
onto sys.path and imports the top-level module.
"""
import sys
from pathlib import Path

# Ensure the repository root (parent of this package) is on sys.path so we can import
# modules that live at the repository root (legacy layout).
repo_root = Path(__file__).resolve().parent.parent
root_str = str(repo_root)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Re-export any top-level modules as package modules by importing them when needed.
__all__ = []

