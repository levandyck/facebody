"""Filesystem roots for datasets, project outputs, and figures."""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("FACEBODY_DATA_ROOT", "./data"))
PROJECT_ROOT = Path(os.environ.get("FACEBODY_PROJECT_ROOT", DATA_ROOT / "projects" / "facebody"))
FIG_ROOT = Path(os.environ.get("FACEBODY_FIG_ROOT", "./figs"))

SUBJECTS = tuple(f"subj{i:02d}" for i in range(1, 9))
