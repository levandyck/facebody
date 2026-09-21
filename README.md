# Faces and bodies are increasingly integrated along the visual hierarchy in humans and deep neural networks

This repository contains analysis code for the paper [*Faces and bodies are increasingly integrated along the visual hierarchy in humans and deep neural networks*](https://doi.org/10.64898/2026.02.16.706115) by Leonard E. van Dyck and Katharina Dobs.

## Installation
Python 3.10.
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e . # core
pip install -e ".[fmri]" # + nilearn, for the localizer GLM
pip install -e ".[flatmaps]" # + pycortex, for the flatmap figures
```
`requirements.txt` pins the versions the analyses were run with.

## Configuration
All paths come from three environment variables (`src/facebody/config.py`).
```bash
export FACEBODY_DATA_ROOT=/path/to/data # datasets/ and models/
export FACEBODY_PROJECT_ROOT=/path/to/output # analysis output
export FACEBODY_FIG_ROOT=/path/to/figs # figures
```

## Data
The Natural Scenes Dataset is available [here](https://naturalscenesdataset.org/)
