# LaPSEE Minicourse — Introduction to Power Networks as Intelligent Graphs

Materials for the UNESP / LaPSEE graduate minicourse **"Introduction to Power
Networks as Intelligent Graphs: From Topology and Spatial Econometrics to
Multi-Agent Decision Making"**.

## Contents

### Session 1 — From Circuits to Graphs, From Graphs to Space

| File | Purpose |
|------|---------|
| `s01_slides.pdf` | Lecture slide deck (projected in class) |
| `s01_reading-student.pdf` | Pre- and post-class student reading |
| `s01_graphs-and-space.ipynb` | Lab notebook (Labs 1a + 1b + capstone) |
| `lapsee_s01.py` | Helper module imported by the notebook |

### Session 2 — Spatial Analysis of Prices and Generation Siting

| File | Purpose |
|------|---------|
| `s02_spatial_3.pdf` | Lecture slide deck (projected in class) |
| `s02_reading-student.pdf` | Pre- and post-class student reading |
| `s02_spatial-prices-siting.ipynb` | Lab notebook (Labs 2a + 2b + capstone) |

### Shared

| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned Python environment (covers all sessions) |

## Prerequisites

- **Python 3.10 or newer** (3.10, 3.11, or 3.12 — `pandapower 2.14` is not
  yet validated on 3.13).
- A working LaTeX installation is **not** required: the PDFs are pre-built.
- ~1 GB of disk space for the virtual environment.

## Setup (do this once, before class)

### macOS / Linux

```bash
cd <folder containing this README>
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd <folder containing this README>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

The install pulls a large scientific stack (NumPy, pandas, NetworkX,
pandapower, PySAL, scikit-learn, Mesa, PettingZoo, stable-baselines3,
PyTorch CPU). On a typical laptop expect 5–10 minutes.

## Launching the notebook

With the virtual environment active:

```bash
jupyter notebook s01_graphs-and-space.ipynb
```

or with JupyterLab:

```bash
jupyter lab s01_graphs-and-space.ipynb
```

The notebook uses the IEEE 30-bus test system from `pandapower.networks`
— **no external data files are required**.

## Reproducibility

The notebook sets `RANDOM_SEED = 2026` in its first cell. Any stochastic
result (Moran's I permutation test, Motter–Lai random-attack baseline) is
reproducible if the same seed is preserved.

The optional capstone exercise is gated behind `RUN_CAPSTONE = False` at
the bottom of the notebook. Flip it to `True` to run the end-of-session
integration exercise.

## Troubleshooting

- **`pandapower` import error mentioning `numpy.float_`**: your NumPy is
  too new. The pin `numpy<2.0` in `requirements.txt` should prevent this;
  if you bypassed the pin, reinstall with `pip install -r requirements.txt
  --force-reinstall`.
- **`spreg` / `libpysal` import warnings about `pkg_resources`**: harmless
  deprecation warnings from the PySAL stack, safe to ignore.
- **Slow first cell**: the first `pandapower.networks.case30()` call
  triggers JSON deserialization of the case file from the installed
  package — subsequent calls are fast.

## License and attribution

Course materials © 2026 LaPSEE / UNESP, authored for the graduate
minicourse. IEEE 30-bus data ships with `pandapower` under its own
license.
