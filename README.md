# AAD Course Project — Convex Hull Animations (Manim)

This repository contains clean Manim animations for classic convex-hull algorithms used in our AAD course project:
- Chan’s Algorithm (`ChansAlgorithm.py`)
- Graham Scan (`grahamscan.py`)
- Jarvis March / Gift Wrapping (`Jarvisalgorithm.py`)
- QuickHull (`Quickhull.py`)
- Monotone Chain / Andrew’s Algorithm (`monotonechain.py`)
- Divide & Conquer (merge-based) (`DivideAndConquer.py`)

Each file exposes a Manim scene named `Algorithm`. You can render any of them with Manim. Minimal non-visual (console) variants are included in the same files.

## Recommended installation (Micromamba)

If you don’t already have Manim, we recommend using Micromamba for a consistent, cross‑platform setup.

1) Install Micromamba (Linux/WSL/macOS)
- Docs: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
- One‑liner (Linux/WSL):

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

2) Create and activate an environment, then install Manim

```
micromamba create -n manim -c conda-forge python=3.12
micromamba activate manim
micromamba install -c conda-forge manim
```

3) Optional — LaTeX for MathTex text (TinyTeX shown for Linux)

```
wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
tlmgr install amsmath babel-english cm-super dvisvgm fontspec latex-bin microtype preview rsfs standalone xcolor xetex xkeyval
tlmgr path add
```

Verify setup:

```
manim checkhealth
```

Alternatives: Conda (Miniconda/Anaconda) with `conda install -c conda-forge manim`, or pip + venv using `requirements.txt` in this folder.

## How to render (from this folder)

Quality presets: `-pql` (fast), `-pqm`, `-pqh` (1080p) and `-p` to auto‑open.

```
# Chan’s Algorithm
manim -pqh ChansAlgorithm.py Algorithm

# Graham Scan
manim -pqh grahamscan.py Algorithm

# Jarvis March
manim -pqh Jarvisalgorithm.py Algorithm

# QuickHull
manim -pqh Quickhull.py Algorithm

# Monotone Chain (Andrew)
manim -pqh monotonechain.py Algorithm

# Divide & Conquer (merge-based)
manim -pqh DivideAndConquer.py Algorithm
```

Outputs are saved under `media/videos/<ModuleName>/<quality>/Algorithm.mp4`.

## Console‑only runs (no animation)

```bash
python ChansAlgorithm.py
python grahamscan.py
python Jarvisalgorithm.py
python Quickhull.py
python monotonechain.py
python DivideAndConquer.py
```

## Benchmarking & Performance Analysis

This project includes two comprehensive benchmark scripts:

### 1. Distribution-Based Benchmark (`comprehensive_benchmark_fixed.py`)
Tests algorithms on 3 classic distributions:
- **Uniform Random**: Average case performance
- **Circular**: Worst case (h = n, all points on hull)
- **Clustered**: Best case (small h, few hull points)

**Run:**
```bash
python comprehensive_benchmark_fixed.py
```

**Generates:**
- 6 tables (execution time + hull size for each distribution)
- 4 graphs (3 distribution comparisons + 1 combined view)
- Output: `media/graphs/comparison_*.png`

### 2. Shape-Based Benchmark (`shape_distribution_benchmark.py`)
Tests algorithms on 6 geometric shapes:
- **Square**: Points in square with boundary
- **Disc**: Points in circle with circumference
- **Circle**: All points on perimeter (worst case, h=n)
- **Star**: 5-pointed star (best case, h=5)
- **Though (Cloud)**: Irregular blob shape
- **Line**: Collinear points (degenerate case)

**Run:**
```bash
python shape_distribution_benchmark.py
```

**Generates:**
- Summary table comparing all algorithms across shapes
- Single graph with 2×3 grid (6 subplots, one per algorithm)
- Shows Time vs n for all 6 shapes on each algorithm
- Output: `media/graphs/shape_distribution_comparison.png`



## Requirements (pip users)

See `requirements.txt` in this folder. Installing via Micromamba/Conda can skip this file.

## Notes

- Python 3.10–3.12 recommended; CPU rendering is fine (GPU not required).
- If LaTeX isn’t installed, MathTex labels may fail; use `Text` or install TinyTeX/TeX Live/MiKTeX.
- This is a GitHub repo for our AAD course project.
- Housekeeping: large Manim outputs in `media/` are ignored by `.gitignore` by default to keep the repo clean, while PDFs are kept tracked.
