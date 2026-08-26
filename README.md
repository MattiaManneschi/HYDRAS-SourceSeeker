# HYDRAS - Hydrodynamic-aware Distributed Robots for Marine Source-Seeking

Marine pollutant source localization with autonomous underwater vehicles (AUVs).
The project compares a gradient method (Field Climbing Method) and a Reinforcement
Learning approach (PPO) in guiding an agent up the concentration field to the
source, accounting for wind and sea current. Data come from MIKE21 hydrodynamic
simulations of the Cecina bay.

## How to use it (macOS · Windows)

1. Download the launcher for your system — **`launcher.command`** (macOS) or
   **`launcher.bat`** (Windows) — and place it in a folder with at least ~8 GB free.
2. Double-click it. On first launch it downloads everything it needs (code, data,
   models) into that same folder, then opens the interface.
3. Tick one or more **models** to compare and set the scenario (see below), then
   press **Start** to watch them run together, in real time, on the same scenario.

### Requirements

To run the launcher you only need to provide **Python** — it installs every Python
package it needs by itself (via `pip`). Specifically:

- **Python 3.9 or newer** (a recent build, e.g. 3.11+, is recommended), and it must
  include **Tcl/Tk** (the `tkinter` module that draws the interface). Install it from
  [python.org](https://www.python.org/downloads/) — on Windows keep *"tcl/tk and
  IDLE"* checked in the installer. The version floor is not optional: the launcher
  installs **NumPy 2.x**, which the trained models require (they are pickled with
  NumPy 2.x), and NumPy 2.x needs Python ≥ 3.9.
- An **internet connection** on first launch: it downloads the code, ~7.4 GB of data
  and the models. Later launches reuse the local copy and start offline.
- About **8 GB of free disk space** in the launcher's folder.

`curl`/`wget` (macOS) or `curl`/PowerShell (Windows), used for the very first
download, are already bundled in recent macOS and Windows 10/11.

### What to choose

Pick **one or more models** to compare — they run together on the same scenario,
from a common start, each drawn in its own colour — plus the scenario:

| Control | Options | Meaning |
|---|---|---|
| **Models** (checkboxes) | FCM · PPO no ring · PPO single ring · PPO double ring | Which agents to show. Any subset runs simultaneously. |
| **Wind** | V0 · V1 · V2 · V3 | Wind scenario (four hydrodynamic runs with different wind conditions). |
| **Time Chunk** | Q1/4 · Q1/2 · Q3/4 | When in the simulation the episode starts — first quarter, middle, or third quarter (plume more or less dispersed). |
| **Max Speed** | 0.1–5 m/s | Agents' maximum speed. The FCM step is derived from it; the *no ring* model exists only up to 2 m/s (skipped above). |

Once **Wind** and **Time Chunk** are set, a random source is picked for that
scenario. Press **Start** again for a new scenario.

## Repository structure

- `src/` — training (`train_ppo`, `run_adaptive_sweeps`), evaluation/inference
  (`inference`, `explainability`) and the live app (`live_sim`).
- `utils/` — environment, data loading, and video generation (`video_generator`,
  with a standalone CLI: `python utils/video_generator.py batch all`).
- `thesis/` — thesis deliverables: LaTeX source (`.zip`), the three PDF/A files
  (thesis, abstract, index) and all report PDFs (`reports/`).
- `deliverables/` — figures, tables and the comparison videos (`videos/v0|v1|v2/`).
- `evaluations/` — held-out **analysis plots** and **lean** result files
  (`episodes_data_lean.json`); per-episode plots and full JSONs stay local.
- `data/`, `trained_models/` — large assets kept out of git (fetched by the
  launcher / distributed via GitHub Release).
