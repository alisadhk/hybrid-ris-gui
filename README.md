<h1 align="center">Hybrid RIS - DRL Visualizer</h1>

<p align="center"><b>Interactive GUI for a hybrid active-passive RIS system with a DRL approach</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white" alt="Platform">
  <img src="https://img.shields.io/badge/updated-2025--09--08-success" alt="Updated">
  <a href="https://github.com/alisadhk/hybrid-ris-gui/releases/latest"><img src="https://img.shields.io/badge/downloads-Releases-blue" alt="Releases"></a>
</p>

---

## Overview
A PyQt5 + Matplotlib desktop application to configure and visualize a hybrid active-passive reconfigurable intelligent surface (RIS) scenario against eavesdropping and jamming, driven by a deep reinforcement learning (DRL) approach. It uses a dark theme and shows tooltips on the plots without changing them.

## Features
- Dark mode UI with default Matplotlib colors
- Parameter panels (Main Params, Settings) on the left, plots on the right
- Position inputs in (x, y, z) form
- Run button that regenerates all plots
- Non-intrusive Qt tooltips over the plots
- Four result plots: secrecy rate vs number of hybrid-RIS elements, average reward vs episodes (various learning rates), secrecy rate vs total transmit power, and secrecy rate vs active elements

## How it works
- **Main Params** hold the system model: BS antennas (M), jammer antennas (L), RIS elements (N), active elements (N_a), maximum powers, noise, path loss and Rician K-factor.
- **Settings** hold the node positions and the DRL hyper-parameters (learning rates, discount, batch size, replay buffer, target update, episodes, steps).
- Press **- Run -** to compute the curves; the four plots refresh together.
- Hover a plot to read exact values through a non-intrusive tooltip.

## Requirements
- Python 3.10+ with PyQt5, numpy and matplotlib.
- Windows, Linux or macOS with a desktop session.

## Screenshots
<p align="center"><img src="screenshots/screenshot-1.png" alt="Hybrid RIS - DRL Visualizer screenshot" width="720"></p>

## Download and install
### Option A - Download
No prebuilt binary is published for this project. Use Option B below.

### Option B - Run from source
Requirements: Windows 10/11 and Python 3.10+.

```bash
git clone https://github.com/alisadhk/hybrid-ris-gui.git
cd hybrid-ris-gui
python -m pip install -r requirements.txt
python hybrid_ris_black.py
```

## How to use
1. Start the application with `python hybrid_ris_black.py`.
2. Set the main parameters and the node positions (x, y, z) in the left panel.
3. Adjust the extra values in the **Settings** tab.
4. Press **- Run -** and inspect the plots on the right; hover the plot to read values.

## Configuration and privacy
- No credentials are used by this project.

## Troubleshooting
- Plots are empty at start: that is normal, press **Run**.
- Slow first run: matplotlib builds its font cache once.
- Missing module error: python -m pip install -r requirements.txt.

## Build the executable yourself
```bash
python -m pip install pyinstaller
python -m PyInstaller --noconfirm 
```
The executable is created in the `dist/` folder.

## Notes
- The related research paper is not included for copyright reasons.

---

<p align="center">Maintained by <a href="https://github.com/alisadhk">@alisadhk</a> - last update 2025-09-08</p>