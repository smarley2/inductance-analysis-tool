# Inductance Analysis Tool

This repository contains a Python script for analyzing and visualizing the inductance behavior of an inductor based on measured current and voltage signals. The tool processes data from a CSV file, applies noise filtering, computes the rate of change of current (\( dI/dt \)), and calculates various inductor parameters.

## Features

- **Low-Pass Filtering**: Removes high-frequency noise from the voltage signal using a Butterworth filter.
- **Custom Gradient Calculation**: Computes \( dI/dt \) over adjustable intervals, tailored for different current ranges.
- **Advanced Inductor Parameter Calculations**:
  - **Incremental Inductance** (\(L_\text{inc}\)): Local inductance based on the slope of the flux linkage vs. current curve.
  - **Secant Inductance** (\(L_\text{sec}\)): Average inductance over the range of current and flux linkage.
  - **Flux Linkage** (\( \Phi \)): Cumulative integration of voltage, representing the total magnetic flux linked to the coil.
  - **Magnetic Co-Energy** (\(W_\text{co}\)): Energy stored in the magnetic field, computed as an integral of flux linkage and current.
  - **Flux Density** (\(B\)): Normalized magnetic flux per unit cross-sectional area of the core.
- **Data Visualization**:
  - Plot of incremental inductance and secant inductance vs. current.
  - Plot of current, voltage (filtered and unfiltered), and flux density vs. time.
- **CSV Export**: Saves processed data, including computed parameters, to a CSV file.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/smarley2/inductance-analysis-tool.git
   cd inductance-analysis-tool
