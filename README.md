# Inductance Analysis Tool

This repository contains a Python script for analyzing and visualizing the inductance behavior of an inductor based on measured current and voltage signals. The tool processes data from a CSV file, applies noise filtering, computes the rate of change of current (\( dI/dt \)), and calculates inductance (\( L \)).

## Features

- **Low-Pass Filtering**: Removes high-frequency noise from the voltage signal using a Butterworth filter.
- **Custom Gradient Calculation**: Computes \( dI/dt \) over adjustable intervals, tailored for different current ranges.
- **Inductance Calculation**: Derives inductance values using the relationship \( L = V / (dI/dt) \).
- **Data Visualization**:
  - Plot of inductance vs. current.
  - Plot of current and voltage (filtered and unfiltered) vs. time.
- **CSV Export**: Saves processed data, including computed inductance, to a CSV file.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/inductance-analysis-tool.git
   cd inductance-analysis-tool
