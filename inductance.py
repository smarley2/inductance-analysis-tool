import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# User-configurable variables
file_path = "example/Tek001_ALL.csv"  # Path to the CSV file
skip_rows = 16  # Number of rows to skip in the CSV
time_min = 0.5e-6  # Minimum time of csv data (in seconds)
time_max = 120e-6  # Maximum time of csv data (in seconds)
cutoff_frequency = 5e5  # Cutoff frequency for low-pass filter (in Hz)
low_pass_filter_order = 4  # Order of the Butterworth low-pass filter
interval_ranges = {  # Interval ranges for gradient calculation based on current ranges
    (0, np.inf): 2000  # Interval for current range 0 A to max
}
output_file = "inductance_vs_current.csv"  # Output file to save processed data

# Physical constants
N = 31  # Number of turns
Ae = 188.3e-6  # Cross-sectional area in m^2
flux_initial = 0  # Initial flux linkage in Weber

# Define column names for the input CSV file
time_column = "TIME"  # Column name for time
current_column = "CH8"  # Column name for current
voltage_column = "CH3"  # Column name for voltage

# Define a low-pass Butterworth filter
def low_pass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the data.

    Parameters:
        data (array): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        array: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Custom function to calculate gradient with a larger interval
def custom_gradient(y, x, interval):
    dy_dx = np.empty_like(y)
    dy_dx[:] = np.nan  # Default to NaN for out-of-bounds calculations

    for i in range(interval, len(y) - interval):
        dy_dx[i] = (y[i + interval] - y[i - interval]) / (x[i + interval] - x[i - interval])
    return dy_dx

# Load the CSV file and skip the required number of rows
data = pd.read_csv(file_path, skiprows=skip_rows)

# Rename the columns for clarity
data.rename(columns={time_column: 'time', current_column: 'current', voltage_column: 'voltage'}, inplace=True)

# Filter time values and explicitly create a copy
filtered_data = data[(data['time'] > time_min) & (data['time'] < time_max)].copy()

# Extract time, current, and voltage
time = filtered_data['time'].values
current = filtered_data['current'].values
voltage = filtered_data['voltage'].values

# Calculate the sampling frequency from time array
sampling_frequency = 1 / np.mean(np.diff(time))  # Sampling frequency in Hz

# Apply low-pass filter to voltage
filtered_voltage = low_pass_filter(voltage, cutoff_frequency, sampling_frequency, order=low_pass_filter_order)

# Initialize dI/dt
dI_dt = np.zeros_like(current)

# Apply different gradients based on current ranges
for (min_current, max_current), interval in interval_ranges.items():
    range_indices = (current >= min_current) & (current < max_current)
    current_subset = current[range_indices]
    time_subset = time[range_indices]

    # Compute gradient only for the subset
    gradient_subset = custom_gradient(current_subset, time_subset, interval)

    # Assign the computed gradient back to the appropriate indices
    dI_dt[range_indices] = gradient_subset

# Avoid division by zero
dI_dt[dI_dt == 0] = np.nan  # Replace zero with NaN to prevent division errors

# Calculate incremental inductance
incremental_inductance = filtered_voltage / dI_dt

# Calculate flux linkage
flux_linkage = np.cumsum(filtered_voltage) * np.mean(np.diff(time)) + flux_initial

# Calculate secant inductance
secant_inductance = flux_linkage / current
secant_inductance[current == 0] = np.nan  # Avoid division by zero

# Calculate magnetic co-energy
magnetic_co_energy = np.cumsum(flux_linkage * np.gradient(current)) * np.mean(np.diff(time))

# Calculate flux density
flux_density = flux_linkage / (N * Ae)

# Add calculated values to the DataFrame using .loc
filtered_data.loc[:, 'incremental_inductance'] = incremental_inductance
filtered_data.loc[:, 'secant_inductance'] = secant_inductance
filtered_data.loc[:, 'flux_linkage'] = flux_linkage
filtered_data.loc[:, 'magnetic_co_energy'] = magnetic_co_energy
filtered_data.loc[:, 'flux_density'] = flux_density


# Plotting incremental inductance vs current
plt.figure()
plt.plot(filtered_data['current'], filtered_data['incremental_inductance'], label="Incremental Inductance")
plt.xlabel("Current (A)")
plt.ylabel("Inductance (H)")
plt.title("Incremental Inductance vs Current")
plt.grid()
plt.legend()
plt.show()

# Plot the current, unfiltered voltage, and filtered voltage on the primary y-axis
plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(time, current, label="Current (A)", alpha=0.7)
ax1.plot(time, voltage, label="Unfiltered Voltage (V)", alpha=0.7)
ax1.plot(time, filtered_voltage, label="Filtered Voltage (V)", alpha=0.7, linestyle="--")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.legend(loc="upper left")
ax1.grid()

# Create a secondary y-axis for flux density
ax2 = ax1.twinx()

# Plot flux density on the secondary y-axis
ax2.plot(time, flux_density, label="Flux Density (B)", color="purple", linestyle="--", alpha=0.7)
ax2.set_ylabel("Flux Density (B)")
ax2.legend(loc="lower right")

# Set the title
plt.title("Current, Filtered, and Unfiltered Voltage vs Time with Flux Density")
plt.show()

# Save the processed data
filtered_data.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")
