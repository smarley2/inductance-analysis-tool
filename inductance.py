import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# User-configurable variables
file_path = "example/Tek001_ALL.csv"  # Path to the CSV file
skip_rows = 16  # Number of rows to skip in the CSV
time_min = 0.5e-6  # Minimum time for filtering (in seconds)
time_max = 120e-6  # Maximum time for filtering (in seconds)
cutoff_frequency = 5e5  # Cutoff frequency for low-pass filter (in Hz)
low_pass_filter_order = 4  # Order of the Butterworth low-pass filter
inductance_min = 0  # Minimum inductance to be plotted (in H)
inductance_max = 400e-6  # Maximum inductance to be plotted (in H)
interval_ranges = {  # Interval ranges for gradient calculation based on current ranges
#    (0, 10): 2000,      # Interval for current range 0 to 10 A
#    (10, 30): 2000,     # Interval for current range 10 to 30 A
    (0, np.inf): 2000 # Interval for current range 30 A to max
}
output_file = "inductance_vs_current.csv"  # Output file to save processed data

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
data.rename(columns={'TIME': 'time', 'CH8': 'current', 'CH3': 'voltage'}, inplace=True)

# Filter time values
filtered_data = data[(data['time'] > time_min) & (data['time'] < time_max)]

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

# Calculate inductance using the filtered voltage
inductance = filtered_voltage / dI_dt

# Add inductance vs current to the DataFrame
filtered_data['inductance'] = inductance

# Filter data for the selected inductance range
filtered_plot_data = filtered_data[
    (filtered_data['inductance'] >= inductance_min) &
    (filtered_data['inductance'] <= inductance_max)
]

# Plot inductance vs current
plt.figure()
plt.plot(filtered_plot_data['current'], filtered_plot_data['inductance'], label="Inductance vs Current")
plt.xlabel("Current (A)")
plt.ylabel("Inductance (H)")
plt.title("Inductance vs Current")
plt.grid()
plt.legend()
plt.show()

# Plot current and both filtered and unfiltered voltage vs time
plt.figure()
plt.plot(time, current, label="Current (A)", alpha=0.7)
plt.plot(time, voltage, label="Unfiltered Voltage (V)", alpha=0.7)
plt.plot(time, filtered_voltage, label="Filtered Voltage (V)", alpha=0.7, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Current, Filtered, and Unfiltered Voltage vs Time")
plt.grid()
plt.legend()
plt.show()

# Save the processed data
filtered_data.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")
