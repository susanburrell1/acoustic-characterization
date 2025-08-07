#%% FFT to find waveform amplitude and phase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data from oscilloscope
filename = "scope_57.csv"  # change to correct file name 
data = pd.read_csv(filename, header=1, skiprows=[2])

# Extract time and voltage from scope file 
time = data['second'].values
voltage = data['Volt'].values
# Plot input waveform
plt.figure(figsize=(10, 4))
plt.plot(time * 1e6, voltage)  # time in microseconds, voltage in millivolts
plt.title("Pulse Waveform")
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Sampling parameters
dt = time[1] - time[0]
fs = 1 / dt
n = len(voltage)

# Perform FFT
voltage_fft = np.fft.fft(voltage)
freqs = np.fft.fftfreq(n, dt)

# Keep only positive frequencies
positive_freqs = freqs[:n // 2]
fft_magnitude = np.abs(voltage_fft[:n // 2]) * 2 / n
fft_phase = np.angle(voltage_fft[:n // 2], deg=True)

# Find dominant frequency
peak_idx = np.argmax(fft_magnitude)
dominant_freq = positive_freqs[peak_idx]
freq_mhz = dominant_freq * 1e-6  
amplitude = fft_magnitude[peak_idx]
phase = fft_phase[peak_idx]

print(f"Center Frequency: {freq_mhz:.2f} MHz")
print(f"Amplitude: {amplitude:.4f} V")
print(f"Phase: {phase:.2f} degrees")
