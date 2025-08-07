### Wang Method 1: Finding Speed of Sound and Attenuation
# Full implementation of M1, requires finding sample thickness

#%% Signal Preparation and Visual Inspection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

def average_raw_signals(filenames):
    """
    Averages the raw voltage signals across multiple runs.
    
    Parameters:
    filenames (list): List of CSV filenames to process.
    
    Returns:
    tuple: Average time and voltage arrays.
    """
    all_voltages = []
    
    for filename in filenames:
        data = pd.read_csv(filename, header=1, skiprows=[2])
        time = data['second'].values
        voltage = data['Volt'].values
        
        all_voltages.append(voltage)
    
    avg_voltage = np.mean(all_voltages, axis=0)
    
    return time, avg_voltage
def extract_window(time, voltage, center_time, window_width_us, alpha=0):
    window_width = window_width_us * 1e-6
    dt = time[1] - time[0]
    n_points = int(window_width / dt)
    if n_points % 2 == 0:
        n_points += 1

    center_idx = np.argmin(np.abs(time - center_time))
    half_n = n_points // 2
    start_idx = center_idx - half_n
    end_idx = center_idx + half_n + 1

    if start_idx < 0 or end_idx > len(time):
        raise ValueError("Window exceeds bounds")

    window = tukey(n_points, alpha)
    return time[start_idx:end_idx], voltage[start_idx:end_idx] * window

# Sample Signals
filenames = ["scope_77.csv", "scope_78.csv", "scope_79.csv"]  # Add your filenames here
time, voltage = average_raw_signals(filenames)

# Water Signals
filenames = ["scope_74.csv", "scope_75.csv", "scope_76.csv"]  # Add your filenames here
time_w, voltage_w = average_raw_signals(filenames)

# Plot S1 Waveform
plt.figure(figsize=(10, 4))
plt.plot(time * 1e6, voltage, label='Original S1 Signal')
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (V)")
plt.title("S1 Pulse Waveform")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Water Waveform
plt.figure(figsize=(10, 4))
plt.plot(time_w * 1e6, voltage_w, label='Original Water Signal')  
plt.title("Water Pulse Waveform")
plt.xlabel("Time (µs)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Run FFTs

def phase_amp_calc(time, voltage, center_freq):
    """
    Returns amplitude and phase of a waveform from FFT of signal
    
    Parameters: 
    time(np.ndarray): Time values of the signal (seconds)
    voltage(np.ndarray): Voltage values of the signal (Volts)
    center_freq(float): Center frequency of the signal (Hz)

    Returns:
    amplitude(float): Amplitude of the signal (Volts)
    phase(float): Phase of the signal (degrees)
    """
    n = len(voltage)
    dt = time[1] - time[0]

    # FFT
    voltage_fft = np.fft.fft(voltage)
    freqs = np.fft.fftfreq(n, dt)

    # Use only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = voltage_fft[pos_mask]

    # Find index of frequency closest to target
    idx = np.argmin(np.abs(freqs - center_freq))
    amplitude = 2 * np.abs(fft_vals[idx]) / n  # scale for single-sided spectrum
    phase = np.angle(fft_vals[idx], deg=True)

    return amplitude, phase

#Center Frequency Input 
f = 4.6e6  # Center frequency in Hz

amp_1, phi_1 = phase_amp_calc(time, voltage, f)
amp_w, phi_w = phase_amp_calc(time_w, voltage_w, f)

print(f"Water Amplitude: {amp_w:.4f} V, Phase: {phi_w:.2f} degrees")
print(f"S1 Amplitude: {amp_1:.4f} V, Phase: {phi_1:.2f} degrees")

# %% Wang Method I Calculations 

# Input variables
d = 0.00150    # Thickness of sample (m)
v_w = 1486.0    # Speed of sound in water (m/s)
rho_w = 997     # Density of water (kg/m^3)
rho_s =  1100   # Density of specimen (kg/m^3)

# Phase Velocity
phi_diff = phi_w - phi_1
v_f = v_w / (1 + ((phi_diff * v_w) / (2 * np.pi * f * d)))

# Impedance Values 
z_w = rho_w * v_w  # Impedance of water
z_s = rho_s * v_f  # Impedance of sample

# Convert to MRayl
z_w /= 1e6  
z_s /= 1e6  

# Transmission coefficient T(f) (@ normal incidence ie theta = 0)
t_f = (4 * z_w * z_s) / ((z_w + z_s) ** 2)

# Attenuation 
alpha_f = (1 / d) * np.log((t_f * amp_w) / amp_1)

# Convert to dB/cm
alpha_dB_cm = alpha_f * 8.686 / 100

# Print Results
print(f"Impedance Water: {z_w:.2f} MRayl")
print(f"Impedance Sample: {z_s:.2f} MRayl")
print(f"Phase Velocity: {v_f:.2f} m/s")
print(f"Transmission: {t_f:.4f}")
print(f"Attenuation: {alpha_dB_cm:.4f} dB/cm")