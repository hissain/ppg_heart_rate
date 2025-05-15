#Copyright (c) 2025, Md. Sazzad Hissain Khan
#All rights reserved.
# This source code is licensed under the Apache 2.0 License

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def simulate_ppg(duration_sec=30, fs=100, heart_rate_bpm=75, resp_rate_bpm=15, noise_std=0.01, motion=False):
    t = np.linspace(0, duration_sec, duration_sec * fs)
    heart_rate_hz = heart_rate_bpm / 60
    resp_rate_hz = resp_rate_bpm / 60

    # 1. Simulate basic pulse (simplified triangular PPG pulse shape)
    pulse_template = np.concatenate([np.linspace(0, 1, int(fs*0.05)), np.linspace(1, 0, int(fs*0.2))])
    pulse_length = len(pulse_template)
    
    ibi = fs / heart_rate_hz
    pulse_indices = np.arange(0, len(t), int(ibi))
    ppg = np.zeros_like(t)
    for i in pulse_indices:
        if i + pulse_length < len(ppg):
            ppg[i:i+pulse_length] += pulse_template

    # 2. Apply amplitude modulation by respiration
    am = 1 + 0.1 * np.sin(2 * np.pi * resp_rate_hz * t)
    ppg *= am

    # 3. Add baseline wander (respiratory-induced)
    baseline = 0.1 * np.sin(2 * np.pi * resp_rate_hz * t)
    ppg += baseline

    # 4. Add random Gaussian noise
    ppg += np.random.normal(0, noise_std, size=len(ppg))

    # 5. Optional: Add motion artifact (e.g., transient spike)
    if motion:
        for i in range(2, duration_sec, 10):
            idx = int(i * fs)
            if idx + 50 < len(ppg):
                ppg[idx:idx+50] += 0.5 * np.sin(2 * np.pi * 1.5 * np.linspace(0, 0.5, 50))

    return t, ppg

# Example usage and plot
fs = 100  # Sampling frequency in Hz
t, ppg = simulate_ppg(duration_sec=30, fs=fs, heart_rate_bpm=72, resp_rate_bpm=14, noise_std=0.02, motion=True)

plt.figure(figsize=(12, 4))
plt.plot(t, ppg, label="Simulated PPG data", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Synthetoc PPG Signal")
plt.title("Simulated Raw PPG Signal (with respiration, noise, and motion artifact)")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/simulated_ppg.png")
plt.show()