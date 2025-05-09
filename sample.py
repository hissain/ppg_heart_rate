import numpy as np
import matplotlib.pyplot as plt
import json
import os
from ppg import PPGRespirationAnalyzer

def generate_sample_data(duration_seconds=60, sampling_rate=30):
    """Generate synthetic PPG data for demonstration"""
    # Time vector
    t = np.arange(0, duration_seconds, 1/sampling_rate)
    
    # Heart rate component (around 1.2 Hz, or 72 bpm)
    heart_rate = 1.2
    hr_signal = np.sin(2 * np.pi * heart_rate * t)
    
    # Respiratory component (around 0.25 Hz, or 15 breaths per minute)
    resp_rate = 0.25
    resp_signal = 0.3 * np.sin(2 * np.pi * resp_rate * t)
    
    # Baseline drift
    baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t)
    
    # Combine components
    ppg_signal = hr_signal + resp_signal + baseline
    
    # Add some noise
    noise = np.random.normal(0, 0.05, len(t))
    ppg_signal += noise
    
    # Adjust for physiological modulation (respiration affects heart rate)
    modulation = np.ones_like(t) + 0.2 * resp_signal
    ppg_signal *= modulation
    
    # Scale to realistic PPG values
    ppg_signal = ppg_signal * 50 + 500
    
    # Generate timestamps (in milliseconds)
    timestamps = np.array([i * (1000 / sampling_rate) for i in range(len(t))])
    
    return ppg_signal, timestamps

def save_sample_data(ppg_signal, timestamps, file_path="sample_ppg_data.json"):
    """Save sample data to file"""
    data = {
        "ppg_data": ppg_signal.tolist(),
        "timestamps": timestamps.tolist()
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Sample data saved to {file_path}")

def main():
    # Generate sample data
    print("Generating sample PPG data...")
    sampling_rate = 30  # Hz
    ppg_signal, timestamps = generate_sample_data(duration_seconds=60, sampling_rate=sampling_rate)
    
    # Save sample data
    save_sample_data(ppg_signal, timestamps)
    
    # Initialize analyzer
    analyzer = PPGRespirationAnalyzer(sampling_rate=sampling_rate)
    
    # Analyze the generated data
    print("Analyzing PPG data...")
    results = analyzer.analyze_ppg(ppg_signal, timestamps)
    
    # Print results
    resp_rate = results['respiration_rate']
    hr = results['heart_rate']
    
    print(f"===== PPG Analysis Results =====")
    print(f"Data length: {len(ppg_signal)} samples ({len(ppg_signal)/sampling_rate:.1f} seconds)")
    print(f"Respiration Rate: {resp_rate:.1f} breaths/min" if resp_rate else "Respiration Rate: Not detected")
    print(f"Heart Rate: {hr:.1f} bpm" if hr else "Heart Rate: Not detected")
    
    # Plot results
    fig = analyzer.plot_results(results)
    plt.show()

    fig.savefig("figures/ppg_analysis_results.png")
    
    return 0

if __name__ == "__main__":
    main()