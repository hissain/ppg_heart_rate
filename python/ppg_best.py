import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
import pywt
import os
from sklearn.decomposition import FastICA


OUTPUT_DIR = "output_best"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a bandpass Butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def filter_signal(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.filtfilt(b, a, data)

def preprocess_ppg(ppg, fs):
    """Preprocess the PPG signal with denoising and normalization."""
    # Remove DC component
    ppg = ppg - np.mean(ppg)
    
    # Apply wavelet denoising
    coeffs = pywt.wavedec(ppg, 'db4', level=8)
    # Threshold the coefficients
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i])*0.3, mode='soft')
    # Reconstruct the signal
    denoised_ppg = pywt.waverec(coeffs, 'db4')
    denoised_ppg = denoised_ppg[:len(ppg)]  # Match original length
    
    # Normalize
    denoised_ppg = (denoised_ppg - np.min(denoised_ppg)) / (np.max(denoised_ppg) - np.min(denoised_ppg))
    
    # Apply a bandpass filter (0.5-8 Hz for heart rate components)
    filtered_ppg = filter_signal(denoised_ppg, 0.5, 8, fs)
    
    return filtered_ppg

def compute_heart_rate(ppg, fs, time_window=10, overlap=0.5):
    """Compute heart rate from PPG signal with sliding window."""
    window_size = int(time_window * fs)
    step_size = int(window_size * (1 - overlap))
    n_windows = (len(ppg) - window_size) // step_size + 1
    
    hr_estimates = []
    time_points = []
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        segment = ppg[start_idx:end_idx]
        
        # Band-pass filter for heart rate (0.5-8 Hz / 30-240 bpm)
        filtered_segment = filter_signal(segment, 0.5, 8, fs)
        
        # Calculate FFT
        n = len(filtered_segment)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_mag = np.abs(np.fft.rfft(filtered_segment))
        
        # Find the peak in the HR frequency band (0.5-4 Hz / 30-240 bpm)
        hr_band = (freqs >= 0.5) & (freqs <= 4.0)
        if np.any(hr_band):
            peak_freq = freqs[hr_band][np.argmax(fft_mag[hr_band])]
            hr = peak_freq * 60  # Convert Hz to bpm
            hr_estimates.append(hr)
            time_points.append((start_idx + window_size/2) / fs)  # Middle of the window
    
    return np.array(time_points), np.array(hr_estimates)

def detect_peaks_advanced(ppg, fs):
    """Advanced peak detection with adaptive thresholding."""
    # Apply bandpass filter for heart rate components
    filtered_ppg = filter_signal(ppg, 0.5, 8, fs)
    
    # Find the average peak distance using autocorrelation
    corr = np.correlate(filtered_ppg, filtered_ppg, mode='full')
    corr = corr[len(corr)//2:]
    
    # Find peaks in autocorrelation (excluding the first peak which is self-correlation)
    ac_peaks, _ = signal.find_peaks(corr, height=0.3*np.max(corr), distance=0.5*fs)
    if len(ac_peaks) > 0:
        avg_peak_distance = ac_peaks[0]
    else:
        avg_peak_distance = int(0.7 * fs)  # Default to 70 bpm if no clear peaks
    
    # Use adaptive threshold for peak detection
    min_peak_distance = int(0.7 * avg_peak_distance)
    
    # Detect peaks
    peaks, properties = signal.find_peaks(
        filtered_ppg, 
        distance=min_peak_distance,
        height=0.3,          # Minimum height
        prominence=0.2,      # Minimum prominence
        width=(0.06*fs, 0.22*fs)  # Width constraints (60-220ms)
    )
    
    return peaks, properties

def extract_am(ppg, peaks):
    """Extract amplitude modulation."""
    return ppg[peaks]

def extract_bw(ppg, fs):
    """Extract baseline wander."""
    b, a = butter_bandpass(0.1, 0.5, fs, order=3)
    return signal.filtfilt(b, a, ppg)

def extract_fm(ppg, fs, peaks=None):
    """Extract frequency modulation (HRV)."""
    if peaks is None:
        peaks, _ = detect_peaks_advanced(ppg, fs)
    
    if len(peaks) < 3:
        return np.array([]), np.array([])
    
    # Calculate inter-beat intervals
    ibi = np.diff(peaks) / fs  # in seconds
    time_ibi = peaks[1:] / fs
    
    # Remove outliers (physiologically impossible IBIs)
    valid_ibi = (ibi >= 0.33) & (ibi <= 2.0)  # 30-180 bpm
    ibi = ibi[valid_ibi]
    time_ibi = time_ibi[valid_ibi]
    
    if len(ibi) < 3:
        return np.array([]), np.array([])
    
    # Uniform resampling of IBI series at 4 Hz
    fs_hrv = 4
    t_uniform = np.arange(time_ibi[0], time_ibi[-1], 1/fs_hrv)
    
    # Cubic spline interpolation for better results
    ibi_interp_func = interp1d(time_ibi, ibi, kind='cubic', bounds_error=False, fill_value="extrapolate")
    ibi_interp = ibi_interp_func(t_uniform)
    
    # Bandpass filter in respiration frequency range
    b, a = butter_bandpass(0.1, 0.5, fs_hrv, order=3)
    hrv_filtered = signal.filtfilt(b, a, ibi_interp)
    
    return hrv_filtered, t_uniform

def extract_features_with_ica(ppg, fs):
    """Extract respiratory features using Independent Component Analysis."""
    # Extract different features
    peaks, _ = detect_peaks_advanced(ppg, fs)
    
    # Amplitude modulation
    am_values = extract_am(ppg, peaks)
    t_am = peaks / fs
    
    # Baseline wander
    bw_signal = extract_bw(ppg, fs)
    
    # Frequency modulation
    fm_signal, t_fm = extract_fm(ppg, fs, peaks)
    
    # If FM extraction failed, return only AM and BW
    if len(fm_signal) == 0:
        features = np.vstack((
            np.interp(np.linspace(0, 1, 1000), np.linspace(0, 1, len(bw_signal)), bw_signal),
            np.interp(np.linspace(0, 1, 1000), np.linspace(0, 1, len(t_am)), am_values)
        )).T
        # Normalize features
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        return features, None, None
    
    # Normalize and resample all signals to the same length
    norm_length = 1000
    features = np.vstack((
        np.interp(np.linspace(0, 1, norm_length), np.linspace(0, 1, len(bw_signal)), bw_signal),
        np.interp(np.linspace(0, 1, norm_length), np.linspace(0, 1, len(t_am)), am_values),
        np.interp(np.linspace(0, 1, norm_length), np.linspace(0, 1, len(t_fm)), fm_signal)
    )).T
    
    # Normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    # Apply ICA to extract independent respiratory components
    if features.shape[0] > features.shape[1]:  # ICA requires samples >= components
        ica = FastICA(n_components=min(3, features.shape[1]), random_state=42)
        ica_components = ica.fit_transform(features)
        return features, ica_components, ica.mixing_
    else:
        return features, None, None

def estimate_rr(signal_mod, fs, min_freq=0.1, max_freq=0.5):
    """Estimate respiration rate from a modulation signal."""
    n = len(signal_mod)
    if n < 2:
        return 0, np.array([]), np.array([])
    
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = signal_mod * window
    
    # Compute FFT with zero-padding for better frequency resolution
    n_fft = max(n, 2048)  # Use at least 2048 points for FFT
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(windowed_signal, n=n_fft))
    
    # Find the peak in the breathing frequency band
    rr_band = (freqs >= min_freq) & (freqs <= max_freq)
    if np.any(rr_band) and np.any(fft_mag[rr_band]):
        peak_freq = freqs[rr_band][np.argmax(fft_mag[rr_band])]
        return peak_freq * 60, freqs, fft_mag  # bpm, freq axis, spectrum
    else:
        return 0, freqs, fft_mag

def estimate_rr_combined(ppg, fs, confidence_threshold=0.4):
    """Estimate respiration rate with multiple methods and confidence measures."""
    # Preprocess the signal
    ppg_preprocessed = preprocess_ppg(ppg, fs)
    
    # Detect peaks
    peaks, peak_props = detect_peaks_advanced(ppg_preprocessed, fs)
    
    # Extract features (potentially with ICA)
    features, ica_components, mixing_matrix = extract_features_with_ica(ppg_preprocessed, fs)
    
    # AM: Amplitude modulation
    am_values = extract_am(ppg_preprocessed, peaks)
    t_am = peaks / fs
    am_interp = np.interp(np.arange(len(ppg_preprocessed)) / fs, t_am, am_values) if len(am_values) > 1 else np.zeros_like(ppg_preprocessed)
    rr_am, freqs_am, fft_am = estimate_rr(am_interp, fs)
    
    # AM confidence based on spectral peak prominence
    if rr_am > 0:
        am_peak_idx = np.argmax(fft_am[(freqs_am >= 0.1) & (freqs_am <= 0.5)])
        am_peak_value = np.max(fft_am[(freqs_am >= 0.1) & (freqs_am <= 0.5)])
        am_mean_power = np.mean(fft_am[(freqs_am >= 0.1) & (freqs_am <= 0.5)])
        am_confidence = (am_peak_value - am_mean_power) / (am_peak_value + am_mean_power)
    else:
        am_confidence = 0
    
    # BW: Baseline wander
    bw_signal = extract_bw(ppg_preprocessed, fs)
    rr_bw, freqs_bw, fft_bw = estimate_rr(bw_signal, fs)
    
    # BW confidence based on spectral peak prominence
    if rr_bw > 0:
        bw_peak_idx = np.argmax(fft_bw[(freqs_bw >= 0.1) & (freqs_bw <= 0.5)])
        bw_peak_value = np.max(fft_bw[(freqs_bw >= 0.1) & (freqs_bw <= 0.5)])
        bw_mean_power = np.mean(fft_bw[(freqs_bw >= 0.1) & (freqs_bw <= 0.5)])
        bw_confidence = (bw_peak_value - bw_mean_power) / (bw_peak_value + bw_mean_power)
    else:
        bw_confidence = 0
    
    # FM: Frequency modulation (HRV)
    fm_signal, t_fm = extract_fm(ppg_preprocessed, fs, peaks)
    
    if len(fm_signal) > 0:
        rr_fm, freqs_fm, fft_fm = estimate_rr(fm_signal, 4)
        # FM confidence based on spectral peak prominence
        if rr_fm > 0:
            fm_peak_idx = np.argmax(fft_fm[(freqs_fm >= 0.1) & (freqs_fm <= 0.5)])
            fm_peak_value = np.max(fft_fm[(freqs_fm >= 0.1) & (freqs_fm <= 0.5)])
            fm_mean_power = np.mean(fft_fm[(freqs_fm >= 0.1) & (freqs_fm <= 0.5)])
            fm_confidence = (fm_peak_value - fm_mean_power) / (fm_peak_value + fm_mean_power)
        else:
            fm_confidence = 0
    else:
        rr_fm = 0
        fm_confidence = 0
        freqs_fm = np.array([])
        fft_fm = np.array([])
    
    # ICA-based estimates (if available)
    if ica_components is not None:
        rr_ica_values = []
        ica_confidences = []
        
        for i in range(ica_components.shape[1]):
            rr_ica, freqs_ica, fft_ica = estimate_rr(ica_components[:, i], fs / (len(ppg_preprocessed) / len(ica_components)))
            
            if rr_ica > 0:
                ica_peak_value = np.max(fft_ica[(freqs_ica >= 0.1) & (freqs_ica <= 0.5)])
                ica_mean_power = np.mean(fft_ica[(freqs_ica >= 0.1) & (freqs_ica <= 0.5)])
                ica_confidence = (ica_peak_value - ica_mean_power) / (ica_peak_value + ica_mean_power)
                
                rr_ica_values.append(rr_ica)
                ica_confidences.append(ica_confidence)
        
        if len(rr_ica_values) > 0:
            # Select the ICA component with highest confidence
            best_ica_idx = np.argmax(ica_confidences)
            rr_ica = rr_ica_values[best_ica_idx]
            ica_confidence = ica_confidences[best_ica_idx]
        else:
            rr_ica = 0
            ica_confidence = 0
    else:
        rr_ica = 0
        ica_confidence = 0
    
    # Compute heart rate
    time_hr, hr_estimates = compute_heart_rate(ppg_preprocessed, fs)
    avg_hr = np.median(hr_estimates) if len(hr_estimates) > 0 else 0
    
    # Store the RR estimates and their confidence
    rr_methods = []
    confidences = []
    
    if rr_am > 4 and rr_am < 60 and am_confidence > confidence_threshold:
        rr_methods.append(rr_am)
        confidences.append(am_confidence)
    
    if rr_bw > 4 and rr_bw < 60 and bw_confidence > confidence_threshold:
        rr_methods.append(rr_bw)
        confidences.append(bw_confidence)
    
    if rr_fm > 4 and rr_fm < 60 and fm_confidence > confidence_threshold:
        rr_methods.append(rr_fm)
        confidences.append(fm_confidence)
    
    if rr_ica > 4 and rr_ica < 60 and ica_confidence > confidence_threshold:
        rr_methods.append(rr_ica)
        confidences.append(ica_confidence)
    
    # Combine the estimates using confidence-weighted average
    if len(rr_methods) > 0:
        rr_combined = np.average(rr_methods, weights=confidences)
        rr_combined_std = np.std(rr_methods)
    else:
        # Fallback if no method has sufficient confidence
        rr_methods = [r for r in [rr_am, rr_bw, rr_fm, rr_ica] if r > 4 and r < 60]
        rr_combined = np.median(rr_methods) if len(rr_methods) > 0 else 0
        rr_combined_std = np.std(rr_methods) if len(rr_methods) > 0 else 0
    
    # Visualize the results
    visualize_results(
        ppg_preprocessed, fs, peaks, 
        bw_signal, rr_bw, freqs_bw, fft_bw, bw_confidence,
        am_interp, rr_am, freqs_am, fft_am, am_confidence,
        fm_signal, t_fm, rr_fm, freqs_fm, fft_fm, fm_confidence,
        rr_ica, ica_confidence, 
        rr_combined, rr_combined_std,
        time_hr, hr_estimates, avg_hr
    )
    
    return {
        "RR_BW": rr_bw,
        "RR_AM": rr_am,
        "RR_FM": rr_fm,
        "RR_ICA": rr_ica,
        "RR_Combined": rr_combined,
        "RR_Combined_Std": rr_combined_std,
        "Confidence_BW": bw_confidence,
        "Confidence_AM": am_confidence,
        "Confidence_FM": fm_confidence,
        "Confidence_ICA": ica_confidence,
        "HR_Avg": avg_hr,
        "HR_Std": np.std(hr_estimates) if len(hr_estimates) > 0 else 0
    }

def visualize_results(
    ppg, fs, peaks, 
    bw_signal, rr_bw, freqs_bw, fft_bw, bw_confidence,
    am_interp, rr_am, freqs_am, fft_am, am_confidence,
    fm_signal, t_fm, rr_fm, freqs_fm, fft_fm, fm_confidence,
    rr_ica, ica_confidence, 
    rr_combined, rr_combined_std,
    time_hr, hr_estimates, avg_hr
):
    """Visualize the PPG analysis results."""
    # Create a comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Define grid for subplots
    gs = plt.GridSpec(4, 4, figure=fig)
    
    # Original PPG with detected peaks
    ax1 = fig.add_subplot(gs[0, :2])
    time_axis = np.arange(len(ppg)) / fs
    ax1.plot(time_axis, ppg)
    ax1.plot(peaks / fs, ppg[peaks], 'r*')
    ax1.set_title("Preprocessed PPG with Detected Peaks")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    
    # Heart rate over time
    ax2 = fig.add_subplot(gs[0, 2:])
    if len(time_hr) > 0:
        ax2.plot(time_hr, hr_estimates)
        ax2.axhline(y=avg_hr, color='r', linestyle='--', label=f"Avg: {avg_hr:.1f} bpm")
        ax2.set_title(f"Heart Rate Over Time (Avg: {avg_hr:.1f} bpm)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Heart Rate (bpm)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Insufficient data for HR estimation", 
                 ha='center', va='center', transform=ax2.transAxes)
    
    # AM signal and spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(np.arange(len(am_interp)) / fs, am_interp)
    ax3.set_title(f"AM (Conf: {am_confidence:.2f})")
    ax3.set_xlabel("Time (s)")
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(freqs_am, np.abs(fft_am))
    ax4.axvline(x=rr_am/60, color='r', linestyle='--', label=f"{rr_am:.1f} bpm")
    ax4.set_xlim(0.05, 0.6)
    ax4.set_title("AM Spectrum")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.legend()
    
    # BW signal and spectrum
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(np.arange(len(bw_signal)) / fs, bw_signal)
    ax5.set_title(f"BW (Conf: {bw_confidence:.2f})")
    ax5.set_xlabel("Time (s)")
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(freqs_bw, np.abs(fft_bw))
    ax6.axvline(x=rr_bw/60, color='r', linestyle='--', label=f"{rr_bw:.1f} bpm")
    ax6.set_xlim(0.05, 0.6)
    ax6.set_title("BW Spectrum")
    ax6.set_xlabel("Frequency (Hz)")
    ax6.legend()
    
    # FM signal and spectrum
    ax7 = fig.add_subplot(gs[3, 0])
    if len(fm_signal) > 0 and len(t_fm) > 0:
        ax7.plot(t_fm, fm_signal)
        ax7.set_title(f"FM/HRV (Conf: {fm_confidence:.2f})")
    else:
        ax7.text(0.5, 0.5, "Insufficient data for FM analysis", 
                 ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title("FM/HRV (N/A)")
    ax7.set_xlabel("Time (s)")
    
    ax8 = fig.add_subplot(gs[3, 1])
    if len(freqs_fm) > 0 and len(fft_fm) > 0:
        ax8.plot(freqs_fm, np.abs(fft_fm))
        ax8.axvline(x=rr_fm/60, color='r', linestyle='--', label=f"{rr_fm:.1f} bpm")
        ax8.set_xlim(0.05, 0.6)
        ax8.set_title("FM Spectrum")
        ax8.legend()
    else:
        ax8.text(0.5, 0.5, "Insufficient data for FM spectrum", 
                 ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title("FM Spectrum (N/A)")
    ax8.set_xlabel("Frequency (Hz)")
    
    # ICA Results (if available)
    ax9 = fig.add_subplot(gs[1:3, 2])
    if rr_ica > 0:
        ax9.text(0.5, 0.5, f"ICA-based RR: {rr_ica:.1f} bpm\nConfidence: {ica_confidence:.2f}", 
                 ha='center', va='center', transform=ax9.transAxes, fontsize=12)
    else:
        ax9.text(0.5, 0.5, "ICA analysis not available", 
                 ha='center', va='center', transform=ax9.transAxes, fontsize=12)
    ax9.set_title("ICA Results")
    ax9.axis('off')
    
    # Combined Results
    ax10 = fig.add_subplot(gs[1:3, 3])
    ax10.text(0.5, 0.7, f"ESTIMATED RESPIRATION RATE:\n{rr_combined:.1f} ± {rr_combined_std:.1f} bpm", 
             ha='center', va='center', transform=ax10.transAxes, fontsize=14, fontweight='bold')
    ax10.text(0.5, 0.3, f"ESTIMATED HEART RATE:\n{avg_hr:.1f} bpm", 
             ha='center', va='center', transform=ax10.transAxes, fontsize=14, fontweight='bold')
    ax10.axis('off')
    ax10.set_title("Combined Results", fontsize=14)
    
    # Method comparison
    ax11 = fig.add_subplot(gs[3, 2:])
    methods = []
    values = []
    colors = []
    
    if rr_am > 0:
        methods.append('AM')
        values.append(rr_am)
        colors.append('skyblue')
    if rr_bw > 0:
        methods.append('BW')
        values.append(rr_bw)
        colors.append('lightgreen')
    if rr_fm > 0:
        methods.append('FM')
        values.append(rr_fm)
        colors.append('salmon')
    if rr_ica > 0:
        methods.append('ICA')
        values.append(rr_ica)
        colors.append('purple')
    if rr_combined > 0:
        methods.append('Combined')
        values.append(rr_combined)
        colors.append('gold')
    
    if len(methods) > 0:
        bars = ax11.bar(methods, values, color=colors)
        ax11.axhline(y=rr_combined, color='r', linestyle='--')
        ax11.set_ylabel('Respiration Rate (bpm)')
        ax11.set_title('Method Comparison')
    else:
        ax11.text(0.5, 0.5, "No valid RR estimates", 
                 ha='center', va='center', transform=ax11.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ppg_analysis_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_ppg_signal(file_path, fs=125, segment_length=120):
    """Analyze a PPG signal from a file with windowed processing."""
    # Load the data
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=1)
        print(f"Loaded PPG data with {len(data)} samples")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Parameters for windowed analysis
    window_size = segment_length * fs
    
    # Process the entire signal if it's shorter than the window
    if len(data) <= window_size:
        return estimate_rr_combined(data, fs)
    
    # Process in windows with 50% overlap
    results = []
    step_size = window_size // 2
    windows = (len(data) - window_size) // step_size + 1
    
    for i in range(windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        segment = data[start_idx:end_idx]
        
        print(f"Processing window {i+1}/{windows} (samples {start_idx}-{end_idx})")
        window_results = estimate_rr_combined(segment, fs)
        window_results["Window"] = i+1
        window_results["Start_Sample"] = start_idx
        window_results["End_Sample"] = end_idx
        
        results.append(window_results)
    
    # Combine results from all windows (using weighted average based on confidence)
    combined_rr = 0
    total_confidence = 0
    
    valid_results = [r for r in results if r["RR_Combined"] > 0]
    confidences = []
    for r in valid_results:
        # Calculate an overall confidence for this window
        window_conf = max(r["Confidence_AM"], r["Confidence_BW"], 
                         r["Confidence_FM"], r["Confidence_ICA"])
        confidences.append(window_conf)
        combined_rr += r["RR_Combined"] * window_conf
        total_confidence += window_conf
    
    final_results = {}
    if total_confidence > 0:
        final_results["RR_Combined"] = combined_rr / total_confidence
    else:
        # If no confidence, use simple median
        rr_values = [r["RR_Combined"] for r in valid_results if r["RR_Combined"] > 0]
        final_results["RR_Combined"] = np.median(rr_values) if len(rr_values) > 0 else 0
    
    # Calculate HR in the same way
    combined_hr = 0
    total_hr_windows = 0
    
    for r in results:
        if r["HR_Avg"] > 0:
            combined_hr += r["HR_Avg"]
            total_hr_windows += 1
    
    if total_hr_windows > 0:
        final_results["HR_Avg"] = combined_hr / total_hr_windows
    else:
        final_results["HR_Avg"] = 0
    
    # Add window-based results
    final_results["Window_Results"] = results
    
    return final_results

def visualize_time_series(data, fs, window_size=30, overlap=0.5):
    """Visualize time-series evolution of heart rate and respiration rate."""
    # Process data in windows
    window_samples = int(window_size * fs)
    step_size = int(window_samples * (1 - overlap))
    windows = (len(data) - window_samples) // step_size + 1
    
    time_points = []
    hr_values = []
    rr_values = []
    
    for i in range(windows):
        start_idx = i * step_size
        end_idx = start_idx + window_samples
        segment = data[start_idx:end_idx]
        
        # Get heart rate
        time_hr, hr_est = compute_heart_rate(segment, fs)
        if len(hr_est) > 0:
            hr_values.append(np.median(hr_est))
        else:
            hr_values.append(np.nan)
        
        # Get respiration rate
        rr_results = estimate_rr_combined(segment, fs)
        rr_values.append(rr_results["RR_Combined"])
        
        # Store time point (middle of the window)
        time_points.append((start_idx + window_samples/2) / fs)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot heart rate over time
    ax1.plot(time_points, hr_values, 'o-', color='red')
    ax1.set_title("Heart Rate Over Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Heart Rate (bpm)")
    ax1.grid(True)
    
    # Plot respiration rate over time
    ax2.plot(time_points, rr_values, 'o-', color='blue')
    ax2.set_title("Respiration Rate Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Respiration Rate (bpm)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/physiological_parameters_over_time.png", dpi=300, bbox_inches='tight')
    plt.show()

def wavelet_analysis(signal, fs):
    """Perform wavelet analysis on the signal to detect multi-scale features."""
    # Wavelet decomposition
    scales = np.arange(1, 128)
    wavelet = 'morl'  # Morlet wavelet
    
    # Calculate the CWT
    coef, freqs = pywt.cwt(signal, scales, wavelet, 1/fs)
    
    # Convert frequencies to bpm
    freqs_bpm = freqs * 60
    
    # Plot the wavelet scalogram
    plt.figure(figsize=(12, 8))
    
    # Plot the scalogram
    plt.subplot(211)
    plt.pcolormesh(np.arange(len(signal))/fs, freqs_bpm, np.abs(coef), shading='gouraud')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Frequency (bpm)')
    plt.xlabel('Time (s)')
    plt.title('Wavelet Scalogram')
    
    # Highlight the respiration (0.1-0.5 Hz / 6-30 bpm) and heart rate (0.5-3 Hz / 30-180 bpm) bands
    plt.axhline(y=6, color='r', linestyle='--', label='Respiration band (lower)')
    plt.axhline(y=30, color='r', linestyle='--', label='Respiration band (upper)')
    plt.axhline(y=30, color='g', linestyle='--', label='Heart rate band (lower)')
    plt.axhline(y=180, color='g', linestyle='--', label='Heart rate band (upper)')
    plt.legend(loc='upper right')
    
    # Plot the original signal for reference
    plt.subplot(212)
    plt.plot(np.arange(len(signal))/fs, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original PPG Signal')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/wavelet_analysis.png", dpi=300)
    plt.show()
    
    return coef, freqs_bpm

def phase_space_reconstruction(signal, delay=10, dimension=3):
    """Perform phase space reconstruction of the signal."""
    # Normalize the signal
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    # Construct the phase space vectors
    num_points = len(signal) - (dimension - 1) * delay
    phase_space = np.zeros((num_points, dimension))
    
    for i in range(dimension):
        phase_space[:, i] = signal[i*delay:i*delay + num_points]
    
    # Create a 3D phase space plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the phase space trajectory
    if dimension >= 3:
        ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], 'b-', linewidth=0.5)
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t+τ)')
        ax.set_zlabel('x(t+2τ)')
    else:
        ax.plot(phase_space[:, 0], phase_space[:, 1], np.zeros_like(phase_space[:, 0]), 'b-', linewidth=0.5)
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t+τ)')
        ax.set_zlabel('')
    
    ax.set_title(f'Phase Space Reconstruction (τ={delay}, dim={dimension})')
    plt.savefig(f"{OUTPUT_DIR}/phase_space_reconstruction.png", dpi=300)
    plt.show()
    
    return phase_space

def compute_signal_quality_index(ppg, fs):
    """Compute signal quality index for PPG signal."""
    # Extract segments of 5 seconds
    segment_length = 5 * fs
    num_segments = len(ppg) // segment_length
    
    # Quality metrics for each segment
    sqi_values = []
    
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = ppg[start_idx:end_idx]
        
        # 1. Signal-to-noise ratio
        # - Use wavelet denoising to estimate noise
        coeffs = pywt.wavedec(segment, 'db4', level=5)
        # - Reconstruct with just the first level (signal)
        coeffs_signal = coeffs.copy()
        for j in range(1, len(coeffs)):
            coeffs_signal[j] = np.zeros_like(coeffs[j])
        signal_approx = pywt.waverec(coeffs_signal, 'db4')[:len(segment)]
        
        # - Estimate noise as difference between original and approximation
        noise = segment - signal_approx
        snr = 10 * np.log10(np.sum(signal_approx**2) / np.sum(noise**2)) if np.sum(noise**2) > 0 else 0
        
        # 2. Skewness (PPG should be asymmetric)
        skewness = stats.skew(segment)
        
        # 3. Kurtosis (measure of peakedness)
        kurtosis = stats.kurtosis(segment)
        
        # 4. Spectral purity (power in heart rate band vs. total power)
        freqs, psd = signal.welch(segment, fs=fs, nperseg=fs)
        total_power = np.sum(psd)
        hr_band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 3.0)])  # 30-180 bpm
        spectral_purity = hr_band_power / total_power if total_power > 0 else 0
        
        # Combine metrics into a single SQI (normalized between 0 and 1)
        # - Normalize SNR (higher is better)
        snr_norm = np.clip(snr / 20, 0, 1)  # Assume max SNR of 20dB
        
        # - Normalize skewness (PPG should be negatively skewed)
        skew_norm = np.clip((skewness + 2) / 4, 0, 1)  # Range: -2 to 2
        
        # - Normalize kurtosis (higher is better, but not too high)
        kurt_norm = np.clip((kurtosis + 3) / 6, 0, 1)  # Range: -3 to 3
        
        # Combine (with weights)
        sqi = 0.4 * snr_norm + 0.2 * skew_norm + 0.1 * kurt_norm + 0.3 * spectral_purity
        
        sqi_values.append(sqi)
    
    # Plot SQI over time
    plt.figure(figsize=(12, 5))
    segment_times = np.arange(num_segments) * 5  # Every 5 seconds
    plt.plot(segment_times, sqi_values, 'o-')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Quality Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Quality Index (SQI)')
    plt.title('PPG Signal Quality Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/signal_quality_index.png", dpi=300)
    plt.show()
    
    return np.mean(sqi_values), sqi_values

# Example usage
if __name__ == "__main__":
    # Set the file path
    file_path = 'dataset/MAUS/MAUS/Data/Raw_data/002/inf_ppg.csv'

    # Load the data (adjust skiprows and usecols as needed)
    try:
        ppg = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=0)
        fs = 256  # Sampling frequency in Hz
        
        print(f"PPG signal length: {len(ppg)}")
        print(f"First 10 PPG values: {ppg[:10]}")
        
        # Limit data for testing if needed
        segment_length = 60  # seconds
        data_subset = ppg[:int(segment_length * fs)]
        
        # If you want to process the entire recording:
        # results = analyze_ppg_signal(file_path, fs)
        
        # For testing with a subset:
        results = estimate_rr_combined(data_subset, fs)
        print("\nRespiration Rate Results:")
        for key, value in results.items():
            if not key.startswith("Window"):
                print(f"{key}: {value}")
        
        # Additional analyses
        print("\nPerforming additional analyses...")
        
        # Preprocess signal
        ppg_preprocessed = preprocess_ppg(data_subset, fs)
        
        # Compute signal quality
        print("\nComputing signal quality index...")
        avg_sqi, sqi_values = compute_signal_quality_index(ppg_preprocessed, fs)
        print(f"Average Signal Quality Index: {avg_sqi:.3f}")
        
        # Wavelet analysis
        print("\nPerforming wavelet analysis...")
        coef, freqs_bpm = wavelet_analysis(ppg_preprocessed, fs)
        
        # Phase space reconstruction
        print("\nPerforming phase space reconstruction...")
        phase_space = phase_space_reconstruction(ppg_preprocessed, delay=int(fs * 0.2), dimension=3)
        
        # Time series visualization if enough data
        if len(ppg) > fs * 60:  # If more than 60 seconds of data
            print("\nVisualizing physiological parameters over time...")
            visualize_time_series(ppg[:int(fs * 120)], fs)  # Analyze first 2 minutes
            
        print("\nAnalysis complete. Results saved to 'output' folder.")
        
    except Exception as e:
        print(f"Error: {e}")