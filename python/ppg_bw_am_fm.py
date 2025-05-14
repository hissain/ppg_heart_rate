import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def extract_am(ppg, peaks):
    return ppg[peaks]

def extract_bw(ppg, fs):
    b, a = butter_bandpass(0.1, 0.5, fs)
    return signal.filtfilt(b, a, ppg)

def extract_fm(ppg, fs):
    peaks, _ = signal.find_peaks(ppg, distance=fs * 0.4)
    ibi = np.diff(peaks) / fs
    time_ibi = peaks[1:] / fs
    fs_hrv = 4
    t_uniform = np.arange(time_ibi[0], time_ibi[-1], 1/fs_hrv)
    ibi_interp = np.interp(t_uniform, time_ibi, ibi)
    b, a = butter_bandpass(0.1, 0.5, fs_hrv)
    hrv_filtered = signal.filtfilt(b, a, ibi_interp)
    return hrv_filtered, t_uniform

def estimate_rr(signal_mod, fs):
    n = len(signal_mod)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(signal_mod))
    rr_band = (freqs >= 0.1) & (freqs <= 0.5)
    peak_freq = freqs[rr_band][np.argmax(fft_mag[rr_band])]
    return peak_freq * 60, freqs, fft_mag  # bpm, freq axis, spectrum

def estimate_rr_combined(ppg, fs):
    # Detect peaks
    peaks, _ = signal.find_peaks(ppg, distance=fs * 0.4)

    # AM: Amplitude modulation
    am_values = extract_am(ppg, peaks)
    t_am = peaks / fs
    am_interp = np.interp(np.linspace(t_am[0], t_am[-1], len(ppg)), t_am, am_values)
    rr_am, freqs_am, fft_am = estimate_rr(am_interp, fs)

    # BW: Baseline wander
    bw_signal = extract_bw(ppg, fs)
    rr_bw, freqs_bw, fft_bw = estimate_rr(bw_signal, fs)

    # FM: Frequency modulation (HRV)
    fm_signal, t_fm = extract_fm(ppg, fs)
    rr_fm, freqs_fm, fft_fm = estimate_rr(fm_signal, 4)

    # Combine estimate
    rr_combined = np.median([rr_am, rr_bw, rr_fm])

    # Plotting
    fig, axs = plt.subplots(4, 2, figsize=(15, 10))
    axs[0, 0].plot(ppg); axs[0, 0].set_title("Raw PPG")
    axs[1, 0].plot(bw_signal); axs[1, 0].set_title(f"Baseline Wander (RR = {rr_bw:.2f} bpm)")
    axs[2, 0].plot(am_interp); axs[2, 0].set_title(f"Amplitude Modulation (RR = {rr_am:.2f} bpm)")
    axs[3, 0].plot(t_fm, fm_signal); axs[3, 0].set_title(f"Frequency Modulation / HRV (RR = {rr_fm:.2f} bpm)")

    axs[1, 1].plot(freqs_bw, np.abs(fft_bw)); axs[1, 1].set_xlim(0.05, 0.6); axs[1, 1].set_title("BW FFT")
    axs[2, 1].plot(freqs_am, np.abs(fft_am)); axs[2, 1].set_xlim(0.05, 0.6); axs[2, 1].set_title("AM FFT")
    axs[3, 1].plot(freqs_fm, np.abs(fft_fm)); axs[3, 1].set_xlim(0.05, 0.6); axs[3, 1].set_title("FM FFT")

    axs[0, 1].axis("off")
    axs[0, 1].text(0.1, 0.5, f"Estimated RR (Combined): {rr_combined:.2f} bpm", fontsize=14)

    plt.tight_layout()
    plt.show()

    return {
        "RR_BW": rr_bw,
        "RR_AM": rr_am,
        "RR_FM": rr_fm,
        "RR_Combined": rr_combined
    }

# Example:
# ppg = np.loadtxt('your_ppg_data.txt')  # Load your signal here
# fs = 100  # Sampling frequency in Hz
# rr_results = estimate_rr_combined(ppg, fs)
# print(rr_results)