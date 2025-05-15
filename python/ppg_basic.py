import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import argparse
import json

class PPGRespirationAnalyzer:
    def __init__(self, sampling_rate=30, limit=500):
        """
        Initialize the PPG respiration analyzer
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate of the PPG signal in Hz (default: 30 Hz)
        """
        self.sampling_rate = sampling_rate
        self.limit = limit
        
        # Configuration parameters
        self.window_size = 10 * sampling_rate  # 10 seconds of data
        self.respiratory_min_freq = 0.16  # ~10 breaths per minute
        self.respiratory_max_freq = 0.5   # ~30 breaths per minute
        self.min_peak_distance = int(sampling_rate / self.respiratory_max_freq * 0.8)  # Min distance between peaks
    
    def load_data(self, file_path):
        """
        Load PPG data from file
        
        Parameters:
        -----------
        file_path : str
            Path to the PPG data file (CSV or JSON)
            
        Returns:
        --------
        ppg_data : numpy.ndarray
            Raw PPG signal
        timestamps : numpy.ndarray
            Timestamps corresponding to PPG samples
        """
        if file_path.endswith('.csv'):
            # Assuming CSV format with columns: timestamp, ppg_value
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns and 'ppg_value' in df.columns:
                timestamps = df['timestamp'].values
                ppg_data = df['ppg_value'].values
            else:
                # Fallback to first two columns
                timestamps = df.iloc[:, 0].values
                ppg_data = df.iloc[:, 1].values
        elif file_path.endswith('.json'):
            # Assuming JSON format with ppg_data and timestamps fields
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                # List of dictionaries with timestamp and value fields
                timestamps = np.array([entry.get('timestamp', i) for i, entry in enumerate(data)])
                ppg_data = np.array([entry.get('value', 0) for entry in data])
            else:
                # Dictionary with separate arrays
                timestamps = np.array(data.get('timestamps', range(len(data.get('ppg_data', [])))))
                ppg_data = np.array(data.get('ppg_data', []))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Ensure timestamps are monotonically increasing
        if not np.all(np.diff(timestamps) >= 0):
            raise ValueError("Timestamps must be monotonically increasing")
            
        return ppg_data[:self.limit], timestamps[:self.limit]
    
    def preprocess_signal(self, ppg_data):
        """
        Preprocess the PPG signal to remove noise and artifacts
        
        Parameters:
        -----------
        ppg_data : numpy.ndarray
            Raw PPG signal
            
        Returns:
        --------
        filtered_ppg : numpy.ndarray
            Filtered PPG signal
        """
        # Step 1: Remove outliers (simple z-score method)
        z_scores = np.abs((ppg_data - np.mean(ppg_data)) / np.std(ppg_data))
        ppg_cleaned = np.copy(ppg_data)
        ppg_cleaned[z_scores > 3] = np.nan  # Mark outliers as NaN
        
        # Interpolate NaNs
        mask = np.isnan(ppg_cleaned)
        ppg_cleaned[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ppg_cleaned[~mask])
        
        # Step 2: Apply low-pass filter to remove high-frequency noise
        b, a = butter(4, 5/(self.sampling_rate/2), 'low')
        filtered_ppg = filtfilt(b, a, ppg_cleaned)
        
        # Step 3: Detrend to remove baseline wander
        detrended_ppg = signal.detrend(filtered_ppg)
        
        return detrended_ppg
    
    def extract_respiratory_signal(self, ppg_data):
        """
        Extract respiratory signal from PPG data
        
        Parameters:
        -----------
        ppg_data : numpy.ndarray
            Preprocessed PPG signal
            
        Returns:
        --------
        resp_signal : numpy.ndarray
            Extracted respiratory signal
        """
        # Apply bandpass filter to isolate respiratory frequency components
        nyquist = self.sampling_rate / 2
        low = self.respiratory_min_freq / nyquist
        high = self.respiratory_max_freq / nyquist
        b, a = butter(2, [low, high], 'bandpass')
        resp_signal = filtfilt(b, a, ppg_data)
        
        return resp_signal
    
    def detect_respiration_rate(self, resp_signal):
        """
        Detect respiration rate from the respiratory signal
        
        Parameters:
        -----------
        resp_signal : numpy.ndarray
            Extracted respiratory signal
            
        Returns:
        --------
        resp_rate : float
            Estimated respiration rate in breaths per minute
        peak_indices : numpy.ndarray
            Indices of detected respiratory peaks
        """
        # Find peaks in the respiratory signal
        peak_indices, _ = find_peaks(resp_signal, distance=self.min_peak_distance)
        
        if len(peak_indices) < 2:
            return None, peak_indices
        
        # Calculate average time between peaks
        peak_times = peak_indices / self.sampling_rate
        intervals = np.diff(peak_times)
        avg_interval = np.mean(intervals)
        
        # Convert to breaths per minute
        resp_rate = 60 / avg_interval
        
        return resp_rate, peak_indices
    
    def analyze_ppg(self, ppg_data, timestamps=None):
        """
        Analyze PPG data to extract respiration rate
        
        Parameters:
        -----------
        ppg_data : numpy.ndarray
            Raw PPG signal
        timestamps : numpy.ndarray, optional
            Timestamps corresponding to PPG samples
            
        Returns:
        --------
        results : dict
            Dictionary containing analysis results
        """
        if timestamps is None:
            timestamps = np.arange(len(ppg_data)) / self.sampling_rate
            
        # Preprocess the signal
        filtered_ppg = self.preprocess_signal(ppg_data)
        
        # Extract respiratory signal
        resp_signal = self.extract_respiratory_signal(filtered_ppg)
        
        # Detect respiration rate
        resp_rate, peak_indices = self.detect_respiration_rate(resp_signal)
        
        # Calculate heart rate using peak detection on filtered PPG
        b, a = butter(4, [0.5/(self.sampling_rate/2), 4/(self.sampling_rate/2)], 'bandpass')
        heartbeat_signal = filtfilt(b, a, filtered_ppg)
        hr_peak_indices, _ = find_peaks(heartbeat_signal, distance=int(self.sampling_rate * 0.5))
        
        heart_rate = None
        if len(hr_peak_indices) >= 2:
            hr_intervals = np.diff(hr_peak_indices) / self.sampling_rate
            avg_hr_interval = np.mean(hr_intervals)
            heart_rate = 60 / avg_hr_interval
        
        results = {
            'respiration_rate': resp_rate,
            'heart_rate': heart_rate,
            'processed_data': {
                'raw_ppg': ppg_data,
                'filtered_ppg': filtered_ppg,
                'respiratory_signal': resp_signal,
                'resp_peak_indices': peak_indices,
                'hr_peak_indices': hr_peak_indices,
                'timestamps': timestamps
            }
        }
        
        return results
    
    def plot_results(self, results):
        """
        Plot PPG analysis results
        
        Parameters:
        -----------
        results : dict
            Dictionary containing analysis results
        """
        data = results['processed_data']
        timestamps = data['timestamps']
        time_seconds = (timestamps - timestamps[0]) / 1e9 if np.max(timestamps) > 1e9 else timestamps
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot raw and filtered PPG
        axes[0].plot(time_seconds, data['raw_ppg'], 'b-', alpha=0.5, label='Raw PPG')
        axes[0].plot(time_seconds, data['filtered_ppg'], 'r-', label='Filtered PPG')
        axes[0].set_title('PPG Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot respiratory signal
        axes[1].plot(time_seconds, data['respiratory_signal'], 'g-', label='Respiratory Signal')
        if len(data['resp_peak_indices']) > 0:
            axes[1].plot(time_seconds[data['resp_peak_indices']], 
                       data['respiratory_signal'][data['resp_peak_indices']], 
                       'ro', label='Respiratory Peaks')
        axes[1].set_title('Extracted Respiratory Signal')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot heart rate peaks
        axes[2].plot(time_seconds, data['filtered_ppg'], 'b-', label='Filtered PPG')
        if len(data['hr_peak_indices']) > 0:
            axes[2].plot(time_seconds[data['hr_peak_indices']], 
                       data['filtered_ppg'][data['hr_peak_indices']], 
                       'go', label='Heart Rate Peaks')
        axes[2].set_title('Heart Rate Detection')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Amplitude')
        axes[2].legend()
        axes[2].grid(True)
        
        # Add results as text
        resp_rate = results['respiration_rate']
        hr = results['heart_rate']
        
        info_text = f"Respiration Rate: {resp_rate:.1f} breaths/min\n" if resp_rate else "Respiration Rate: Not detected\n"
        info_text += f"Heart Rate: {hr:.1f} bpm" if hr else "Heart Rate: Not detected"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        return fig

def main():
    parser = argparse.ArgumentParser(description='PPG Respiration Rate Analysis Tool')
    parser.add_argument('file', help='Path to PPG data file (CSV or JSON)')
    parser.add_argument('--sampling-rate', type=float, default=125.0, help='Sampling rate in Hz (default: 30Hz)')
    parser.add_argument('--save-plot', action='store_true', default=True, help='Save plot to file')
    parser.add_argument('--output', default='output/ppg_analysis.png', help='Output file name for plot (default: ppg_analysis.png)')
    parser.add_argument('--limit', type=int, default=10000, help='Number of initial values to plot (default: 10000)')

    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PPGRespirationAnalyzer(sampling_rate=args.sampling_rate, limit=args.limit)
    
    try:
        # Load data
        ppg_data, timestamps = analyzer.load_data(args.file)

        ppg_data = ppg_data[:args.limit]
        timestamps = timestamps[:args.limit]
        
        # Analyze PPG data
        results = analyzer.analyze_ppg(ppg_data, timestamps)
        
        # Print results
        resp_rate = results['respiration_rate']
        hr = results['heart_rate']
        
        print(f"===== PPG Analysis Results =====")
        print(f"Data length: {len(ppg_data)} samples ({len(ppg_data)/args.sampling_rate:.1f} seconds)")
        print(f"Respiration Rate: {resp_rate:.1f} breaths/min" if resp_rate else "Respiration Rate: Not detected")
        print(f"Heart Rate: {hr:.1f} bpm" if hr else "Heart Rate: Not detected")
        
        # Plot results
        fig = analyzer.plot_results(results)
        
        if args.save_plot:
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {args.output}")
        else:
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()