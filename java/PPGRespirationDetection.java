package com.example.ppgprocessing;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class PPGRespirationDetection extends AppCompatActivity implements SensorEventListener {
    private static final String TAG = "PPGRespirationDetection";
    
    // Configuration parameters
    private static final int WINDOW_SIZE = 10 * 30; // 10 seconds at 30Hz
    private static final int BUFFER_SIZE = 5 * 60 * 30; // 5 minutes at 30Hz
    private static final float LOW_PASS_ALPHA = 0.1f; // Low pass filter coefficient
    private static final float RESPIRATORY_MIN_FREQUENCY = 0.16f; // ~10 breaths per minute
    private static final float RESPIRATORY_MAX_FREQUENCY = 0.5f; // ~30 breaths per minute
    private static final int MIN_PEAK_DISTANCE = 30; // Minimum distance between peaks (samples)
    
    private SensorManager sensorManager;
    private Sensor ppgSensor;
    
    // Data buffers
    private List<Float> ppgBuffer = new ArrayList<>();
    private List<Float> filteredBuffer = new ArrayList<>();
    private List<Long> timestampBuffer = new ArrayList<>();
    
    // State variables
    private float lastFilteredValue = 0;
    private int respirationRate = 0;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize sensor
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        ppgSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE);
        
        if (ppgSensor == null) {
            Log.e(TAG, "PPG sensor not available on this device");
        }
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        if (ppgSensor != null) {
            sensorManager.registerListener(this, ppgSensor, SensorManager.SENSOR_DELAY_FASTEST);
            Log.i(TAG, "PPG sensor listener registered");
        }
    }
    
    @Override
    protected void onPause() {
        super.onPause();
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
            Log.i(TAG, "PPG sensor listener unregistered");
        }
    }
    
    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_HEART_RATE) {
            // Extract raw PPG value
            // Note: This is a simplification. Actual raw PPG data format may vary by device
            float rawPpgValue = event.values[0];
            long timestamp = event.timestamp;
            
            // Process the raw PPG value
            processPpgValue(rawPpgValue, timestamp);
        }
    }
    
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Handle accuracy changes if needed
    }
    
    private void processPpgValue(float value, long timestamp) {
        // Add raw value to buffer
        ppgBuffer.add(value);
        timestampBuffer.add(timestamp);
        
        // Apply low-pass filter to remove high-frequency noise
        float filteredValue = applyLowPassFilter(value);
        filteredBuffer.add(filteredValue);
        
        // Maintain buffer size
        if (ppgBuffer.size() > BUFFER_SIZE) {
            ppgBuffer.remove(0);
            filteredBuffer.remove(0);
            timestampBuffer.remove(0);
        }
        
        // Process complete window of data
        if (filteredBuffer.size() >= WINDOW_SIZE) {
            extractRespirationRate();
        }
    }
    
    private float applyLowPassFilter(float currentValue) {
        // Simple low-pass filter: y[n] = α * x[n] + (1 - α) * y[n-1]
        lastFilteredValue = LOW_PASS_ALPHA * currentValue + (1 - LOW_PASS_ALPHA) * lastFilteredValue;
        return lastFilteredValue;
    }
    
    private void extractRespirationRate() {
        // Get the most recent window of data
        int startIdx = Math.max(0, filteredBuffer.size() - WINDOW_SIZE);
        List<Float> windowData = new ArrayList<>(filteredBuffer.subList(startIdx, filteredBuffer.size()));
        List<Long> windowTimestamps = new ArrayList<>(timestampBuffer.subList(startIdx, timestampBuffer.size()));
        
        // Apply bandpass filtering to isolate respiratory frequency components
        List<Float> respiratoryBandData = applyBandpassFilter(windowData, RESPIRATORY_MIN_FREQUENCY, RESPIRATORY_MAX_FREQUENCY);
        
        // Find peaks in the respiratory band data
        List<Integer> peakIndices = findPeaks(respiratoryBandData, MIN_PEAK_DISTANCE);
        
        // Calculate respiration rate from peaks
        if (peakIndices.size() >= 2) {
            // Convert timestamps from nanoseconds to seconds
            float firstPeakTime = windowTimestamps.get(peakIndices.get(0)) / 1_000_000_000.0f;
            float lastPeakTime = windowTimestamps.get(peakIndices.get(peakIndices.size() - 1)) / 1_000_000_000.0f;
            
            // Calculate average period
            float totalTime = lastPeakTime - firstPeakTime;
            int numCycles = peakIndices.size() - 1;
            
            if (totalTime > 0 && numCycles > 0) {
                // Calculate breaths per minute
                respirationRate = Math.round(60.0f * numCycles / totalTime);
                Log.i(TAG, "Detected respiration rate: " + respirationRate + " breaths per minute");
                
                // Here you would update UI or notify listeners about the new respiration rate
            }
        } else {
            Log.d(TAG, "Not enough peaks detected to calculate respiration rate");
        }
    }
    
    private List<Float> applyBandpassFilter(List<Float> data, float lowFreq, float highFreq) {
        // Simple implementation using a moving average filter for demonstration
        // In a real implementation, you might use a proper bandpass filter like Butterworth
        
        int n = data.size();
        List<Float> filteredData = new ArrayList<>(n);
        
        // First apply a simple detrending (remove baseline/DC component)
        float mean = 0;
        for (float value : data) {
            mean += value;
        }
        mean /= n;
        
        for (int i = 0; i < n; i++) {
            filteredData.add(data.get(i) - mean);
        }
        
        // Apply smoothing to remove high-frequency components
        int smoothingWindow = Math.round(1.0f / highFreq); // Window size based on high cutoff
        smoothingWindow = Math.max(3, smoothingWindow); // Ensure minimum window size
        
        List<Float> smoothedData = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float sum = 0;
            int count = 0;
            
            for (int j = Math.max(0, i - smoothingWindow / 2); j < Math.min(n, i + smoothingWindow / 2 + 1); j++) {
                sum += filteredData.get(j);
                count++;
            }
            
            smoothedData.add(sum / count);
        }
        
        return smoothedData;
    }
    
    private List<Integer> findPeaks(List<Float> data, int minDistance) {
        List<Integer> peaks = new ArrayList<>();
        int n = data.size();
        
        if (n < 3) return peaks; // Need at least 3 points to find peaks
        
        for (int i = 1; i < n - 1; i++) {
            float prev = data.get(i - 1);
            float current = data.get(i);
            float next = data.get(i + 1);
            
            // Check if current point is a peak
            if (current > prev && current > next) {
                // Check distance from last peak
                if (peaks.isEmpty() || i - peaks.get(peaks.size() - 1) >= minDistance) {
                    peaks.add(i);
                } else {
                    // If this peak is higher than the last one within minDistance, replace it
                    int lastPeakIdx = peaks.get(peaks.size() - 1);
                    if (current > data.get(lastPeakIdx)) {
                        peaks.set(peaks.size() - 1, i);
                    }
                }
            }
        }
        
        return peaks;
    }
    
    // Additional methods to expose results
    public int getRespirationRate() {
        return respirationRate;
    }
    
    public List<Float> getRawPpgData() {
        return new ArrayList<>(ppgBuffer);
    }
    
    public List<Float> getFilteredPpgData() {
        return new ArrayList<>(filteredBuffer);
    }
}