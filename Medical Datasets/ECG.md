# ECG Dataset

## Introduction to ECG Data
Electrocardiogram (ECG or EKG) data represents the electrical activity of the heart, typically recorded over time. It is used for diagnosing and monitoring various cardiovascular conditions, such as arrhythmias, heart attacks, and other heart diseases. ECG signals are typically represented as time-series data where each data point corresponds to the heart's electrical activity at a specific time.

### Key Features of ECG Data:
- **Time-series Data:** ECG data is typically represented as continuous time-series, with each value indicating the electrical potential at a specific time.
- **Heart Rate Variability:** Variation in the time interval between successive R-peaks, an important feature for diagnosing cardiac conditions.
- **Waveforms:** ECG data typically includes several key waveforms: P-wave, QRS complex, and T-wave, which are related to different phases of the heart's electrical cycle.
- **Sampling Rate:** ECG signals are often sampled at high rates (e.g., 250 Hz to 1000 Hz) to capture detailed heart activity.

---

## Structure of ECG Data
The ECG dataset typically includes several files, each corresponding to an ECG recording for a patient. The data may be in different formats, such as CSV, EDF (European Data Format), or even in specialized formats like PhysioNet’s WFDB format.

### Common ECG Data Formats:
- **CSV Format:** A simple text format with time-series data, where each row corresponds to a timestamp and its respective ECG value.
- **WFDB Format:** The PhysioNet database often uses the WFDB format, which includes both header files and data files, storing ECG signals and their metadata.
- **EDF Format:** A standard file format used to store continuous physiological signals, such as ECG, EEG, and EMG.

### Components of ECG Data:
- **Lead Channels:** ECG data can include multiple leads, which are the different electrode placements on the body. Common leads are Lead I, Lead II, Lead III, V1, V2, etc.
- **Signal Values:** The signal represents the electrical activity of the heart recorded over time, typically with units of microvolts (µV).
- **Annotations:** In some datasets, annotations mark specific events in the ECG signal (e.g., R-peaks, QRS complex, arrhythmia events).

---

## Preprocessing ECG Data
Preprocessing ECG data is crucial to remove noise, normalize signals, extract features, and prepare the data for machine learning models.

### 1. Noise Filtering
ECG signals can be affected by noise from sources such as power-line interference, motion artifacts, and muscle contractions. A bandpass filter helps remove unwanted frequency components.

**Code Example:**
```python
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Generate a simulated noisy ECG signal
ecgr = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 1000)) + np.random.normal(0, 0.1, 1000)

# Design a bandpass filter
fs = 1000  # Sampling frequency
low_cutoff = 0.5  # Low cutoff frequency
high_cutoff = 50  # High cutoff frequency
b, a = signal.butter(4, [low_cutoff / (0.5 * fs), high_cutoff / (0.5 * fs)], btype='band')
filtered_ecg = signal.filtfilt(b, a, ecgr)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(ecgr)
plt.title("Original Noisy ECG Signal")
plt.subplot(2, 1, 2)
plt.plot(filtered_ecg)
plt.title("Filtered ECG Signal")
plt.tight_layout()
plt.show()
```

**Explanation of Code:**
- **Generates a noisy ECG signal** (replace with real ECG data).
- **Applies a Butterworth bandpass filter** to remove noise outside the 0.5-50 Hz range.
- **Plots the original and filtered signals.**

**Expected Output:**
- A filtered ECG signal with reduced noise.

---

### 2. Signal Normalization
Normalization ensures that ECG values fall within a consistent range, improving model performance.

**Code Example:**
```python
# Normalize the ECG signal to range [0, 1]
normalized_ecg = (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg))

# Alternatively, normalize to range [-1, 1]
normalized_ecg_alt = 2 * (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg)) - 1
```

**Explanation of Code:**
- **Rescales ECG values** to the range [0,1] or [-1,1].
- **Helps with convergence during machine learning training.**

**Expected Output:**
- Normalized ECG signal.

---

### 3. Baseline Wander Removal
Baseline wander is a low-frequency noise component that can distort ECG signals.

**Code Example:**
```python
# Remove baseline wander using a high-pass filter
baseline_wander_filtered = signal.detrend(filtered_ecg)
```

**Explanation of Code:**
- **Removes slow signal drifts** caused by respiration or electrode movement.
- **Uses `detrend` function** to eliminate baseline shifts.

**Expected Output:**
- ECG signal without baseline drift.

---

### 4. Peak Detection (R-Peak Identification)
Detecting R-peaks is essential for heart rate variability (HRV) analysis and classification tasks.

**Code Example:**
```python
from scipy.signal import find_peaks

# Detect R-peaks
peaks, _ = find_peaks(filtered_ecg, height=0.5, distance=50)  # Adjust height & distance accordingly

# Plot detected peaks
plt.plot(filtered_ecg)
plt.plot(peaks, filtered_ecg[peaks], "ro")  # Mark R-peaks
plt.title("Detected R-Peaks")
plt.show()
```

**Explanation of Code:**
- **Uses `find_peaks`** to identify R-peaks in the ECG signal.
- **Adjusts height and distance parameters** based on signal properties.
- **Plots the detected peaks.**

**Expected Output:**
- An ECG signal with R-peaks marked.

---

### 5. Segmenting ECG Beats
ECG signals are often divided into segments for training deep learning models.

**Code Example:**
```python
# Segment the ECG signal into beats of 256 samples each
ecgr_segments = np.array([filtered_ecg[i:i+256] for i in range(0, len(filtered_ecg)-256, 256)])
```

**Explanation of Code:**
- **Splits ECG signals** into fixed-length segments (e.g., 256 samples).
- **Useful for deep learning models like CNNs and LSTMs.**

**Expected Output:**
- An array of segmented ECG beats.

---

### 6. Converting ECG Data into Images
For CNN-based models, ECG signals can be converted into spectrograms or waveforms.

**Code Example:**
```python
import matplotlib.pyplot as plt
import os

# Define output directory
output_dir = "ecg_images"
os.makedirs(output_dir, exist_ok=True)

# Save each ECG segment as an image
for i, segment in enumerate(ecgr_segments):
    plt.figure(figsize=(3, 3))
    plt.plot(segment)
    plt.axis("off")  # Remove axes
    plt.savefig(f"{output_dir}/ecg_{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()
```

**Explanation of Code:**
- **Creates spectrogram-like images** of ECG waveforms.
- **Useful for CNN-based image classification models.**

**Expected Output:**
- ECG waveform images stored as PNG files.

---

### 7. Transforming ECG Data for LSTMs
For sequential models like LSTMs, ECG data needs to be reshaped properly.

**Code Example:**
```python
# Reshape for LSTM input
lstm_ready_ecg = ecgr_segments.reshape(ecgr_segments.shape[0], ecgr_segments.shape[1], 1)
```

**Explanation of Code:**
- **Reshapes ECG segments** to match LSTM input format `(samples, timesteps, features)`.

**Expected Output:**
- A properly shaped ECG dataset for LSTM models.

---

## Conclusion
Preprocessing ECG data is crucial for noise reduction, normalization, feature extraction, and preparing the data for machine learning models. The techniques described ensure that ECG signals are properly formatted for deep learning models like CNNs and LSTMs. By following these steps, ECG data can be effectively used in AI-driven medical diagnostics and real-time heart monitoring systems.

