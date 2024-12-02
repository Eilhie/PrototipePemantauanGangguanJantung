from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import wfdb
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and scaler
MODEL_PATH = 'saved_model/model.pkl'
SCALER_PATH = 'saved_model/scaler.pkl'
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Helper Functions ---
def extract_hrv_features(rr_intervals):
    """Extract time-domain and frequency-domain HRV features."""
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    # Frequency-domain HRV features
    fs_rr = 1 / (np.mean(rr_intervals) / 1000)  # Sampling frequency in Hz
    freqs, psd = welch(rr_intervals, fs=fs_rr, nperseg=len(rr_intervals))
    lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
    hf = np.trapz(psd[(freqs >= 0.15) & (freqs <= 0.4)])
    lf_hf_ratio = lf / hf if hf != 0 else 0
    
    return [mean_rr, sdnn, rmssd, lf, hf, lf_hf_ratio]

def preprocess_ecg_segment(ecg_signal, r_peaks, fs, segment_duration=60):
    """Preprocess ECG signal into HRV features for each segment."""
    segment_samples = segment_duration * fs
    num_segments = len(ecg_signal) // segment_samples
    all_features = []

    for seg_idx in range(num_segments):
        start_idx = seg_idx * segment_samples
        end_idx = start_idx + segment_samples
        segment_r_peaks = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]

        if len(segment_r_peaks) > 1:  # Ensure enough R-peaks
            rr_intervals = np.diff(segment_r_peaks) / fs * 1000
            hrv_features = extract_hrv_features(rr_intervals)
        else:
            hrv_features = [np.nan] * 6  # Missing HRV features for this segment

        all_features.append(hrv_features + [seg_idx])
    return all_features

def process_patient_data(ecg_file, qrs_file, segment_duration=60):
    """Extract HRV features from a single patient's ECG and QRS data."""
    # Load ECG data
    record = wfdb.rdrecord(ecg_file)
    ecg_signal = record.p_signal[:, 0]
    fs = record.fs  # Sampling frequency

    # Load QRS detections
    qrs_annotations = wfdb.rdann(qrs_file, 'qrs')
    r_peaks = qrs_annotations.sample

    # Preprocess segments
    features = preprocess_ecg_segment(ecg_signal, r_peaks, fs, segment_duration)
    return features

def predict_hrv_features(features):
    """Make predictions using the trained model."""
    # Handle NaN values
    features = np.nan_to_num(features)

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    predictions = model.predict(scaled_features)
    return ['stress' if p == 1 else 'no_stress' for p in predictions]

# --- Flask Routes ---
@app.route('/')
def home():
    return "Welcome to the HRV Inference API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded files
        ecg_file = request.files.get('ecg_file')
        qrs_file = request.files.get('qrs_file')

        if not ecg_file or not qrs_file:
            return jsonify({"error": "Both ECG and QRS files are required"}), 400

        # Save files temporarily
        ecg_path = os.path.join(UPLOAD_FOLDER, 'ecg_file')
        qrs_path = os.path.join(UPLOAD_FOLDER, 'qrs_file')
        ecg_file.save(ecg_path)
        qrs_file.save(qrs_path)

        # Process data
        features = process_patient_data(ecg_path, qrs_path, segment_duration=60)
        columns = ['Mean RR', 'SDNN', 'RMSSD', 'LF', 'HF', 'LF/HF', 'Segment']
        df = pd.DataFrame(features, columns=columns)

        # Predict stress/no-stress
        predictions = predict_hrv_features(df.iloc[:, :-1].values)
        df['Prediction'] = predictions

        # Clean up temporary files
        os.remove(ecg_path)
        os.remove(qrs_path)

        # Return results as JSON
        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
