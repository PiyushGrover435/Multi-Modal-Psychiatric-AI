import pandas as pd
import numpy as np
import pickle
import os
from tier2_eeg import _vectorized_extract

eeg_raw_path = r'data/raw/eeg_raw/synthetic_eeg_data_testv1.csv'
MODEL_OUT_PATH = r'models/tier2_eeg_model.pkl'

with open(MODEL_OUT_PATH, "rb") as fh:
    payload = pickle.load(fh)
model = payload["model"]
feature_cols = payload["feature_cols"]

header_df = pd.read_csv(eeg_raw_path, nrows=0)
eeg_cols  = [c for c in header_df.columns if c.startswith("EEG_Elektrot_")]

chunk_iter = pd.read_csv(eeg_raw_path, chunksize=500, low_memory=False)

found_high = False
found_normal = False

print("Scanning 20% dataset for a High Risk and a Normal patient...")
for i, chunk in enumerate(chunk_iter):
    if found_high and found_normal:
        break
        
    features = _vectorized_extract(chunk, eeg_cols)
    
    # Predict using the loaded model
    preds = model.predict(features[feature_cols])
    
    high_count = np.sum(preds == 'High Risk')
    normal_count = np.sum(preds == 'Normal/Mild')
    total = len(preds)
    
    if (high_count / total) > 0.4 and not found_high:
        chunk.to_csv("test_patient_high_risk.csv", index=False)
        found_high = True
        print(f"-> Created 'test_patient_high_risk.csv' (Chunk {i})")
        
    elif (normal_count / total) > 0.6 and not found_normal:
        chunk.to_csv("test_patient_normal.csv", index=False)
        found_normal = True
        print(f"-> Created 'test_patient_normal.csv' (Chunk {i})")

print("SUCCESS: 2 custom test files generated!")
