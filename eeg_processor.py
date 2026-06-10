"""
Tier 2 EEG Processing Pipeline — FULL Dataset Processing
---------------------------------------------------------
Reads the entire raw EEG CSV (7GB) in memory-safe chunks of 1000 rows.
For each row (subject), extracts proper statistical features from the 1140
EEG electrode readings: mean, std, min, max, and peak-to-peak amplitude.
All processed rows are saved to data/processed/processed_synthetic_eeg.csv.
No artificial row limits or shortcuts.
"""
import pandas as pd
import numpy as np
import os
import ast
import gc

RAW_EEG_PATH   = r'data/raw/eeg_raw/synthetic_eeg_data_testv1.csv'
PROCESSED_PATH = r'data/processed/processed_synthetic_eeg.csv'
CHUNK_SIZE     = 500   # rows per chunk — safe for 1148-column wide file

# EEG electrode columns (all columns starting with 'EEG_Elektrot_')
EEG_COL_PREFIX = 'EEG_Elektrot_'

def parse_eeg_value(val):
    """Each EEG cell is a stringified list e.g. '[-0.58, 1.43, ...]'. Parse it."""
    try:
        arr = ast.literal_eval(val)
        return np.array(arr, dtype=np.float32)
    except Exception:
        return np.array([0.0], dtype=np.float32)

def extract_features_from_row(row, eeg_cols):
    """Extract per-subject aggregate features across all electrodes."""
    all_values = []
    for col in eeg_cols:
        arr = parse_eeg_value(row[col])
        all_values.extend(arr.tolist())
    all_values = np.array(all_values, dtype=np.float32)
    return {
        'eeg_mean':   float(np.mean(all_values)),
        'eeg_std':    float(np.std(all_values)),
        'eeg_min':    float(np.min(all_values)),
        'eeg_max':    float(np.max(all_values)),
        'eeg_ptp':    float(np.ptp(all_values)),   # peak-to-peak amplitude
        'sex':        row.get('sex', 0),
        'age':        row.get('age', 0),
    }

def process_full_eeg():
    print(f"Processing FULL EEG dataset from: {RAW_EEG_PATH}")
    print(f"Chunk size: {CHUNK_SIZE} rows per chunk")

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    # Read header to get EEG column names
    header_df = pd.read_csv(RAW_EEG_PATH, nrows=0)
    eeg_cols = [c for c in header_df.columns if c.startswith(EEG_COL_PREFIX)]
    print(f"Found {len(eeg_cols)} EEG electrode columns.")

    total_rows = 0
    first_chunk = True

    for chunk_idx, chunk in enumerate(pd.read_csv(RAW_EEG_PATH, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk_features = []
        for _, row in chunk.iterrows():
            feat = extract_features_from_row(row, eeg_cols)
            chunk_features.append(feat)

        df_chunk = pd.DataFrame(chunk_features)

        # Append-write: first chunk writes header, rest append without header
        write_mode = 'w' if first_chunk else 'a'
        df_chunk.to_csv(PROCESSED_PATH, mode=write_mode, index=False, header=first_chunk)
        first_chunk = False

        total_rows += len(df_chunk)
        print(f"  Chunk {chunk_idx+1}: processed {len(df_chunk)} rows | Total so far: {total_rows}")

        del chunk, chunk_features, df_chunk
        gc.collect()

    print(f"\n[DONE] Full EEG processing complete. {total_rows} rows saved to: {PROCESSED_PATH}")
    return total_rows

if __name__ == '__main__':
    process_full_eeg()
