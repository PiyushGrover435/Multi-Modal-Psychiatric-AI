"""
Tier 2 EEG Classifier — LightGBM on Full Processed EEG Features
----------------------------------------------------------------
Loads the FULL processed_synthetic_eeg.csv (all subjects).
Performs 80/20 train-test split.
Trains LGBMClassifier optimized for speed and low memory footprint
(critical for edge-deployed, resource-constrained hardware).
Serializes trained model to models/tier2_eeg_model.pkl using pickle.
"""
import pandas as pd
import numpy as np
import pickle
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_PATH = r'data/processed/processed_synthetic_eeg.csv'
MODEL_OUT_PATH = r'models/tier2_eeg_model.pkl'

def train_tier2_model():
    print("Loading FULL processed EEG feature dataset...")
    df = pd.read_csv(PROCESSED_PATH)
    print(f"  Total records loaded: {len(df)} rows x {len(df.columns)} columns")

    # Drop rows with any NaN
    df = df.dropna()
    print(f"  After dropping NaN rows: {len(df)} records remaining")

    # ── Label Engineering ─────────────────────────────────────────────────────
    # Label: eeg_mean above median = High Risk (1), else Low Risk (0)
    # In production this would use expert-annotated clinical diagnoses.
    median_val = df['eeg_mean'].median()
    df['risk_label'] = (df['eeg_mean'] > median_val).astype(int)
    label_map = {0: 'Low Risk', 1: 'High Risk'}
    print(f"  Label distribution:\n{df['risk_label'].value_counts().rename(label_map)}\n")

    # ── Features ──────────────────────────────────────────────────────────────
    feature_cols = ['eeg_mean', 'eeg_std', 'eeg_min', 'eeg_max', 'eeg_ptp']
    for col in ['sex', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            feature_cols.append(col)

    X = df[feature_cols]
    y = df['risk_label']

    # ── Train / Test Split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── LightGBM Training ─────────────────────────────────────────────────────
    print("\nTraining LGBMClassifier (Tier 2 EEG)...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=-1          # suppress LightGBM console spam
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

    # ── Serialize with pickle ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    model_payload = {
        'model':         clf,
        'feature_cols':  feature_cols,
        'label_map':     label_map,
        'test_accuracy': acc,
    }
    with open(MODEL_OUT_PATH, 'wb') as f:
        pickle.dump(model_payload, f)

    print(f"\n[DONE] Tier 2 LightGBM model serialized -> {MODEL_OUT_PATH}")

if __name__ == '__main__':
    train_tier2_model()
