import pickle
import joblib
import numpy as np
import pandas as pd
from tier15_acoustic import tier15_validate_audio
from tier2_eeg import process_eeg_data, tier2_eeg_inference


def build_report(t1_result, t15_result=None, t2_result=None, flag=None):
    return {
        'tier1': t1_result,
        'tier15': t15_result,
        'tier2': t2_result,
        'flag': flag
    }


def tier1_classify(dass42_responses: dict) -> dict:
    """
    Runs the Optuna-optimized LightGBM DASS-42 classifier.

    sentin_edge_model.pkl is a pickle dict with keys:
        'model'        — the trained LGBMClassifier
        'feature_cols' — ordered list of column names expected by the model

    The feature vector is the 42 individual question scores (q1..q42),
    NOT a single aggregated score, so column order must be preserved exactly.
    """
    # ── Load model payload (pickle dict, NOT a raw joblib classifier) ──────────
    try:
        with open('models/sentin_edge_model.pkl', 'rb') as fh:
            payload = pickle.load(fh)

        # sentin_edge_model.pkl may be a dict (new format) or a bare classifier
        # (legacy joblib dump). Handle both gracefully.
        if isinstance(payload, dict):
            model        = payload['model']
            feature_cols = payload['feature_cols']
        else:
            # Legacy bare-classifier path (joblib.dump of the clf directly)
            model        = payload
            feature_cols = joblib.load('models/sentin_edge_model_features.pkl')

    except FileNotFoundError:
        return {'label': 'Error: Model not found', 'confidence': 0.0, 'score': 0}
    except Exception as e:
        return {'label': f'Error: {e}', 'confidence': 0.0, 'score': 0}

    # ── Build input DataFrame in the exact column order the model expects ──────
    df = pd.DataFrame([dass42_responses])

    # feature_cols may be ['q1', 'q2', ..., 'q42'] — align to that order
    # If any expected column is missing, fill with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    # ── Inference ─────────────────────────────────────────────────────────────
    pred_label = model.predict(df)[0]
    proba      = model.predict_proba(df)[0]
    confidence = float(np.max(proba))
    score      = int(sum(dass42_responses.values()))

    # pred_label may be an integer (0/1/2) if the model was trained on encoded
    # labels.  Map back to human-readable strings if needed.
    label_map = {0: 'Low Risk', 1: 'Moderate', 2: 'High Risk'}
    if isinstance(pred_label, (int, np.integer)):
        pred_label = label_map.get(int(pred_label), str(pred_label))

    return {
        'label':      str(pred_label),
        'confidence': confidence,
        'score':      score,
    }


def run_pipeline(dass42_responses: dict, audio_data=None,
                 eeg_progress_callback=None) -> dict:
    # Tier 1
    t1_result = tier1_classify(dass42_responses)

    if 'Error' in t1_result['label']:
        return build_report(t1_result, flag="Error in Tier 1")

    if t1_result['label'] != 'High Risk':
        return build_report(t1_result, t15_result=None, t2_result=None,
                            flag="T1 Passed, Gating Stopped")

    # Tier 1.5 (gated)
    t15_result = tier15_validate_audio(audio_data)
    if not t15_result['validation_passed']:
        return build_report(t1_result, t15_result, t2_result=None,
                            flag="T1.5 Refuted")

    # Tier 2 (gated)
    eeg_raw_path       = r'data/raw/eeg_raw/synthetic_eeg_data_testv1.csv'
    eeg_processed_path = r'data/processed/processed_synthetic_eeg.csv'
    process_eeg_data(eeg_raw_path, eeg_processed_path,
                     progress_callback=eeg_progress_callback)
    t2_result = tier2_eeg_inference(eeg_processed_path)
    return build_report(t1_result, t15_result, t2_result,
                        flag="All Tiers Completed")
