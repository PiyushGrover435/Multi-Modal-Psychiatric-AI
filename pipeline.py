import joblib
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

def tier1_classify(dass42_responses):
    try:
        model = joblib.load('models/sentin_edge_model.pkl')
        features = joblib.load('models/sentin_edge_model_features.pkl')
    except Exception as e:
        return {'label': 'Error: Model not found', 'confidence': 0.0, 'score': 0}
        
    df = pd.DataFrame([dass42_responses])
    # Ensure correct column order
    df = df[features]
    
    pred_label = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    
    confidence = max(proba)
    score = sum(dass42_responses.values())
    
    return {
        'label': pred_label,
        'confidence': confidence,
        'score': score
    }

def run_pipeline(dass42_responses: dict, audio_data=None, eeg_progress_callback=None) -> dict:
    # Tier 1
    t1_result = tier1_classify(dass42_responses)
    
    if t1_result['label'] != 'High Risk':
        return build_report(t1_result, t15_result=None, t2_result=None, flag="T1 Passed, Gating Stopped")
    
    # Tier 1.5 (gated)
    t15_result = tier15_validate_audio(audio_data)
    if not t15_result['validation_passed']:
        return build_report(t1_result, t15_result, t2_result=None, flag="T1.5 Refuted")
    
    # Tier 2 (gated)
    eeg_raw_path = r'data/raw/eeg_raw/synthetic_eeg_data_testv1.csv'
    eeg_processed_path = r'data/processed/processed_synthetic_eeg.csv'
    process_eeg_data(eeg_raw_path, eeg_processed_path, progress_callback=eeg_progress_callback)
    t2_result = tier2_eeg_inference(eeg_processed_path)
    return build_report(t1_result, t15_result, t2_result, flag="All Tiers Completed")
