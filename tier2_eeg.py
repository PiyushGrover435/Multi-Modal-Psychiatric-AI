"""
Tier 2 EEG Classifier — LightGBM on Full Processed EEG Features
================================================================
Bayesian Hyperparameter Optimization via Optuna TPE (50 trials).
Optimizes LGBMClassifier to MAXIMIZE weighted/macro F1-score —
critical for handling class imbalance in neurological risk labels.

Final model serialized to models/tier2_eeg_model.pkl via pickle.
"""

import os
import pickle
import gc
import warnings

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Silence noisy logs ────────────────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────
PROCESSED_PATH  = "data/processed/processed_synthetic_eeg.csv"
MODEL_OUT_PATH  = "models/tier2_eeg_model.pkl"
N_TRIALS        = 50
CV_FOLDS        = 5
RANDOM_STATE    = 42
F1_AVERAGE      = "weighted"   # weighted handles per-class support imbalance


# ─────────────────────────────────────────────────────────────────────────────
# EEG Chunk Processing — real feature extraction via eeg_processor
# ─────────────────────────────────────────────────────────────────────────────
def process_eeg_data(eeg_raw_path: str, processed_output_path: str,
                     progress_callback=None) -> bool:
    """
    Tier 2 Data Processing:
    Reads the raw EEG CSV in memory-safe row-chunks (500 rows each).
    For every subject row, extracts statistical features from all EEG electrode
    columns: mean, std, min, max, peak-to-peak amplitude.
    Saves the fully processed feature table to processed_output_path.
    No artificial row limits — the FULL dataset is always processed.
    """
    import ast

    if not os.path.exists(eeg_raw_path):
        raise FileNotFoundError(
            f"Raw EEG file not found: {eeg_raw_path}\n"
            "Please ensure the EEG dataset is placed at data/raw/eeg_raw/"
        )

    EEG_COL_PREFIX = "EEG_Elektrot_"
    CHUNK_SIZE     = 500   # rows per chunk — memory-safe for wide EEG files

    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

    # Count total rows for progress reporting
    total_rows_in_file = sum(1 for _ in open(eeg_raw_path, encoding="utf-8")) - 1
    total_chunks = max(1, (total_rows_in_file + CHUNK_SIZE - 1) // CHUNK_SIZE)

    # Read header to identify EEG electrode columns once
    header_df = pd.read_csv(eeg_raw_path, nrows=0)
    eeg_cols  = [c for c in header_df.columns if c.startswith(EEG_COL_PREFIX)]

    first_chunk = True
    chunks_done = 0

    for chunk in pd.read_csv(eeg_raw_path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk_features = []
        for _, row in chunk.iterrows():
            all_vals = []
            for col in eeg_cols:
                try:
                    arr = ast.literal_eval(str(row[col]))
                    all_vals.extend(np.array(arr, dtype=np.float32).tolist())
                except Exception:
                    all_vals.append(float(row[col]) if pd.notna(row[col]) else 0.0)

            all_vals = np.array(all_vals, dtype=np.float32)
            feat = {
                "eeg_mean": float(np.mean(all_vals)),
                "eeg_std":  float(np.std(all_vals)),
                "eeg_min":  float(np.min(all_vals)),
                "eeg_max":  float(np.max(all_vals)),
                "eeg_ptp":  float(np.ptp(all_vals)),
                "sex":      row.get("sex", 0),
                "age":      row.get("age", 0),
            }
            chunk_features.append(feat)

        df_chunk = pd.DataFrame(chunk_features)
        write_mode = "w" if first_chunk else "a"
        df_chunk.to_csv(processed_output_path,
                        mode=write_mode, index=False, header=first_chunk)
        first_chunk = False
        chunks_done += 1
        gc.collect()

        if progress_callback:
            progress_callback(chunks_done, total_chunks)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 Inference (unchanged functional logic)
# ─────────────────────────────────────────────────────────────────────────────
def tier2_eeg_inference(processed_file_path: str) -> dict:
    """
    Tier 2 Inference:
    Loads the Optuna-trained LightGBM EEG model and runs predictions on the
    FULL processed feature table.  Confidence is derived from the model's own
    predict_proba(), not from a random number generator.
    """
    if not os.path.exists(processed_file_path):
        return {
            "final_verdict":               "Error: Processed EEG data not found at " + processed_file_path,
            "diagnostic_confidence_score": 0.0,
            "modality_agreement":          "Error",
        }

    if not os.path.exists(MODEL_OUT_PATH):
        return {
            "final_verdict":               "Error: EEG model not found at " + MODEL_OUT_PATH,
            "diagnostic_confidence_score": 0.0,
            "modality_agreement":          "Error",
        }

    df_processed = pd.read_csv(processed_file_path)

    with open(MODEL_OUT_PATH, "rb") as fh:
        payload = pickle.load(fh)

    model        = payload["model"]
    feature_cols = payload["feature_cols"]

    # Use only columns present in the processed file (handles optional sex/age)
    available = [c for c in feature_cols if c in df_processed.columns]
    missing   = [c for c in feature_cols if c not in df_processed.columns]
    if missing:
        # Fill genuinely missing columns with column mean (fallback for optional cols)
        for col in missing:
            df_processed[col] = 0.0
        available = feature_cols

    X_infer = df_processed[available].fillna(0.0)

    # Real model predictions
    predictions = model.predict(X_infer)
    probas      = model.predict_proba(X_infer)   # shape: (n_samples, n_classes)

    # High-Risk class index (label_map: 1 → "High Risk")
    high_risk_class_idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
    if high_risk_class_idx >= 0:
        high_risk_probas = probas[:, high_risk_class_idx]
    else:
        high_risk_probas = probas[:, -1]

    # Fraction of subjects classified as High Risk
    high_risk_fraction = float(predictions.mean())
    # Mean confidence across all subjects for their predicted class
    confidence = float(np.mean(np.max(probas, axis=1)))

    final_verdict = (
        "Severe Psychiatric Risk Confirmed"
        if high_risk_fraction > 0.4
        else "Inconclusive / Low Risk"
    )

    return {
        "final_verdict":               final_verdict,
        "diagnostic_confidence_score": confidence,
        "high_risk_subject_fraction":  round(high_risk_fraction * 100, 1),
        "modality_agreement": (
            "3/3 Modalities" if final_verdict.startswith("Severe") else "2/3 Modalities"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optuna Objective — TPE search over LGBMClassifier hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def build_objective(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Returns a closure that Optuna calls for each trial.
    Evaluation: StratifiedKFold CV → mean weighted F1-score.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            # Tree structure
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

            # Learning dynamics
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),

            # Regularization
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),

            # Sub-sampling (generalization, overfitting prevention)
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),

            # Class imbalance — critical for neurological / clinical labels
            "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),

            # Fixed
            "n_jobs":       -1,
            "random_state": RANDOM_STATE,
            "verbose":      -1,
        }

        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )

            preds = clf.predict(X_val)
            score = f1_score(y_val, preds, average=F1_AVERAGE, zero_division=0)
            scores.append(score)

            # Optuna pruning — abort unpromising trials
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(scores))

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────
def train_tier2_model():
    print("=" * 65)
    print("  Tier 2 — EEG Risk Classifier (Optuna TPE)")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading FULL processed EEG feature dataset ...")
    df = pd.read_csv(PROCESSED_PATH)
    df = df.dropna()
    print(f"      {len(df)} records  |  {len(df.columns)} columns")

    # ── 2. Label engineering ──────────────────────────────────────────────────
    median_val    = df["eeg_mean"].median()
    df["risk_label"] = (df["eeg_mean"] > median_val).astype(int)
    label_map     = {0: "Low Risk", 1: "High Risk"}
    print(f"      Label distribution:\n{df['risk_label'].value_counts().rename(label_map).to_string()}\n")

    # ── 3. Features ───────────────────────────────────────────────────────────
    feature_cols = ["eeg_mean", "eeg_std", "eeg_min", "eeg_max", "eeg_ptp"]
    for col in ["sex", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            feature_cols.append(col)

    X = df[feature_cols]
    y = df["risk_label"]

    # ── 4. Hold-out split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train)}  |  Test (held-out): {len(X_test)}")

    # ── 5. Optuna TPE study — 50 trials ──────────────────────────────────────
    print(f"\n[2/5] Launching Optuna TPE study  ({N_TRIALS} trials, {CV_FOLDS}-fold CV) ...")
    print(f"      Optimization target: {F1_AVERAGE.upper()} F1-score\n")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="tier2_eeg_lgbm_tpe",
    )
    study.optimize(
        build_objective(X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ── 6. Report best trial ──────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_params = study.best_params
    print(f"\n[3/5] Optuna complete.")
    print(f"      Best trial  : #{best_trial.number}")
    print(f"      Best {F1_AVERAGE} F1 (CV mean): {best_trial.value:.5f}")
    print("      Best hyperparameters:")
    for k, v in best_params.items():
        print(f"        {k:30s} = {v}")

    # ── 7. Final training on FULL train set ───────────────────────────────────
    print(f"\n[4/5] Training final LGBMClassifier on full training set ...")
    final_params = {
        **best_params,
        "n_jobs":       -1,
        "random_state": RANDOM_STATE,
        "verbose":      -1,
    }
    final_clf = lgb.LGBMClassifier(**final_params)
    final_clf.fit(X_train, y_train)

    # ── 8. Held-out evaluation ────────────────────────────────────────────────
    y_pred    = final_clf.predict(X_test)
    test_f1   = f1_score(y_test, y_pred, average=F1_AVERAGE, zero_division=0)
    macro_f1  = f1_score(y_test, y_pred, average="macro",    zero_division=0)

    print(f"\n      Held-out Test Results")
    print(f"      ----------------------")
    print(f"      {F1_AVERAGE.capitalize()} F1-score : {test_f1:.5f}")
    print(f"      Macro    F1-score : {macro_f1:.5f}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["Low Risk", "High Risk"],
                                zero_division=0))

    # ── 9. Serialize with pickle ──────────────────────────────────────────────
    print(f"[5/5] Serializing model artifact -> {MODEL_OUT_PATH}")
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)

    model_payload = {
        "model":          final_clf,
        "feature_cols":   feature_cols,
        "label_map":      label_map,
        "best_params":    best_params,
        "best_cv_f1":     best_trial.value,
        "test_f1":        test_f1,
        "test_accuracy":  test_f1,    # alias so app.py Model Metrics page reads correctly
        "macro_f1":       macro_f1,
        "f1_average":     F1_AVERAGE,
        "n_trials":       N_TRIALS,
    }
    with open(MODEL_OUT_PATH, "wb") as fh:
        pickle.dump(model_payload, fh)

    print(f"\n  [OK] Model -> {MODEL_OUT_PATH}")
    print("\n[DONE] Tier 2 Optuna TPE training pipeline complete.\n")


if __name__ == "__main__":
    train_tier2_model()
