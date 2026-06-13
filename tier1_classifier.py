"""
Tier 1 Classifier — DASS-42 Behavioral Risk Classifier
=======================================================
Bayesian Hyperparameter Optimization via Optuna TPE (50 trials).
Optimizes LGBMClassifier to MAXIMIZE weighted macro F1-score to
properly handle clinical class imbalance in psychiatric risk labels.

Final model is serialized to models/ using pickle.
"""

import os
import pickle
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Silence noisy LightGBM / Optuna logs ─────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH           = "data/processed/cleaned_dass42.csv"
MODEL_OUT_PATH      = "models/sentin_edge_model.pkl"
FEATURES_OUT_PATH   = "models/sentin_edge_model_features.pkl"
CLASSES_OUT_PATH    = "models/sentin_edge_model_classes.pkl"
N_TRIALS            = 50
CV_FOLDS            = 3          # 3-fold keeps each trial ~10-15s instead of ~47s
RANDOM_STATE        = 42
TRIAL_TIMEOUT_SEC   = 120        # skip runaway trials (slow hyperparams)
F1_AVERAGE          = "weighted"   # weighted handles per-class support; change to 'macro' if preferred


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
            # Tree structure — tightened upper bounds for speed
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),

            # Learning dynamics — floor raised to 0.01 to avoid 1000-tree crawls
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500, step=50),

            # Regularization
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),

            # Sub-sampling (prevents overfitting, improves generalization)
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 5),

            # Class imbalance — critical for clinical psychiatric labels
            "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),

            # Fixed / infrastructure
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
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )

            preds  = clf.predict(X_val)
            score  = f1_score(y_val, preds, average=F1_AVERAGE, zero_division=0)
            scores.append(score)

            # Optuna pruning — abort bad trials early
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(scores))

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────
def train_tier1_model():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Tier 1 — DASS-42 Behavioral Risk Classifier (Optuna TPE)")
    print("=" * 65)
    print(f"\n[1/5] Loading processed DASS-42 data from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    print(f"      {len(df)} records  |  {len(df.columns)} columns")

    X = df.drop("label", axis=1)
    y = df["label"]

    # ── 2. Hold-out split (Optuna trains on X_train; final eval on X_test) ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train)}  |  Test (held-out): {len(X_test)}")
    print(f"      Class distribution (train):\n{y_train.value_counts().to_string()}\n")

    # ── 3. Optuna TPE study — 50 trials ──────────────────────────────────────
    print(f"[2/5] Launching Optuna TPE study  ({N_TRIALS} trials, {CV_FOLDS}-fold CV) ...")
    print(f"      Optimization target: {F1_AVERAGE.upper()} F1-score\n")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="tier1_lgbm_tpe",
    )

    study.optimize(
        build_objective(X_train, y_train),
        n_trials=N_TRIALS,
        timeout=None,
        show_progress_bar=True,
        catch=(Exception,),           # don't crash on rare sklearn edge cases
    )

    # ── 4. Report best trial ──────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_params = study.best_params
    print(f"\n[3/5] Optuna complete.")
    print(f"      Best trial  : #{best_trial.number}")
    print(f"      Best {F1_AVERAGE} F1 (CV mean): {best_trial.value:.5f}")
    print("      Best hyperparameters:")
    for k, v in best_params.items():
        print(f"        {k:30s} = {v}")

    # ── 5. Final training on FULL train set with best params ──────────────────
    print(f"\n[4/5] Training final LGBMClassifier on full training set ...")
    final_params = {
        **best_params,
        "n_jobs":       -1,
        "random_state": RANDOM_STATE,
        "verbose":      -1,
    }
    final_clf = lgb.LGBMClassifier(**final_params)
    final_clf.fit(X_train, y_train)

    # ── 6. Evaluate on held-out test set ─────────────────────────────────────
    y_pred      = final_clf.predict(X_test)
    test_f1     = f1_score(y_test, y_pred, average=F1_AVERAGE, zero_division=0)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)

    print(f"\n      Held-out Test Results")
    print(f"      ----------------------")
    print(f"      {F1_AVERAGE.capitalize()} F1-score : {test_f1:.5f}")
    print(f"      Macro    F1-score : {macro_f1:.5f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── 7. Serialize ──────────────────────────────────────────────────────────
    print(f"[5/5] Serializing model artifacts ...")
    os.makedirs("models", exist_ok=True)

    model_payload = {
        "model":         final_clf,
        "feature_cols":  list(X.columns),
        "classes":       list(final_clf.classes_),
        "best_params":   best_params,
        "best_cv_f1":    best_trial.value,
        "test_f1":       test_f1,
        "macro_f1":      macro_f1,
        "f1_average":    F1_AVERAGE,
        "n_trials":      N_TRIALS,
    }

    # Primary: pickle (as specified)
    with open(MODEL_OUT_PATH, "wb") as fh:
        pickle.dump(model_payload, fh)

    # Compatibility shims (joblib) for downstream consumers
    joblib.dump(final_clf,            FEATURES_OUT_PATH.replace("features", "model_jl"))
    joblib.dump(list(X.columns),      FEATURES_OUT_PATH)
    joblib.dump(list(final_clf.classes_), CLASSES_OUT_PATH)

    print(f"\n  [OK] Model   -> {MODEL_OUT_PATH}")
    print(f"  [OK] Features -> {FEATURES_OUT_PATH}")
    print(f"  [OK] Classes  -> {CLASSES_OUT_PATH}")
    print("\n[DONE] Tier 1 Optuna TPE training pipeline complete.\n")


if __name__ == "__main__":
    train_tier1_model()
