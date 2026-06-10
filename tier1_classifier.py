"""
Tier 1 Classifier — DASS-42 Behavioral Risk Classifier
Uses LightGBM with class_weight='balanced'
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_tier1_model():
    print("Loading processed DASS-42 data...")
    df = pd.read_csv('data/processed/cleaned_dass42.csv')
    print(f"  Loaded {len(df)} records with columns: {list(df.columns)}")

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("Training LightGBM Classifier (Tier 1)...")
    clf = lgb.LGBMClassifier(
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/sentin_edge_model.pkl')
    joblib.dump(list(X.columns), 'models/sentin_edge_model_features.pkl')
    joblib.dump(list(clf.classes_), 'models/sentin_edge_model_classes.pkl')
    print("\n[DONE] Tier 1 Model saved to models/sentin_edge_model.pkl")

if __name__ == '__main__':
    train_tier1_model()
