# =============================================================================
# train_model.py
# PURPOSE: Train a classifier on the landmark data collected by collect_data.py
#          and save it so control.py can use it for live predictions.
# HOW TO RUN: python src/train_model.py
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

DATA_FILE = os.path.join('data', 'gesture_data.csv')
MODEL_FILE = os.path.join('data', 'gesture_model.joblib')


def main():
    if not os.path.isfile(DATA_FILE):
        print(f"[ERROR] {DATA_FILE} not found. Run collect_data.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"[INFO] Loaded {len(df)} samples across {df['label'].nunique()} gestures:")
    print(df['label'].value_counts())

    X = df.drop(columns=['label'])
    y = df['label']

    # stratify=y keeps the same proportion of each gesture in train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest works well here: fast, no scaling needed, handles
    # 14+ classes fine, and gives us predict_proba() for confidence scores.
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n[EVAL] Held-out accuracy report:")
    print(classification_report(y_test, y_pred))

    # Save both the model AND the label list together, so control.py
    # always knows exactly what classes it's predicting.
    joblib.dump({'model': clf, 'labels': sorted(y.unique())}, MODEL_FILE)
    print(f"\n[SAVED] Model saved to {MODEL_FILE}")
    print("[NEXT] Edit gesture_actions.py to map each label to a real action,")
    print("       then run control.py.")


if __name__ == '__main__':
    main()