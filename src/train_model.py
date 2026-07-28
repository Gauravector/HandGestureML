"""
train_model.py

Trains two classifiers on hand-landmark data and keeps whichever one
scores higher on the held-out test set:
  1. Random Forest (scikit-learn)  -- fast, usually strong on this kind
     of tabular data, good baseline.
  2. A small 3-layer neural net (PyTorch) -- input 64 -> hidden 128 ->
     hidden 64 -> output (num gesture classes).

Pipeline:
  CSV (data/gesture_data.csv)
    -> normalize landmarks (position + scale invariant)
    -> encode "hand" (left/right) as a 0/1 feature
    -> encode "label" (gesture name) as an integer class id
    -> train/test split (stratified, so every gesture is represented
       proportionally in both sets)
    -> train Random Forest
    -> train PyTorch NN
    -> print accuracy, confusion matrix, per-class precision/recall for both
    -> save whichever model had higher test accuracy to models/

Run:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import joblib

DATA_PATH = "data/gesture_data.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df['label'].nunique()} gesture classes")
    print(df["label"].value_counts())
    return df


# ---------------------------------------------------------------------
# 2. NORMALIZE LANDMARKS
# ---------------------------------------------------------------------
# Raw (x, y, z) coordinates depend on where your hand is in the frame and
# how close it is to the camera. Two people making the same gesture at
# different distances would look totally different to the model unless
# we normalize. We fix this two ways:
#   - Subtract the wrist (landmark 0) from every point -> position-invariant
#     (gesture no longer depends on WHERE your hand is in the frame)
#   - Divide by the largest distance from wrist to any other point ->
#     scale-invariant (gesture no longer depends on hand size / distance
#     from camera)
def normalize_landmarks(row, landmark_cols):
    coords = row[landmark_cols].values.reshape(21, 3).astype(np.float32)
    wrist = coords[0].copy()
    coords = coords - wrist
    scale = np.sqrt((coords[:, 0] ** 2 + coords[:, 1] ** 2)).max()
    if scale > 1e-6:
        coords = coords / scale
    return coords.flatten()


def build_features(df):
    landmark_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    normalized = df.apply(
        lambda row: normalize_landmarks(row, landmark_cols), axis=1, result_type="expand"
    )
    normalized.columns = landmark_cols

    # "hand" is categorical text (left/right) -> turn into a number (0/1)
    # so the models can use it. Left vs right hand can matter because a
    # gesture can look mirror-flipped depending on which hand made it.
    hand_numeric = (df["hand"].str.lower() == "right").astype(int)
    hand_numeric.name = "hand_is_right"

    X = pd.concat([normalized, hand_numeric], axis=1)  # 63 + 1 = 64 features
    y = df["label"]
    return X, y


# ---------------------------------------------------------------------
# 3. RANDOM FOREST
# ---------------------------------------------------------------------
def train_random_forest(X_train, y_train_enc, X_test, y_test_enc, class_names):
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train_enc)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    print(f"Test accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test_enc, preds),
        index=class_names, columns=class_names,
    ))
    print("\nPer-class precision/recall/F1:")
    print(classification_report(y_test_enc, preds, target_names=class_names))

    return clf, acc


# ---------------------------------------------------------------------
# 4. PYTORCH NEURAL NET
# ---------------------------------------------------------------------
class GestureNet(nn.Module):
    """
    Simple 3-layer feedforward network:
      input (64 features) -> 128 -> 64 -> num_classes
    ReLU activations between layers. This is intentionally small/simple --
    with only a few thousand rows of data, a bigger network would just
    overfit and isn't needed.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_neural_net(X_train, y_train_enc, X_test, y_test_enc, class_names,
                      epochs=60, lr=1e-3, batch_size=32):
    print("\n" + "=" * 60)
    print("PYTORCH NEURAL NET")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long).to(device)

    model = GestureNet(input_dim=X_train.shape[1], num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss={total_loss / len(dataset):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test_enc, preds)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test_enc, preds),
        index=class_names, columns=class_names,
    ))
    print("\nPer-class precision/recall/F1:")
    print(classification_report(y_test_enc, preds, target_names=class_names))

    return model, acc, device


# ---------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    X, y = build_features(df)

    # Turn gesture name strings ("fist", "peace", ...) into integers
    # (0, 1, 2, ...) because both models need numeric labels to train on.
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    # stratify=y_enc keeps the class proportions the same in both the
    # train and test sets -- important with 10 classes so a rare gesture
    # doesn't accidentally end up with zero test examples.
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )
    print(f"\nTrain rows: {len(X_train)}   Test rows: {len(X_test)}")

    rf_model, rf_acc = train_random_forest(
        X_train, y_train_enc, X_test, y_test_enc, class_names
    )
    nn_model, nn_acc, device = train_neural_net(
        X_train, y_train_enc, X_test, y_test_enc, class_names
    )

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Random Forest accuracy: {rf_acc:.4f}")
    print(f"Neural Net accuracy:    {nn_acc:.4f}")

    # Save the label encoder either way -- whichever model we load later
    # needs it to turn predicted integers back into gesture names.
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    if rf_acc >= nn_acc:
        print("\n-> Random Forest wins. Saving models/gesture_model_rf.pkl")
        joblib.dump(rf_model, os.path.join(MODELS_DIR, "gesture_model_rf.pkl"))
        with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
            f.write("random_forest")
    else:
        print("\n-> Neural Net wins. Saving models/gesture_model_nn.pt")
        torch.save(nn_model.state_dict(), os.path.join(MODELS_DIR, "gesture_model_nn.pt"))
        with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
            f.write("neural_net")

    print("\nDone. Check the models/ folder for saved files.")


if __name__ == "__main__":
    main()