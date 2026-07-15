import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "datasets"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 50)
print("Loading dataset...")

X_train = np.load(os.path.join(DATASET_PATH, "X_train_landmarks.npy"))
X_test = np.load(os.path.join(DATASET_PATH, "X_test_landmarks.npy"))

y_train = np.load(os.path.join(DATASET_PATH, "y_train.npy"))
y_test = np.load(os.path.join(DATASET_PATH, "y_test.npy"))

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Feature size     : {X_train.shape[1]}")

# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")

# ============================================================
# TEST MODEL
# ============================================================

print("\nEvaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# ============================================================
# SAVE MODEL
# ============================================================

model_file = os.path.join(MODEL_PATH, "best_model.pkl")

with open(model_file, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully.")

print(f"Location : {model_file}")

print("=" * 50)
print("DONE")
print("=" * 50)