# ============================================================
# train.py — Lightweight ISL Training (SVM + HOG, no PyTorch)
# ============================================================

import os
import numpy as np
import pickle
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CONFIG
# ============================================================
BASE_PATH    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "datasets")
MODEL_PATH   = os.path.join(BASE_PATH, "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "="*55)
print("   ISL Sign Language — SVM + HOG Trainer")
print("="*55)

print("\nLoading cleaned dataset...")

try:
    X_train = np.load(os.path.join(DATASET_PATH, "X_train_images.npy"))
    X_test  = np.load(os.path.join(DATASET_PATH, "X_test_images.npy"))
    y_train = np.load(os.path.join(DATASET_PATH, "y_train.npy"))
    y_test  = np.load(os.path.join(DATASET_PATH, "y_test.npy"))
except FileNotFoundError:
    raise FileNotFoundError("Dataset not found. Run clean.py first.")

with open(os.path.join(DATASET_PATH, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)
print(f"  ✓ Train samples : {len(X_train)}")
print(f"  ✓ Test  samples : {len(X_test)}")
print(f"  ✓ Classes       : {num_classes}")
print(f"  ✓ Class names   : {list(le.classes_)}")

# ============================================================
# HOG FEATURE EXTRACTION
# HOG captures edges and shape — ideal for hand signs
# ============================================================
def extract_hog_features(images, label=""):
    """
    Extracts HOG (Histogram of Oriented Gradients) features.
    Input : (N, 224, 224, 3) float32 normalized images
    Output: (N, feature_dim) float32 feature vectors
    """
    hog = cv2.HOGDescriptor(
        _winSize   =(64, 64),
        _blockSize =(16, 16),
        _blockStride=(8,  8),
        _cellSize  =(8,   8),
        _nbins     =9
    )

    features = []
    total    = len(images)

    for i, img in enumerate(images):
        # Convert normalized float → uint8
        img_uint8 = (img * 255).astype(np.uint8)

        # Resize to 64×64 for HOG (faster + consistent)
        img_small = cv2.resize(img_uint8, (64, 64))

        # Convert RGB → grayscale for HOG
        img_gray  = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)

        # Compute HOG features
        feat = hog.compute(img_gray).flatten()
        features.append(feat)

        # Progress
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  {label} [{i+1}/{total}] {pct:.1f}%")

    return np.array(features, dtype=np.float32)


print("\nExtracting HOG features from training set...")
X_train_hog = extract_hog_features(X_train, label="Train")

print("\nExtracting HOG features from test set...")
X_test_hog  = extract_hog_features(X_test,  label="Test")

print(f"\n  ✓ Feature vector size: {X_train_hog.shape[1]}")

# ============================================================
# SCALE FEATURES
# SVM works best with normalized features
# ============================================================
print("\nScaling features...")
scaler      = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog  = scaler.transform(X_test_hog)
print("  ✓ Features scaled")

# ============================================================
# TRAIN — SVM with RBF kernel
# Best choice for HOG features on sign recognition
# ============================================================
print("\nTraining SVM classifier...")
print("  (This may take 5–15 minutes depending on dataset size)")
print("  ─────────────────────────────────────────────────────")

svm_model = SVC(
    kernel      = "rbf",
    C           = 10,
    gamma       = "scale",
    probability = True,   # needed for confidence scores in pipeline
    verbose     = True,
    cache_size  = 2000    # use 2GB cache for faster training
)

svm_model.fit(X_train_hog, y_train)
print("\n  ✓ SVM training complete")

# ============================================================
# EVALUATE
# ============================================================
print("\nEvaluating on test set...")
y_pred = svm_model.predict(X_test_hog)
acc    = accuracy_score(y_test, y_pred) * 100

print(f"\n{'='*55}")
print(f"  Test Accuracy : {acc:.2f}%")
print(f"{'='*55}")

# Per-class breakdown
print("\nPer-class Report:")
print(classification_report(
    y_test, y_pred,
    target_names=[str(c) for c in le.classes_]
))

# ============================================================
# ALSO TRAIN RANDOM FOREST (backup model)
# Faster, good for real-time fallback
# ============================================================
print("\nTraining Random Forest (backup model)...")
rf_model = RandomForestClassifier(
    n_estimators = 200,
    max_depth    = None,
    n_jobs       = -1,    # use all CPU cores
    random_state = 42,
    verbose      = 1
)
rf_model.fit(X_train_hog, y_train)

rf_pred = rf_model.predict(X_test_hog)
rf_acc  = accuracy_score(y_test, rf_pred) * 100
print(f"\n  ✓ Random Forest Accuracy: {rf_acc:.2f}%")

# Pick best model
if rf_acc > acc:
    print(f"  ★ Random Forest is better ({rf_acc:.2f}% vs {acc:.2f}%)")
    best_model      = rf_model
    best_model_name = "RandomForest"
else:
    print(f"  ★ SVM is better ({acc:.2f}% vs {rf_acc:.2f}%)")
    best_model      = svm_model
    best_model_name = "SVM"

# ============================================================
# SAVE MODELS + SCALER
# ============================================================
print("\nSaving models...")

with open(os.path.join(MODEL_PATH, "sign_svm.pkl"), "wb") as f:
    pickle.dump(svm_model, f)
print("  ✓ SVM saved     → models/sign_svm.pkl")

with open(os.path.join(MODEL_PATH, "sign_rf.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
print("  ✓ Random Forest → models/sign_rf.pkl")

with open(os.path.join(MODEL_PATH, "hog_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("  ✓ Scaler saved  → models/hog_scaler.pkl")

# Save best model separately for pipeline
with open(os.path.join(MODEL_PATH, "best_model.pkl"), "wb") as f:
    pickle.dump({"model": best_model, "name": best_model_name}, f)
print(f"  ✓ Best model    → models/best_model.pkl ({best_model_name})")

print(f"\n{'='*55}")
print(f"  Training complete!")
print(f"  SVM Accuracy          : {acc:.2f}%")
print(f"  Random Forest Accuracy: {rf_acc:.2f}%")
print(f"  Best model            : {best_model_name}")
print(f"{'='*55}\n")