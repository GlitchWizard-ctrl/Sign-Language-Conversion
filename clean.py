# ============================================================
# clean.py — ISL Dataset Preprocessing Pipeline
# ============================================================

import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import cv2
except ImportError:
    raise ImportError("Run: pip install opencv-python")

# ============================================================
# CONFIG
# ============================================================
DATA_PATH   = r"C:\Users\asus\Downloads\data"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
IMG_SIZE    = (224, 224)

os.makedirs(OUTPUT_PATH, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data folder not found at: {DATA_PATH}")

# ============================================================
# HELPER: Background filtering using OpenCV (no MediaPipe)
# Detects skin-colored hand region and crops it
# ============================================================
def extract_hand_region(img_rgb):
    """
    Uses HSV skin detection to find and crop the hand region.
    Returns cropped RGB image or None if no hand detected.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70],  dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img_hsv, lower_skin, upper_skin)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get largest contour (assumed to be the hand)
    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 3000:  # Too small → not a hand
        return None

    x, y, w, h = cv2.boundingRect(largest)

    # Add padding
    pad    = 20
    h_img, w_img = img_rgb.shape[:2]
    x1     = max(x - pad, 0)
    y1     = max(y - pad, 0)
    x2     = min(x + w + pad, w_img)
    y2     = min(y + h + pad, h_img)

    cropped = img_rgb[y1:y2, x1:x2]
    return cropped


# ============================================================
# MAIN: Load & Preprocess Dataset
# ============================================================
images  = []
labels  = []
skipped = 0
loaded  = 0

print(f"\nLoading ISL dataset from: {DATA_PATH}")
print(f"Target size: {IMG_SIZE} | Color: RGB | Background: OpenCV skin filter\n")

for label in sorted(os.listdir(DATA_PATH)):
    folder_path = os.path.join(DATA_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            skipped += 1
            continue

        # BGR → RGB (keep color, no grayscale)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Background filtering
        cropped = extract_hand_region(img_rgb)

        if cropped is None or cropped.size == 0:
            skipped += 1
            continue

        # Resize to 224×224 for MobileNet / ResNet-50
        img_resized   = cv2.resize(cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize 0–255 → 0.0–1.0
        img_normalized = img_resized.astype(np.float32) / 255.0

        images.append(img_normalized)
        labels.append(label)
        loaded += 1

    print(f"  ✓ Class '{label}': processed")

print(f"\n{'='*50}")
print(f"  Loaded  : {loaded} images")
print(f"  Skipped : {skipped} (corrupt or no hand detected)")
print(f"  Classes : {len(set(labels))}")
print(f"{'='*50}\n")

# ============================================================
# ENCODE LABELS
# ============================================================
le             = LabelEncoder()
labels_encoded = le.fit_transform(labels)
print(f"Classes: {list(le.classes_)}")

# ============================================================
# TRAIN / TEST SPLIT (80/20)
# ============================================================
images = np.array(images)  # shape: (N, 224, 224, 3)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# SAVE CLEANED DATASET
# ============================================================
np.save(os.path.join(OUTPUT_PATH, "X_train_images.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "X_test_images.npy"),  X_test)
np.save(os.path.join(OUTPUT_PATH, "y_train.npy"),        y_train)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"),         y_test)

with open(os.path.join(OUTPUT_PATH, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\nDataset saved to: {OUTPUT_PATH}")
print("Files: X_train_images, X_test_images, y_train, y_test, label_encoder")