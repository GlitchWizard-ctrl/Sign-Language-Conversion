# ============================================================
# clean.py — ISL Landmark Extraction
# Top 100 conversational classes, no face landmarks
# ============================================================

import os
import random
import numpy as np
import pickle
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG
# ============================================================
DATA_PATH   = r"C:\Users\asus\Downloads\final_dataset\final_dataset"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
RANDOM_SEED = 42
FEATURE_SIZE= 226

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# SELECTED CLASSES
# ============================================================
SELECTED_CLASSES = {
    # Numbers
    "0","1","2","3","4","5","6","7","8","9",
    # Alphabets
    "A","B","C","D","E",
    # Greetings / basics
    "call","change","check","correct","day",
    "bad","big","cold","cool","dark","deaf","dirty","dry",
    "add","ago","alone","all","already","animal",
    "baby","ball","barely","buy","can","careful",
    "cat","catch","chat","child","city","class",
    "country","cow","cry","decide","drop",
    # Food & drink
    "apple","banana","candy","carrot","corn","Drink",
    "delicious","dog",
    # People
    "brother","cousin","daughter","doctor",
    # Actions
    "accept","again","allow","approve","arrive",
    "argue","analyze","before","because",
    "convince","crash","decorate","delay","dive",
    # Objects
    "backpack","balloon","bar","bed","bird",
    "black","blanket","computer",
    # Emotions
    "awful","cry",
    # Medical
    "accident","allergy",
}

print(f"\n{'='*60}")
print(f"  ISL Landmark Extraction")
print(f"{'='*60}")
print(f"  Data         : {DATA_PATH}")
print(f"  Classes      : {len(SELECTED_CLASSES)} selected")
print(f"  Feature size : {FEATURE_SIZE}")
print(f"{'='*60}\n")

# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================
mp_holistic = mp.solutions.holistic
holistic    = mp_holistic.Holistic(
    static_image_mode       = True,
    model_complexity        = 1,
    enable_segmentation     = False,
    refine_face_landmarks   = False,
    min_detection_confidence= 0.5
)

# ============================================================
# EXTRACT LANDMARKS — hands + upper body only, no face
# ============================================================
def extract_landmarks(results):
    features = []

    # Left hand — 21 × 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features += [lm.x, lm.y, lm.z]
    else:
        features += [0.0] * 63

    # Right hand — 21 × 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features += [lm.x, lm.y, lm.z]
    else:
        features += [0.0] * 63

    # Upper body pose only — first 25 × 4 = 100
    if results.pose_landmarks:
        for lm in list(results.pose_landmarks.landmark)[:25]:
            features += [lm.x, lm.y, lm.z, lm.visibility]
    else:
        features += [0.0] * 100

    return np.array(features, dtype=np.float32)

# ============================================================
# AUGMENT
# ============================================================
def augment_img(img):
    h, w = img.shape[:2]

    angle = random.uniform(-10, 10)
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w, h),
                            borderMode=cv2.BORDER_REFLECT_101)

    img   = img.astype(np.float32)
    img   = np.clip(img + random.uniform(-25, 25), 0, 255)
    img   = img.astype(np.uint8)

    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    return img

# ============================================================
# MAIN
# ============================================================
all_labels = sorted([
    f for f in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, f))
])

print(f"  Total folders found : {len(all_labels)}")
print(f"  Processing selected : {len(SELECTED_CLASSES)}\n")

features_list = []
labels_list   = []
skipped       = 0
no_landmark   = 0

for label_idx, label in enumerate(all_labels):

    # Skip non-selected classes
    if label not in SELECTED_CLASSES:
        continue

    folder = os.path.join(DATA_PATH, label)
    files  = [f for f in os.listdir(folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    random.shuffle(files)

    class_features = []

    for f in files:
        img_bgr = cv2.imread(os.path.join(folder, f))
        if img_bgr is None:
            skipped += 1
            continue

        # Original
        rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        feat    = extract_landmarks(results)
        if not np.all(feat == 0):
            class_features.append(feat)
        else:
            no_landmark += 1

        # 4 augmentations per image
        for _ in range(4):
            aug     = augment_img(img_bgr)
            rgb_aug = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            res_aug = holistic.process(rgb_aug)
            feat    = extract_landmarks(res_aug)
            if not np.all(feat == 0):
                class_features.append(feat)

    if not class_features:
        print(f"  ✗ '{label}': no landmarks detected — skipping")
        continue

    features_list.extend(class_features)
    labels_list.extend([label] * len(class_features))

    print(f"  ✓ '{label}': {len(files)} imgs → {len(class_features)} vectors")

holistic.close()

# ============================================================
# REPORT
# ============================================================
print(f"\n{'='*60}")
print(f"  Total vectors : {len(features_list)}")
print(f"  Skipped       : {skipped}")
print(f"  No landmark   : {no_landmark}")
print(f"  Classes       : {len(set(labels_list))}")
print(f"{'='*60}\n")

if not features_list:
    raise ValueError("No features extracted. Check DATA_PATH.")

# ============================================================
# ENCODE + SPLIT
# ============================================================
le             = LabelEncoder()
labels_encoded = le.fit_transform(labels_list)
features_arr   = np.array(features_list, dtype=np.float32)

print(f"  Classes : {len(le.classes_)}")
print(f"  Shape   : {features_arr.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    features_arr, labels_encoded,
    test_size    = 0.2,
    random_state = RANDOM_SEED,
    stratify     = labels_encoded
)

print(f"  Train   : {len(X_train)}")
print(f"  Test    : {len(X_test)}")

# ============================================================
# SAVE
# ============================================================
print("\nSaving...")

np.save(os.path.join(OUTPUT_PATH, "X_train_landmarks.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "X_test_landmarks.npy"),  X_test)
np.save(os.path.join(OUTPUT_PATH, "y_train.npy"),           y_train)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"),            y_test)

with open(os.path.join(OUTPUT_PATH, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\n  ✓ X_train_landmarks.npy → {X_train.shape}")
print(f"  ✓ X_test_landmarks.npy  → {X_test.shape}")
print(f"  ✓ y_train.npy           → {y_train.shape}")
print(f"  ✓ y_test.npy            → {y_test.shape}")
print(f"  ✓ label_encoder.pkl     → {len(le.classes_)} classes")
print(f"\n{'='*60}")
print(f"  Done! Now run: py -3.12 train.py")
print(f"{'='*60}\n")