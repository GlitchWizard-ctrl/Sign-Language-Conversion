# ============================================================
# record.py — Record ISL signs via webcam
# Uses MediaPipe Hands (both hands), 2 samples → 200 per class
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
SIGNS_TO_RECORD    = [
    "hello", "hi", "good_morning", "good_afternoon", "good_evening",
    "goodbye", "thank_you", "please", "sorry", "welcome",

    "how_are_you", "what_are_you_doing", "where_are_you_going",
    "what_is_your_name", "how_old_are_you", "can_you_help_me",
    "do_you_understand", "are_you_okay", "what_happened", "why",

    "eat", "drink", "sleep", "study", "work",
    "come", "go", "sit", "stand", "walk",

    "yes", "no", "water", "food", "medicine",
    "hospital", "home", "school", "friend", "family",

    "i_need_help", "call_the_doctor", "call_the_police",
    "i_am_fine", "i_am_hungry", "i_am_thirsty",
    "nice_to_meet_you", "see_you_later", "take_care", "i_love_you"
]
SAMPLES_PER_SIGN   = 2
AUGMENT_PER_SAMPLE = 99   # 2 × 100 = 200 per class

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets"
)
os.makedirs(DATASET_PATH, exist_ok=True)

# ============================================================
# MEDIAPIPE HANDS — dedicated, detects both hands reliably
# ============================================================
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode       = False,
    max_num_hands           = 2,
    model_complexity        = 1,
    min_detection_confidence= 0.7,
    min_tracking_confidence = 0.6
)

# ============================================================
# EXTRACT LANDMARKS
# Left hand: 21×3=63, Right hand: 21×3=63 → total 126
# We identify left/right from handedness label
# ============================================================
FEATURE_SIZE = 126

def extract_landmarks(results):
    left  = [0.0] * 63
    right = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label
            coords = []
            for lm in hand_lm.landmark:
                coords += [lm.x, lm.y, lm.z]

            if label == "Left":
                left = coords
            else:
                right = coords

    return np.array(left + right, dtype=np.float32)

# ============================================================
# AUGMENT LANDMARKS
# ============================================================
def augment_landmarks(feat, n=99):
    augmented = []
    for _ in range(n):
        noise = np.random.normal(
            0, 0.008, feat.shape
        ).astype(np.float32)
        augmented.append(feat + noise)
    return augmented

# ============================================================
# DRAW UI
# ============================================================
def draw_ui(frame, sign, sign_idx, total_signs,
            collected, total, fh, fw,
            left_detected, right_detected):

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 80), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.putText(frame,
                f"[{sign_idx+1}/{total_signs}]  "
                f"Sign: {sign.upper()}",
                (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 2)

    # Hand detection status
    lcolor = (0, 220, 80)  if left_detected  else (60, 60, 60)
    rcolor = (0, 220, 80)  if right_detected else (60, 60, 60)
    cv2.putText(frame, "L", (fw-110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, lcolor, 2)
    cv2.putText(frame, "R", (fw-70,  35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, rcolor, 2)
    cv2.putText(frame, "hands", (fw-120, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (120, 120, 120), 1)

    # Sample dots
    for i in range(total):
        color = (0, 220, 80) if i < collected else (60, 60, 60)
        cx    = 20 + i * 50
        cv2.circle(frame, (cx, 105), 18, color, -1)
        cv2.putText(frame, str(i+1),
                    (cx-8, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 2)

    cv2.putText(frame,
                f"{collected}/{total} captured",
                (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (180, 180, 180), 1)

    # Controls
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, fh-38), (fw, fh),
                  (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.9, frame, 0.1, 0, frame)
    cv2.putText(frame,
                "SPACE = capture    N = skip    Q = quit & save",
                (10, fh-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 120, 120), 1)

    return frame

# ============================================================
# MAIN
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("✗ Could not open webcam")
    exit()

print("\n" + "="*55)
print("  ISL Recorder — Both Hands")
print("="*55)
print(f"  Signs    : {SIGNS_TO_RECORD}")
print(f"  Per sign : {SAMPLES_PER_SIGN} real → "
      f"{SAMPLES_PER_SIGN*(AUGMENT_PER_SAMPLE+1)} augmented")
print("="*55)
print("\n  SPACE = capture sample")
print("  N     = skip sign")
print("  Q     = quit and save\n")

all_features = []
all_labels   = []
quit_early   = False

for sign_idx, sign in enumerate(SIGNS_TO_RECORD):
    collected = 0
    print(f"\n  [{sign_idx+1}/{len(SIGNS_TO_RECORD)}] "
          f"'{sign}' — press SPACE to capture")

    while collected < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_detected  = False
        right_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = handedness.classification[0].label
                color = (0, 220, 80)

                if label == "Left":
                    left_detected = True
                else:
                    right_detected = True

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                # Label above wrist
                wrist = hand_lm.landmark[0]
                wx    = int(wrist.x * fw)
                wy    = int(wrist.y * fh)
                cv2.putText(frame, label,
                            (wx - 20, wy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color, 2)

        # No hand warning
        if not left_detected and not right_detected:
            cv2.putText(frame, "SHOW YOUR HAND",
                        (fw//2 - 180, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 0, 255), 3)

        frame = draw_ui(frame, sign, sign_idx,
                        len(SIGNS_TO_RECORD),
                        collected, SAMPLES_PER_SIGN,
                        fh, fw,
                        left_detected, right_detected)

        cv2.imshow("ISL Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            quit_early = True
            break

        elif key == 32:  # SPACE — capture
            if left_detected or right_detected:
                feat = extract_landmarks(results)
                if not np.all(feat == 0):
                    all_features.append(feat)
                    all_labels.append(sign)
                    collected += 1
                    print(f"  ✓ Captured {collected}/{SAMPLES_PER_SIGN}")

                    # Green flash
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (fw, fh),
                                  (0, 255, 0), 25)
                    cv2.imshow("ISL Recorder", flash)
                    cv2.waitKey(300)
                else:
                    print("  ✗ No landmarks — try again")
            else:
                print("  ✗ No hand detected")

        elif key == ord('n'):
            print(f"  Skipped '{sign}'")
            break

    if quit_early:
        break

    if collected >= SAMPLES_PER_SIGN:
        print(f"  ✓ '{sign}' complete!")

cap.release()
cv2.destroyAllWindows()
hands.close()

# ============================================================
# AUGMENT + SAVE
# ============================================================
if not all_features:
    print("  No data recorded.")
    exit()

print(f"\n{'='*55}")
print(f"  Raw samples : {len(all_features)}")
print(f"  Augmenting {AUGMENT_PER_SAMPLE+1}x...")

aug_features = []
aug_labels   = []

for feat, label in zip(all_features, all_labels):
    aug_features.append(feat)
    aug_labels.append(label)
    for aug in augment_landmarks(feat, n=AUGMENT_PER_SAMPLE):
        aug_features.append(aug)
        aug_labels.append(label)

print(f"  Total : {len(aug_features)}")

counts = Counter(aug_labels)
print(f"\n  Per class:")
for lbl, cnt in sorted(counts.items()):
    bar = "█" * (cnt // 10)
    print(f"    {lbl:>5}: {cnt}  {bar}")

le             = LabelEncoder()
labels_encoded = le.fit_transform(aug_labels)
features_arr   = np.array(aug_features, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    features_arr, labels_encoded,
    test_size    = 0.2,
    random_state = 42,
    stratify     = labels_encoded
)

np.save(os.path.join(DATASET_PATH,
                     "X_train_landmarks.npy"), X_train)
np.save(os.path.join(DATASET_PATH,
                     "X_test_landmarks.npy"),  X_test)
np.save(os.path.join(DATASET_PATH,
                     "y_train.npy"),           y_train)
np.save(os.path.join(DATASET_PATH,
                     "y_test.npy"),            y_test)

with open(os.path.join(DATASET_PATH,
                       "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\n  ✓ Train   : {X_train.shape}")
print(f"  ✓ Test    : {X_test.shape}")
print(f"  ✓ Classes : {list(le.classes_)}")
print(f"\n{'='*55}")
print(f"  Done! Now run: py -3.12 train.py")
print(f"{'='*55}\n")