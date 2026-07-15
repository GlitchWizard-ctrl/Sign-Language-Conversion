import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import Counter

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join("models", "best_model.pkl")
LABEL_PATH = os.path.join("datasets", "label_encoder.pkl")

# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(results):

    left = [0.0] * 63
    right = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_lm, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            coords = []

            for lm in hand_lm.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            label = handedness.classification[0].label

            if label == "Left":
                left = coords
            else:
                right = coords

    return np.array(left + right, dtype=np.float32)

# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

history = []

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction_text = "No Hand"

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

        feature = extract_landmarks(results)

        if np.any(feature):

            prediction = model.predict(feature.reshape(1, -1))[0]

            word = label_encoder.inverse_transform([prediction])[0]

            history.append(word)

            if len(history) > 10:
                history.pop(0)

            prediction_text = Counter(history).most_common(1)[0][0]

    cv2.rectangle(frame, (0, 0), (650, 70), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "Prediction : " + prediction_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ISL Prediction", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()