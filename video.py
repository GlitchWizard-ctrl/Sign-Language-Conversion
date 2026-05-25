# ============================================================
# video.py — Real-time ISL Recognition
# MediaPipe Hands (both hands) + MLP
# ============================================================

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import threading
import time
import pickle
from collections import Counter, deque

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("Run: pip install mediapipe")

try:
    import pyttsx3
except ImportError:
    raise ImportError("Run: pip install pyttsx3")

# ============================================================
# CONFIG
# ============================================================
BASE_PATH            = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH           = os.path.join(BASE_PATH, "models")
CONFIDENCE_THRESHOLD = 60
SMOOTHING_WINDOW     = 12
SIGN_HOLD_SECONDS    = 1.5

# ============================================================
# TTS
# ============================================================
def speak(text):
    def _run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# ============================================================
# MODEL
# ============================================================
class LandmarkMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# LOAD MODEL
# ============================================================
print("\n" + "="*55)
print("  ISL Recognition — Both Hands MLP")
print("="*55)

model_file = os.path.join(MODEL_PATH, "landmark_mlp.pth")
if not os.path.exists(model_file):
    raise FileNotFoundError("Run train.py first.")

device = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")
ckpt   = torch.load(model_file, map_location=device,
                     weights_only=False)

le           = ckpt["le"]
num_classes  = ckpt["num_classes"]
feature_size = ckpt["feature_size"]

model = LandmarkMLP(feature_size, num_classes).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"  ✓ Model: {num_classes} classes")
print(f"  ✓ Device: {device}")
print(f"  ✓ Classes: {list(le.classes_)}")

# ============================================================
# MEDIAPIPE HANDS
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
# EXTRACT LANDMARKS — matches record.py exactly
# ============================================================
def extract_landmarks(results):
    left  = [0.0] * 63
    right = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label  = handedness.classification[0].label
            coords = []
            for lm in hand_lm.landmark:
                coords += [lm.x, lm.y, lm.z]
            if label == "Left":
                left = coords
            else:
                right = coords

    return np.array(left + right, dtype=np.float32)

# ============================================================
# PREDICT
# ============================================================
def predict(features):
    tensor = torch.tensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(tensor)
        proba = torch.softmax(out, dim=1)[0].cpu().numpy()

    top3_idx = np.argsort(proba)[::-1][:3]
    top3     = [(le.inverse_transform([i])[0],
                  float(proba[i]) * 100)
                for i in top3_idx]
    return top3[0][0], top3[0][1], top3

# ============================================================
# SMOOTHER
# ============================================================
class Smoother:
    def __init__(self, window=SMOOTHING_WINDOW):
        self.labels = deque(maxlen=window)
        self.confs  = deque(maxlen=window)

    def update(self, label, conf):
        self.labels.append(label)
        self.confs.append(conf)

    def get(self):
        if not self.labels:
            return "", 0.0
        voted = Counter(self.labels).most_common(1)[0][0]
        avg   = np.mean([c for l, c in
                         zip(self.labels, self.confs)
                         if l == voted])
        return voted, avg

    def clear(self):
        self.labels.clear()
        self.confs.clear()

# ============================================================
# DRAW UI
# ============================================================
def draw_ui(frame, label, confidence, top3,
            sentence, hold_frac, fh, fw,
            left_detected, right_detected):

    # Sentence bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 58), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, "SENTENCE", (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (100, 100, 100), 1)
    sent  = " ".join(sentence) if sentence else "..."
    max_c = (fw - 20) // 15
    if len(sent) > max_c:
        sent = "..." + sent[-(max_c-3):]
    cv2.putText(frame, sent, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2)

    # Hand status top right
    lcolor = (0, 220, 80) if left_detected  else (60, 60, 60)
    rcolor = (0, 220, 80) if right_detected else (60, 60, 60)
    cv2.putText(frame, "L", (fw-100, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, lcolor, 2)
    cv2.putText(frame, "R", (fw-55,  35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, rcolor, 2)

    # Prediction badge
    if label and confidence >= CONFIDENCE_THRESHOLD:
        color = (0, 200, 80)
        cv2.rectangle(frame, (0, 68), (300, 172), color, -1)
        fs = 2.5 if len(label) <= 5 else 1.5
        cv2.putText(frame, label.upper(), (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"{confidence:.0f}%",
                    (10, 192),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (180, 255, 180), 2)

        # Hold bar
        cv2.rectangle(frame, (0, 198), (300, 212),
                      (40, 40, 40), -1)
        fill = int(300 * hold_frac)
        if fill > 0:
            cv2.rectangle(frame, (0, 198), (fill, 212),
                          (0, 255, 120), -1)
        msg = ("Added!" if hold_frac >= 1.0
               else "Hold to auto-add...")
        cv2.putText(frame, msg, (5, 228),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 120, 120), 1)
    else:
        cv2.rectangle(frame, (0, 68), (300, 172),
                      (35, 35, 35), -1)
        cv2.putText(frame, "No sign",
                    (10, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (70, 70, 70), 2)

    # Top 3 panel
    if top3:
        px, py = fw-250, 68
        ov2 = frame.copy()
        cv2.rectangle(ov2, (px, py),
                      (fw-5, py+155), (18, 18, 18), -1)
        cv2.addWeighted(ov2, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (px, py),
                      (fw-5, py+155), (55, 55, 55), 1)
        cv2.putText(frame, "TOP 3",
                    (px+8, py+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (110, 110, 110), 1)

        colors = [(0,220,120), (0,190,230), (190,130,255)]
        for i, (lbl, conf) in enumerate(top3):
            ry  = py + 28 + i*42
            col = colors[i]
            cv2.putText(frame, lbl.upper(),
                        (px+8, ry+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        col, 2)
            cv2.putText(frame, f"{conf:.0f}%",
                        (fw-58, ry+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        col, 1)
            bar_max = 220
            bar_f   = int(bar_max * min(conf, 100)/100)
            cv2.rectangle(frame,
                          (px+8, ry+24),
                          (px+8+bar_max, ry+34),
                          (50, 50, 50), -1)
            if bar_f > 0:
                cv2.rectangle(frame,
                              (px+8, ry+24),
                              (px+8+bar_f, ry+34),
                              col, -1)

    # Controls bar
    ov3 = frame.copy()
    cv2.rectangle(ov3, (0, fh-38), (fw, fh),
                  (10, 10, 10), -1)
    cv2.addWeighted(ov3, 0.9, frame, 0.1, 0, frame)
    cv2.putText(frame,
                "SPACE=add  ENTER=speak  BKSP=undo  "
                "C=clear  S=speak  Q=quit",
                (8, fh-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (90, 90, 90), 1)

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

print("\n  ✓ Webcam ready")
print("  SPACE     → add word")
print("  ENTER     → speak sentence")
print("  BACKSPACE → remove last word")
print("  C         → clear")
print("  S         → speak")
print("  Q         → quit")
print("  (Hold sign 1.5s to auto-add)\n")

smoother    = Smoother()
sentence    = []
frame_count = 0
cur_label   = ""
cur_conf    = 0.0
cur_top3    = []
hold_start  = None
hold_added  = False
fps_buf     = deque(maxlen=20)
prev_time   = time.time()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    now = time.time()
    fps_buf.append(1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_detected  = False
    right_detected = False
    hand_detected  = False

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_detected = True
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label
            if label == "Left":
                left_detected  = True
            else:
                right_detected = True

            mp_drawing.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

            # Hand label above wrist
            wrist = hand_lm.landmark[0]
            wx    = int(wrist.x * fw)
            wy    = int(wrist.y * fh)
            cv2.putText(frame, label,
                        (wx-20, wy-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 220, 80), 2)

    # Predict every 2 frames
    if frame_count % 2 == 0:
        features = extract_landmarks(results)
        if not np.all(features == 0):
            try:
                label, conf, top3 = predict(features)
                smoother.update(label, conf)
                cur_top3 = top3
            except Exception as e:
                print(f"  Predict error: {e}")
        else:
            smoother.clear()

    cur_label, cur_conf = smoother.get()

    # Hold-to-add
    hold_frac = 0.0
    if cur_label and cur_conf >= CONFIDENCE_THRESHOLD and hand_detected:
        if hold_start is None:
            hold_start = now
            hold_added = False
        elapsed   = now - hold_start
        hold_frac = min(elapsed / SIGN_HOLD_SECONDS, 1.0)
        if elapsed >= SIGN_HOLD_SECONDS and not hold_added:
            sentence.append(cur_label)
            print(f"  Auto-added: '{cur_label}' → {sentence}")
            speak(cur_label)
            hold_added = True
            hold_start = None
            smoother.clear()
    else:
        hold_start = None
        hold_added = False

    frame_count += 1

    frame = draw_ui(frame, cur_label, cur_conf, cur_top3,
                    sentence, hold_frac, fh, fw,
                    left_detected, right_detected)

    cv2.putText(frame, f"FPS {np.mean(fps_buf):.0f}",
                (fw-85, fh-48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (60, 60, 60), 1)

    cv2.imshow("ISL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:
        if cur_label and cur_conf >= CONFIDENCE_THRESHOLD:
            sentence.append(cur_label)
            print(f"  Added: '{cur_label}' → {sentence}")
            speak(cur_label)
            smoother.clear()
            hold_start = None
        else:
            print(f"  ✗ Low confidence ({cur_conf:.0f}%)")
    elif key == 13:
        full = " ".join(sentence)
        if full:
            speak(full)
            print(f"  Speaking: '{full}'")
    elif key == 8:
        if sentence:
            print(f"  Removed: '{sentence.pop()}'")
    elif key == ord('c'):
        sentence = []
        smoother.clear()
        print("  Cleared")
    elif key == ord('s'):
        full = " ".join(sentence)
        if full:
            speak(full)

cap.release()
cv2.destroyAllWindows()
hands.close()

if sentence:
    final = " ".join(sentence)
    print(f"\n  Final: '{final}'")
    speak(final)
print("  Done.")