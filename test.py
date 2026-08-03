# ============================================================
# diagnose.py — figures out WHY the model only predicts O or X
# Run this before retraining. It prints exactly what's wrong.
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
from collections import Counter

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "models")
DATA_PATH  = os.path.join(BASE_PATH, "datasets")

# ── Load model ───────────────────────────────────────────────
checkpoint  = torch.load(os.path.join(MODEL_PATH, "mobilenet_isl.pth"), map_location="cpu")
le          = checkpoint["le"]
num_classes = checkpoint["num_classes"]

model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes)
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = [str(c).upper() for c in le.classes_]
print(f"\n{'='*55}")
print(f"  Classes in model: {classes}")
print(f"  Total classes   : {num_classes}")
print(f"{'='*55}")

# ── TEST 1: Bias test — feed a plain grey frame ───────────────
# A well-trained model should be uncertain (low confidence) on noise.
# If it confidently predicts O or X on a grey frame → strong prior bias.
print("\n── TEST 1: Bias on blank input ──────────────────────────")
grey  = np.ones((224, 224, 3), dtype=np.uint8) * 128
grey_rgb = cv2.cvtColor(grey, cv2.COLOR_BGR2RGB)
t     = transform(grey_rgb).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(t), dim=1)[0].numpy()

top3_idx = np.argsort(probs)[::-1][:3]
print("  Prediction on a blank grey image (should be ~uniform):")
for i in top3_idx:
    print(f"    {classes[i]:>4}: {probs[i]*100:5.1f}%")

if probs[top3_idx[0]] > 0.5:
    print("  ⚠ STRONG BIAS — model is >50% confident on blank input.")
    print("    → Cause 1 (imbalance) or Cause 2 (saturation) is likely.")
else:
    print("  ✓ Reasonably uncertain on blank input.")

# ── TEST 2: Per-class dataset counts ─────────────────────────
print("\n── TEST 2: Sample counts per class ─────────────────────")
dataset_dirs = [
    os.path.join(DATA_PATH, "train"),
    os.path.join(DATA_PATH, "test"),
    os.path.join(DATA_PATH, "val"),
    os.path.join(BASE_PATH,  "data"),
    os.path.join(BASE_PATH,  "dataset"),
]

found_counts = {}
for d in dataset_dirs:
    if os.path.isdir(d):
        for cls in os.listdir(d):
            cls_dir = os.path.join(d, cls)
            if os.path.isdir(cls_dir):
                n = len([f for f in os.listdir(cls_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
                found_counts[cls.upper()] = found_counts.get(cls.upper(), 0) + n

if found_counts:
    total = sum(found_counts.values())
    sorted_counts = sorted(found_counts.items(), key=lambda x: -x[1])
    print(f"  {'Class':<8} {'Count':>7}  {'Share':>7}  Bar")
    print(f"  {'-'*45}")
    for cls, cnt in sorted_counts:
        share = cnt / total * 100
        bar   = "█" * int(share / 2)
        flag  = "  ← dominant" if share > 15 else ""
        print(f"  {cls:<8} {cnt:>7}  {share:>6.1f}%  {bar}{flag}")

    # Imbalance ratio
    max_cnt = sorted_counts[0][1]
    min_cnt = sorted_counts[-1][1]
    ratio   = max_cnt / max(min_cnt, 1)
    print(f"\n  Max/min ratio: {ratio:.1f}x")
    if ratio > 5:
        print("  ⚠ SEVERE IMBALANCE — retrain with class_weight='balanced'")
    elif ratio > 2:
        print("  ⚠ MODERATE IMBALANCE — consider augmenting minority classes")
    else:
        print("  ✓ Counts look balanced")
else:
    print("  Could not find dataset folder. Checked:")
    for d in dataset_dirs:
        print(f"    {d}")
    print("  Edit dataset_dirs in this script to point to your data.")

# ── TEST 3: Logit range ──────────────────────────────────────
# Very large logit spread causes softmax to collapse to one class.
print("\n── TEST 3: Logit saturation ─────────────────────────────")
with torch.no_grad():
    logits = model(t)[0].numpy()

print(f"  Logit range: {logits.min():.2f}  to  {logits.max():.2f}")
print(f"  Spread     : {logits.max() - logits.min():.2f}")
if logits.max() - logits.min() > 20:
    print("  ⚠ LARGE SPREAD — softmax is saturating.")
    print("    Top class gets ~100%, others get ~0%.")
    print("    → Add label smoothing (0.1) or reduce model capacity.")
else:
    print("  ✓ Logit spread looks reasonable.")

# ── TEST 4: Live webcam check (5 frames per sign) ────────────
print("\n── TEST 4: Quick webcam check ───────────────────────────")
print("  Hold each sign in front of the camera.")
print("  Press SPACE to capture, Q to quit.\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("  ✗ Could not open webcam — skipping.")
else:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # Simple centre crop for testing
        margin  = 80
        roi     = frame[margin:fh-margin, margin:fw-margin]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            t2    = transform(rgb_roi).unsqueeze(0)
            probs2 = torch.softmax(model(t2), dim=1)[0].numpy()

        top5   = np.argsort(probs2)[::-1][:5]

        # Overlay
        cv2.rectangle(frame, (margin, margin), (fw-margin, fh-margin), (0,255,0), 2)
        for rank, i in enumerate(top5):
            lbl  = f"{classes[i]}: {probs2[i]*100:.1f}%"
            cv2.putText(frame, lbl, (10, 30 + rank*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0) if rank==0 else (180,180,180), 2)

        cv2.putText(frame, "SPACE=print probs  Q=quit",
                    (10, fh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.imshow("Diagnose — hold a sign", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            print(f"\n  All class probabilities:")
            for i in np.argsort(probs2)[::-1]:
                bar = "█" * int(probs2[i] * 40)
                print(f"    {classes[i]:>4}: {probs2[i]*100:5.1f}%  {bar}")

    cap.release()
    cv2.destroyAllWindows()

print("\n── SUMMARY ──────────────────────────────────────────────")
print("  If Test 1 shows >50% confidence on blank → bias/saturation")
print("  If Test 2 shows >5x imbalance           → add class_weight")
print("  If Test 3 shows spread >20              → add label smoothing")
print("  If Test 4 shows only O/X in top-5       → retrain is needed")
print(f"{'='*55}\n")