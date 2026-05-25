# ============================================================
# train.py — MLP on hand landmarks (both hands)
# ============================================================

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report

# ============================================================
# CONFIG
# ============================================================
BASE_PATH    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "datasets")
MODEL_PATH   = os.path.join(BASE_PATH, "models")
os.makedirs(MODEL_PATH, exist_ok=True)

EPOCHS        = 200
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}")
print(f"  ISL Landmark Trainer")
print(f"{'='*55}")
print(f"  Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ============================================================
# LOAD
# ============================================================
print("\nLoading...")
try:
    X_train = np.load(os.path.join(DATASET_PATH,
                                    "X_train_landmarks.npy"))
    X_test  = np.load(os.path.join(DATASET_PATH,
                                    "X_test_landmarks.npy"))
    y_train = np.load(os.path.join(DATASET_PATH, "y_train.npy"))
    y_test  = np.load(os.path.join(DATASET_PATH, "y_test.npy"))
except FileNotFoundError:
    raise FileNotFoundError("Run record.py first.")

with open(os.path.join(DATASET_PATH, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

num_classes  = len(le.classes_)
classes      = [str(c) for c in le.classes_]
feature_size = X_train.shape[1]

print(f"  Train        : {len(X_train)}")
print(f"  Test         : {len(X_test)}")
print(f"  Classes      : {num_classes} → {classes}")
print(f"  Feature size : {feature_size}")

# ============================================================
# TENSORS
# ============================================================
X_train_t = torch.tensor(X_train).float()
X_test_t  = torch.tensor(X_test).float()
y_train_t = torch.tensor(y_train).long()
y_test_t  = torch.tensor(y_test).long()

# ============================================================
# WEIGHTED SAMPLER
# ============================================================
class_counts   = np.bincount(y_train.astype(int))
class_weights  = 1.0 / np.maximum(class_counts, 1)
sample_weights = class_weights[y_train.astype(int)]
sampler        = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights).float(),
    num_samples = len(sample_weights),
    replacement = True
)

train_ds     = TensorDataset(X_train_t, y_train_t)
test_ds      = TensorDataset(X_test_t,  y_test_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

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

model = LandmarkMLP(feature_size, num_classes).to(device)
total = sum(p.numel() for p in model.parameters()
            if p.requires_grad)
print(f"\n  ✓ MLP — {total:,} params")

# ============================================================
# LOSS + OPTIMIZER + SCHEDULER
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LEARNING_RATE,
                               weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

# ============================================================
# TRAINING LOOP
# ============================================================
print(f"\nTraining {EPOCHS} epochs...\n")
print(f"  {'Ep':>5}  {'Loss':>8}  {'Train%':>7}  "
      f"{'Val%':>7}  Note")
print(f"  {'-'*45}")

best_acc  = 0.0
best_path = os.path.join(MODEL_PATH, "landmark_mlp.pth")

for epoch in range(1, EPOCHS + 1):

    model.train()
    t_loss = 0.0
    t_corr = 0
    t_tot  = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_corr += (out.argmax(1) == y).sum().item()
        t_tot  += y.size(0)

    train_acc = t_corr / t_tot * 100

    model.eval()
    v_corr    = 0
    v_tot     = 0
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y  = X.to(device), y.to(device)
            out   = model(X)
            preds = out.argmax(1)
            v_corr += (preds == y).sum().item()
            v_tot  += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    val_acc = v_corr / v_tot * 100
    scheduler.step()

    note = ""
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state" : model.state_dict(),
            "num_classes" : num_classes,
            "feature_size": feature_size,
            "le"          : le
        }, best_path)
        note = f"★ best {val_acc:.2f}%"

    if epoch % 20 == 0 or epoch == 1 or note:
        print(f"  {epoch:>5}  "
              f"{t_loss/len(train_loader):>8.4f}  "
              f"{train_acc:>7.2f}  {val_acc:>7.2f}  {note}")

print(f"\n{'='*55}")
print(f"  Best accuracy : {best_acc:.2f}%")
print(f"  Model saved   : models/landmark_mlp.pth")
print(f"{'='*55}\n")

# Final report
ckpt = torch.load(best_path, map_location=device,
                  weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X, y in test_loader:
        out = model(X.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_true.extend(y.numpy())
print(classification_report(all_true, all_preds,
                             target_names=classes,
                             zero_division=0))