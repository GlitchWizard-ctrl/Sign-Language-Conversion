import os
import sys
import pickle
import threading
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, send_from_directory

# ============================================================
# CONFIG & PATHS
# ============================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "datasets")
MODEL_PATH = os.path.join(BASE_PATH, "models")
STATIC_PATH = os.path.join(BASE_PATH, "static")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(STATIC_PATH, exist_ok=True)

RAW_DATA_FILE = os.path.join(DATASET_PATH, "raw_data.pkl")

# Initialize Flask
app = Flask(__name__, static_folder=STATIC_PATH, static_url_path="")

# PyTorch Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# SHARED STATE FOR THREADING & TRAINING STATUS
# ============================================================
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "last_loss": 0.0,
    "best_accuracy": 0.0,
    "logs": [],
    "error_message": None
}
training_lock = threading.Lock()

# Inference Model cache
model_cache = {
    "model": None,
    "le": None,
    "classes": [],
    "loaded": False
}
model_lock = threading.Lock()

# ============================================================
# MODEL DEFINITION (Matches train.py)
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
# UTILITIES
# ============================================================
def load_raw_data():
    if not os.path.exists(RAW_DATA_FILE):
        return []
    try:
        with open(RAW_DATA_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading raw data file: {e}")
        return []

def save_raw_data(data):
    try:
        with open(RAW_DATA_FILE, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving raw data file: {e}")
        return False

def augment_landmarks(feat, n=99):
    augmented = []
    for _ in range(n):
        noise = np.random.normal(0, 0.008, feat.shape).astype(np.float32)
        augmented.append(feat + noise)
    return augmented

def load_cached_model():
    global model_cache
    with model_lock:
        model_path = os.path.join(MODEL_PATH, "landmark_mlp.pth")
        le_path = os.path.join(DATASET_PATH, "label_encoder.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(le_path):
            model_cache["loaded"] = False
            return False
            
        try:
            # Load Label Encoder
            with open(le_path, "rb") as f:
                le = pickle.load(f)
            
            # Load Model checkpoint
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            
            model = LandmarkMLP(ckpt["feature_size"], ckpt["num_classes"]).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            
            model_cache["model"] = model
            model_cache["le"] = le
            model_cache["classes"] = [str(c) for c in le.classes_]
            model_cache["loaded"] = True
            return True
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            model_cache["loaded"] = False
            return False

# Load model cache on startup if exists
load_cached_model()

# ============================================================
# WEB ROUTES / STATIC ASSET CONTROLLER
# ============================================================
@app.route("/")
def serve_index():
    return send_from_directory(STATIC_PATH, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_PATH, path)

# ============================================================
# API ENDPOINTS
# ============================================================

# Mock login authentication
@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    
    if username == "admin" and password == "password":
        return jsonify({
            "success": True,
            "token": "bearer-mock-session-token-987654321",
            "message": "Authenticated successfully"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Invalid username or password"
        }), 401

# Status endpoint
@app.route("/api/status", methods=["GET"])
def api_status():
    raw_data = load_raw_data()
    raw_count = len(raw_data)
    
    # Calculate augmented count (1 raw sample -> 100 augmented)
    augmented_count = raw_count * 100
    
    model_trained = os.path.exists(os.path.join(MODEL_PATH, "landmark_mlp.pth"))
    
    # Fetch classes if available
    classes = model_cache["classes"] if model_cache["loaded"] else []
    if not classes and os.path.exists(os.path.join(DATASET_PATH, "label_encoder.pkl")):
        try:
            with open(os.path.join(DATASET_PATH, "label_encoder.pkl"), "rb") as f:
                le = pickle.load(f)
                classes = [str(c) for c in le.classes_]
        except:
            pass

    return jsonify({
        "raw_samples": raw_count,
        "augmented_samples": augmented_count,
        "model_trained": model_trained,
        "classes": classes,
        "training_state": training_state
    })

# Record sample endpoint
@app.route("/api/record", methods=["POST"])
def api_record():
    data = request.json or {}
    sign = data.get("sign")
    features = data.get("features")
    
    if not sign or not features or len(features) != 126:
        return jsonify({
            "success": False,
            "message": "Invalid arguments. Sign label and 126 landmarks features required."
        }), 400
        
    raw_data = load_raw_data()
    raw_data.append({
        "features": features,
        "label": sign
    })
    
    if save_raw_data(raw_data):
        return jsonify({
            "success": True,
            "message": f"Recorded sample for sign {sign} successfully."
        })
    else:
        return jsonify({
            "success": False,
            "message": "Error saving sample to dataset."
        }), 500

# Predict inference endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not model_cache["loaded"]:
        # Try reloading cache first
        if not load_cached_model():
            return jsonify({
                "success": False,
                "message": "Model not loaded. Please train the model first."
            }), 400
            
    data = request.json or {}
    features = data.get("features")
    
    if not features or len(features) != 126:
        return jsonify({
            "success": False,
            "message": "Invalid features array. Length must be 126."
        }), 400
        
    try:
        # Run through model
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model_cache["model"](feat_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = output.argmax(1).item()
            
        pred_class = model_cache["le"].inverse_transform([pred_idx])[0]
        confidence = float(probabilities[pred_idx])
        
        # Build probability map for visual chart representation
        prob_map = {}
        for cls_name, prob in zip(model_cache["le"].classes_, probabilities):
            prob_map[str(cls_name)] = float(prob)
            
        return jsonify({
            "success": True,
            "predicted_class": str(pred_class),
            "confidence": confidence,
            "probabilities": prob_map
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Inference engine failure: {str(e)}"
        }), 500

# Train model trigger endpoint
@app.route("/api/train", methods=["POST"])
def api_train():
    global training_state
    
    with training_lock:
        if training_state["is_training"]:
            return jsonify({
                "success": False,
                "message": "Training is already in progress."
            }), 400
            
        # Get raw data
        raw_data = load_raw_data()
        if not raw_data:
            return jsonify({
                "success": False,
                "message": "No training data found. Please record samples first."
            }), 400
            
        # Verify classes count (need at least 2 distinct classes)
        unique_classes = set(item["label"] for item in raw_data)
        if len(unique_classes) < 2:
            return jsonify({
                "success": False,
                "message": "At least 2 distinct sign classes must be recorded to train the model."
            }), 400
            
        # Set config parameters
        config = request.json or {}
        epochs = config.get("epochs", 200)
        batch_size = config.get("batch_size", 32)
        lr = config.get("lr", 0.001)
        
        # Reset training logs
        training_state["is_training"] = True
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = epochs
        training_state["last_loss"] = 0.0
        training_state["best_accuracy"] = 0.0
        training_state["logs"] = ["Initializing dataset preprocessing..."]
        training_state["error_message"] = None
        
        # Start training worker thread
        thread = threading.Thread(target=training_worker, args=(raw_data, epochs, batch_size, lr))
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Training thread started successfully."
        })

# ============================================================
# BACKGROUND TRAINING WORKER
# ============================================================
def training_worker(raw_data, epochs, batch_size, lr_rate):
    global training_state
    
    try:
        def log(msg):
            print(msg)
            training_state["logs"].append(msg)
            # Limit log size
            if len(training_state["logs"]) > 1000:
                training_state["logs"].pop(0)

        log("=" * 45)
        log("  Starting Training Pipeline")
        log("=" * 45)
        log(f"Raw data samples: {len(raw_data)}")
        
        # 1. Augment landmarks (1 raw sample -> 100 samples)
        log("Augmenting data (100x noise expansion)...")
        all_features = []
        all_labels = []
        
        for item in raw_data:
            feat = np.array(item["features"], dtype=np.float32)
            lbl = item["label"]
            
            all_features.append(feat)
            all_labels.append(lbl)
            
            # 99 augmented
            for aug in augment_landmarks(feat, n=99):
                all_features.append(aug)
                all_labels.append(lbl)
                
        log(f"Total samples after augmentation: {len(all_features)}")
        
        # 2. Split train/test
        log("Splitting dataset (80% train, 20% test)...")
        le = LabelEncoder()
        labels_encoded = le.fit_transform(all_labels)
        features_arr = np.array(all_features, dtype=np.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_arr, labels_encoded,
            test_size=0.2,
            random_state=42,
            stratify=labels_encoded
        )
        
        # Save preprocessed files for record compatibility
        np.save(os.path.join(DATASET_PATH, "X_train_landmarks.npy"), X_train)
        np.save(os.path.join(DATASET_PATH, "X_test_landmarks.npy"), X_test)
        np.save(os.path.join(DATASET_PATH, "y_train.npy"), y_train)
        np.save(os.path.join(DATASET_PATH, "y_test.npy"), y_test)
        
        with open(os.path.join(DATASET_PATH, "label_encoder.pkl"), "wb") as f:
            pickle.dump(le, f)
            
        num_classes = len(le.classes_)
        feature_size = X_train.shape[1]
        
        log(f"Features size: {feature_size}")
        log(f"Total Classes: {num_classes} -> {list(le.classes_)}")
        
        # 3. Create datasets and loader
        X_train_t = torch.tensor(X_train).float()
        X_test_t = torch.tensor(X_test).float()
        y_train_t = torch.tensor(y_train).long()
        y_test_t = torch.tensor(y_test).long()
        
        # Weighted sampler for potential imbalance
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        # 4. Initialize model
        model = LandmarkMLP(feature_size, num_classes).to(device)
        log(f"Model params count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        best_acc = 0.0
        best_path = os.path.join(MODEL_PATH, "landmark_mlp.pth")
        
        log(f"\nTraining for {epochs} epochs on device: {device.type.upper()}")
        log(f"  {'Ep':>5}  {'Loss':>8}  {'Train%':>7}  {'Val%':>7}  Note")
        log(f"  {'-'*45}")
        
        # 5. Training loop
        for epoch in range(1, epochs + 1):
            model.train()
            t_loss = 0.0
            t_corr = 0
            t_tot = 0
            
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                t_loss += loss.item()
                t_corr += (out.argmax(1) == y).sum().item()
                t_tot += y.size(0)
                
            train_acc = t_corr / t_tot * 100
            
            # Validation
            model.eval()
            v_corr = 0
            v_tot = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    preds = out.argmax(1)
                    v_corr += (preds == y).sum().item()
                    v_tot += y.size(0)
                    
            val_acc = v_corr / v_tot * 100
            scheduler.step()
            
            note = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "feature_size": feature_size,
                    "le": le
                }, best_path)
                note = f"★ best {val_acc:.2f}%"
                
            # Update shared state
            training_state["current_epoch"] = epoch
            training_state["last_loss"] = t_loss / len(train_loader)
            training_state["best_accuracy"] = best_acc
            
            if epoch % 20 == 0 or epoch == 1 or note:
                log(f"  {epoch:>5}  {t_loss/len(train_loader):>8.4f}  {train_acc:>7.2f}  {val_acc:>7.2f}  {note}")
                
        log(f"\nTraining Complete! Best Accuracy achieved: {best_acc:.2f}%")
        log(f"Model saved locally at: models/landmark_mlp.pth")
        log("=" * 45)
        
        # Load the newly trained model into the inference cache
        load_cached_model()
        
    except Exception as e:
        training_state["error_message"] = str(e)
        print(f"Error during training execution thread: {e}")
    finally:
        with training_lock:
            training_state["is_training"] = False

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    # Host on localhost, port 5000
    print("\n" + "="*55)
    print("  ISL Sign Platform Server Initialization")
    print("="*55)
    print(f"  Local link : http://127.0.0.1:5000")
    print(f"  Device     : {device.type.upper()}")
    print("="*55 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
