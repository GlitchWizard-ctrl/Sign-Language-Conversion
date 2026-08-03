import os
import time
import threading
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_socketio import SocketIO, emit, join_room, leave_room

import db

# ============================================================
# PATHS & FLASK SETUP
# ============================================================
BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_PATH, "models")
STATIC_PATH = os.path.join(BASE_PATH, "static")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(STATIC_PATH, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_PATH, static_url_path="")
# Choose a compatible async_mode for Flask-SocketIO at runtime to avoid
# "Invalid async_mode specified" when optional dependencies are missing.
import importlib
async_mode = None
if importlib.util.find_spec('eventlet'):
    async_mode = 'eventlet'
elif importlib.util.find_spec('gevent'):
    async_mode = 'gevent'
else:
    async_mode = 'threading'
print(f"[startup] Using Socket.IO async_mode={async_mode}")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db.init_db()

# ============================================================
# TRAINING STATE  (runtime only, not persisted in DB)
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

# ============================================================
# MODEL CACHE  (in-memory; metadata comes from DB)
# ============================================================
model_cache = {"model": None, "classes": [], "loaded": False}
model_lock  = threading.Lock()


# ============================================================
# MLP MODEL
# ============================================================
class LandmarkMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256),        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# UTILITIES
# ============================================================
def augment_landmarks(feat, n=39):
    return [feat + np.random.normal(0, 0.008, feat.shape).astype(np.float32) for _ in range(n)]


def load_cached_model():
    global model_cache
    with model_lock:
        run = db.get_active_model_run()
        if not run or not os.path.exists(run["weights_path"]):
            model_cache["loaded"] = False
            return False
        try:
            ckpt  = torch.load(run["weights_path"], map_location=device, weights_only=False)
            model = LandmarkMLP(run["feature_size"], len(run["classes"])).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            model_cache.update({"model": model, "classes": run["classes"], "loaded": True})
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            model_cache["loaded"] = False
            return False


load_cached_model()

# ============================================================
# SOCKET.IO BACKGROUND STATE
# ============================================================
active_calls = {}
client_sessions = {}
call_lock = threading.Lock()


def get_username_from_token(token):
    return db.verify_session(token)


# ============================================================
# AUTH DECORATOR  (checks Authorization header against sessions table)
# ============================================================
def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not db.verify_session(token):
            return jsonify({"success": False, "message": "Unauthorized. Please log in again."}), 401
        return f(*args, **kwargs)
    return wrapper


# ============================================================
# STATIC ROUTES
# ============================================================
@app.route("/")
def serve_login():
    return send_from_directory(STATIC_PATH, "login.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_PATH, path)


# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.route("/api/register", methods=["POST"])
def api_register():
    data     = request.json or {}
    fullname = data.get("fullname", "").strip()
    email    = data.get("email", "").strip().lower()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not all([fullname, email, username, password]):
        return jsonify({"success": False, "message": "All fields are required."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400
    if len(username) < 3:
        return jsonify({"success": False, "message": "Username must be at least 3 characters."}), 400

    success, error = db.register_user(fullname, email, username, password)
    if success:
        return jsonify({"success": True, "message": "Account created successfully. You can now log in."})
    else:
        return jsonify({"success": False, "message": error}), 409


@app.route("/api/login", methods=["POST"])
def api_login():
    data     = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if username and password and db.verify_user(username, password):
        token     = db.create_session(username)
        user_info = db.get_user_info(username)
        return jsonify({
            "success":  True,
            "token":    token,
            "fullname": user_info["fullname"] if user_info else username,
            "role":     user_info["role"]     if user_info else "user",
            "message":  "Authenticated successfully"
        })
    return jsonify({"success": False, "message": "Invalid username or password."}), 401


@app.route("/api/logout", methods=["POST"])
@require_auth
def api_logout():
    db.delete_session(request.headers.get("Authorization"))
    return jsonify({"success": True, "message": "Logged out."})


# ============================================================
# CORE API ENDPOINTS
# ============================================================

@app.route("/api/status", methods=["GET"])
@require_auth
def api_status():
    raw_count = db.count_samples()
    run       = db.get_active_model_run()
    classes   = run["classes"] if run else []
    model_ok  = run is not None and os.path.exists(run["weights_path"])

    return jsonify({
        "raw_samples":       raw_count,
        "augmented_samples": raw_count * 100,
        "model_trained":     model_ok,
        "classes":           classes,
        "training_state":    training_state
    })


@app.route("/api/profile", methods=["GET"])
@require_auth
def api_profile():
    token = request.headers.get("Authorization")
    username = get_username_from_token(token)
    user = db.get_user_info(username) if username else None
    return jsonify({
        "success": True,
        "username": username,
        "fullname": user["fullname"] if user else username,
        "role": user["role"] if user else "user"
    })


@app.route("/api/profile", methods=["PUT"])
@require_auth
def api_profile_update():
    token = request.headers.get("Authorization")
    username = get_username_from_token(token)
    if not username:
        return jsonify({"success": False, "message": "Unauthorized."}), 401
    data = request.json or {}
    fullname = data.get("fullname")
    email = data.get("email")
    password = data.get("password")
    ok, err = db.update_user(username, fullname=fullname, email=email, password=password)
    if not ok:
        return jsonify({"success": False, "message": err}), 400
    user = db.get_user_info(username)
    return jsonify({"success": True, "message": "Profile updated.", "profile": user})


@app.route("/api/call-history", methods=["GET"])
@require_auth
def api_call_history():
    token = request.headers.get("Authorization")
    username = get_username_from_token(token)
    user = db.get_user_info(username) if username else None
    history = db.get_call_history(limit=100)
    if user and user.get("role") == "admin":
        return jsonify({"success": True, "history": history})

    filtered = [item for item in history if item["caller"] == username or item["callee"] == username]
    return jsonify({"success": True, "history": filtered})


# ============================================================
# SOCKET.IO EVENTS
# ============================================================

@socketio.on("connect")
def handle_connect():
    emit("connected", {"message": "Connected to signaling server."})


@socketio.on("join-room")
def handle_join_room(data):
    token = data.get("token")
    room_id = data.get("room_id")
    interpreter_mode = bool(data.get("interpreter_mode", False))

    username = get_username_from_token(token)
    if not username:
        emit("room-error", {"message": "Unauthorized."})
        return

    if not room_id:
        emit("room-error", {"message": "Room ID is required."})
        return

    join_room(room_id)
    with call_lock:
        call = active_calls.get(room_id, {
            "room_id": room_id,
            "participants": [],
            "interpreter_mode": interpreter_mode,
            "start_time": time.time()
        })

        if username not in call["participants"]:
            call["participants"].append(username)
        call["interpreter_mode"] = call["interpreter_mode"] or interpreter_mode
        active_calls[room_id] = call

    room_owner = call["participants"][0] if call["participants"] else username
    emit("room-joined", {
        "room_id": room_id,
        "participants": call["participants"],
        "interpreter_mode": call["interpreter_mode"],
        "room_owner": room_owner
    }, room=room_id)


@socketio.on("offer")
def handle_offer(data):
    target = data.get("target")
    room_id = data.get("room_id")
    if target and room_id:
        emit("offer", data, room=room_id, include_self=False)


@socketio.on("answer")
def handle_answer(data):
    target = data.get("target")
    room_id = data.get("room_id")
    if target and room_id:
        emit("answer", data, room=room_id, include_self=False)


@socketio.on("ice-candidate")
def handle_ice_candidate(data):
    room_id = data.get("room_id")
    if room_id:
        emit("ice-candidate", data, room=room_id, include_self=False)


@socketio.on("interpreter-toggle")
def handle_interpreter_toggle(data):
    room_id = data.get("room_id")
    enabled = bool(data.get("enabled", False))
    if room_id:
        with call_lock:
            call = active_calls.get(room_id)
            if call:
                call["interpreter_mode"] = enabled
                active_calls[room_id] = call
        emit("interpreter-changed", {"enabled": enabled}, room=room_id)


@socketio.on("camera-toggle")
def handle_camera_toggle(data):
    room_id = data.get("room_id")
    enabled = bool(data.get("enabled", False))
    # broadcast to other participants so they can update UI
    if room_id:
        emit("camera-changed", {"room_id": room_id, "enabled": enabled}, room=room_id, include_self=False)


@socketio.on("mic-toggle")
def handle_mic_toggle(data):
    room_id = data.get("room_id")
    enabled = bool(data.get("enabled", False))
    if room_id:
        emit("mic-changed", {"room_id": room_id, "enabled": enabled}, room=room_id, include_self=False)


@socketio.on("end-call")
def handle_end_call(data):
    token = data.get("token")
    room_id = data.get("room_id")
    username = get_username_from_token(token)
    if not username or not room_id:
        return

    leave_room(room_id)
    with call_lock:
        call = active_calls.pop(room_id, None)

    if call:
        participants = call.get("participants", [])
        start_time = call.get("start_time", time.time())
        end_time = time.time()
        duration = int(end_time - start_time)

        caller = participants[0] if len(participants) > 0 else username
        callee = participants[1] if len(participants) > 1 else username

        db.save_call_history(room_id, caller, callee, call.get("interpreter_mode", False),
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                             duration)

    emit("call-ended", {"room_id": room_id}, room=room_id)


@app.route("/api/record", methods=["POST"])
@require_auth
def api_record():
    data     = request.json or {}
    sign     = data.get("sign")
    features = data.get("features")

    if not sign or not features or len(features) != 126:
        return jsonify({"success": False,
                        "message": "Sign label and 126 landmark features are required."}), 400

    db.insert_sample(sign, features)
    return jsonify({"success": True, "message": f"Sample recorded for sign {sign}."})


@app.route("/api/predict", methods=["POST"])
@require_auth
def api_predict():
    if not model_cache["loaded"] and not load_cached_model():
        return jsonify({"success": False,
                        "message": "No trained model found. Please train the model first."}), 400

    data     = request.json or {}
    features = data.get("features")

    if not features or len(features) != 126:
        return jsonify({"success": False,
                        "message": "Features array must contain exactly 126 values."}), 400

    try:
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model_cache["model"](tensor)
            probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
            idx    = int(output.argmax(1).item())

        classes    = model_cache["classes"]
        pred_class = classes[idx]
        confidence = float(probs[idx])
        prob_map   = {cls: float(p) for cls, p in zip(classes, probs)}

        return jsonify({
            "success":       True,
            "predicted_class": str(pred_class),
            "confidence":    confidence,
            "probabilities": prob_map
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Inference error: {e}"}), 500


@app.route("/api/train", methods=["POST"])
@require_auth
def api_train():
    global training_state

    with training_lock:
        if training_state["is_training"]:
            return jsonify({"success": False, "message": "Training already in progress."}), 400

        samples = db.get_all_samples()
        if not samples:
            return jsonify({"success": False,
                            "message": "No training data found. Record samples first."}), 400

        if len(set(s["label"] for s in samples)) < 2:
            return jsonify({"success": False,
                            "message": "Record at least 2 different signs before training."}), 400

        cfg        = request.json or {}
        epochs     = cfg.get("epochs",     200)
        batch_size = cfg.get("batch_size",  32)
        lr         = cfg.get("lr",        0.001)

        training_state.update({
            "is_training":   True,
            "current_epoch": 0,
            "total_epochs":  epochs,
            "last_loss":     0.0,
            "best_accuracy": 0.0,
            "logs":          ["Initializing training pipeline..."],
            "error_message": None
        })

        threading.Thread(target=training_worker,
                         args=(samples, epochs, batch_size, lr)).start()

        return jsonify({"success": True, "message": "Training started."})


# ============================================================
# BACKGROUND TRAINING WORKER
# ============================================================
def training_worker(samples, epochs, batch_size, lr_rate):
    global training_state

    try:
        def log(msg):
            print(msg)
            training_state["logs"].append(msg)
            if len(training_state["logs"]) > 1000:
                training_state["logs"].pop(0)

        log("=" * 45)
        log("  ISL Training Pipeline")
        log("=" * 45)
        log(f"Samples from DB: {len(samples)}")

        classes      = sorted(set(item["label"] for item in samples))
        label_to_idx = {lbl: i for i, lbl in enumerate(classes)}

        log("Augmenting data (100x)...")
        all_feats, all_labels = [], []
        for item in samples:
            feat = np.array(item["features"], dtype=np.float32)
            idx  = label_to_idx[item["label"]]
            all_feats.append(feat);  all_labels.append(idx)
            for aug in augment_landmarks(feat):
                all_feats.append(aug); all_labels.append(idx)

        log(f"Total after augmentation: {len(all_feats)}")
        log("Splitting 80/20 train/test...")

        X = np.array(all_feats, dtype=np.float32)
        y = np.array(all_labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        num_classes  = len(classes)
        feature_size = X_train.shape[1]
        log(f"Classes: {num_classes} → {classes}")

        X_tr = torch.tensor(X_train).float()
        X_te = torch.tensor(X_test).float()
        y_tr = torch.tensor(y_train).long()
        y_te = torch.tensor(y_test).long()

        counts   = np.bincount(y_train.astype(int), minlength=num_classes)
        w_sample = (1.0 / np.maximum(counts, 1))[y_train.astype(int)]
        sampler  = WeightedRandomSampler(torch.tensor(w_sample).float(),
                                         len(w_sample), replacement=True)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, sampler=sampler)
        test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

        model     = LandmarkMLP(feature_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_acc        = 0.0
        final_loss      = 0.0
        weights_name    = f"landmark_mlp_{int(time.time())}.pth"
        best_path       = os.path.join(MODEL_PATH, weights_name)

        log(f"\nTraining {epochs} epochs on {device.type.upper()}")
        log(f"  {'Ep':>5}  {'Loss':>8}  {'Train%':>7}  {'Val%':>7}  Note")
        log(f"  {'-'*45}")

        for epoch in range(1, epochs + 1):
            model.train()
            t_loss = t_corr = t_tot = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out  = model(X)
                loss = criterion(out, y)
                loss.backward(); optimizer.step()
                t_loss += loss.item()
                t_corr += (out.argmax(1) == y).sum().item()
                t_tot  += y.size(0)

            train_acc = t_corr / t_tot * 100

            model.eval(); v_corr = v_tot = 0
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    v_corr += (model(X).argmax(1) == y).sum().item()
                    v_tot  += y.size(0)

            val_acc    = v_corr / v_tot * 100
            scheduler.step()
            final_loss = t_loss / len(train_loader)

            note = ""
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model_state": model.state_dict()}, best_path)
                note = f"★ best {val_acc:.2f}%"

            training_state["current_epoch"] = epoch
            training_state["last_loss"]     = final_loss
            training_state["best_accuracy"] = best_acc

            if epoch % 20 == 0 or epoch == 1 or note:
                log(f"  {epoch:>5}  {final_loss:>8.4f}  {train_acc:>7.2f}  {val_acc:>7.2f}  {note}")

        log(f"\nDone! Best accuracy: {best_acc:.2f}%")
        log(f"Weights saved: models/{weights_name}")
        log("=" * 45)

        db.save_model_run(epochs, batch_size, lr_rate, best_acc, final_loss,
                          feature_size, classes, best_path)
        load_cached_model()

    except Exception as e:
        training_state["error_message"] = str(e)
        print(f"Training error: {e}")
    finally:
        with training_lock:
            training_state["is_training"] = False


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "false").lower() in ("true", "1")

    print("\n" + "=" * 55)
    print("  ISL Sign Platform  (SQL backend)")
    print("=" * 55)
    print(f"  URL      : http://0.0.0.0:{port}")
    print(f"  Device   : {device.type.upper()}")
    print(f"  Database : {db.DB_PATH}")
    print("=" * 55 + "\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=debug_mode)
