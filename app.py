import os
import base64
import pickle
import re
import threading
import tempfile
from functools import wraps
from collections import Counter

import numpy as np
import cv2

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, join_room, leave_room, emit

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import db
import urllib.request
import shutil

USE_SERVER_MEDIAPIPE = True
try:
    import mediapipe as mp
    # Some build environments expose mediapipe subpackages differently
    if not hasattr(mp, 'solutions'):
        try:
            import mediapipe.python.solutions as _mp_py_solutions
            mp.solutions = _mp_py_solutions
        except Exception:
            try:
                from mediapipe.python.solutions import hands as _mp_hands_mod
                class _Shim:
                    pass
                mp.solutions = _Shim()
                mp.solutions.hands = _mp_hands_mod
            except Exception:
                pass
except Exception:
    # MediaPipe not available on this environment — that's OK if clients send precomputed features.
    mp = None
    USE_SERVER_MEDIAPIPE = False

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, "models")
DATASET_DIR = os.path.join(BASE_PATH, "datasets")
STATIC_PATH = os.path.join(BASE_PATH, "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(STATIC_PATH, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_PATH, static_url_path="")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# A model swap must be atomic: an active video call may be predicting while an
# administrator starts a new training run.
model_lock = threading.RLock()
training_lock = threading.Lock()

db.init_db()

# ------------------------------------------------------------------
# Sign model — loaded at startup if it exists, hot-reloaded after
# every in-app training run (no restart needed).
# ------------------------------------------------------------------
sign_model = None
label_encoder = None


def try_load_model():
    global sign_model, label_encoder
    try:
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        # If model missing but an external MODEL_URL is provided, attempt download
        if not os.path.exists(model_path):
            model_url = os.environ.get('MODEL_URL')
            if model_url:
                try:
                    print(f"[app] downloading model from {model_url} -> {model_path}")
                    tmp_path = model_path + '.download'
                    urllib.request.urlretrieve(model_url, tmp_path)
                    shutil.move(tmp_path, model_path)
                    print('[app] model downloaded successfully')
                except Exception as e:
                    print(f"[app] model download failed: {e}")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                sign_model = pickle.load(f)
        with open(os.path.join(DATASET_DIR, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        print(f"[app] Loaded sign model with {len(label_encoder.classes_)} classes")
    except Exception as e:
        sign_model = None
        label_encoder = None
        print(f"[app] No trained model yet ({e}). Record signs + train from the admin dashboard.")


try_load_model()

hands_detector = None
if USE_SERVER_MEDIAPIPE:
    mp_hands = getattr(mp.solutions, 'hands', None)
    if mp_hands is None:
        # if the environment exposes a nonstandard layout we won't crash here
        USE_SERVER_MEDIAPIPE = False
    else:
        hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
        )


def decode_frame(image_b64):
    """base64 data-URL or raw base64 JPEG -> BGR np array, or None."""
    try:
        header_split = image_b64.split(",", 1)
        raw = header_split[1] if len(header_split) == 2 else header_split[0]
        img_bytes = base64.b64decode(raw)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[decode_frame] error: {e}")
        return None


def normalize_feature_vector(features):
    """Normalize a 126-value left/right hand vector around each wrist.

    Applying this during training keeps previously recorded camera-relative
    samples compatible with the normalized vectors captured by newer versions.
    """
    vector = np.asarray(features, dtype=np.float32).reshape(2, 21, 3).copy()
    for hand in vector:
        if not np.any(hand):
            continue
        hand -= hand[0]
        scale = float(np.max(np.linalg.norm(hand[:, :2], axis=1)))
        if scale > 1e-6:
            hand /= scale
    return vector.reshape(-1)


def augment_landmark_features(features, copies=4, seed=42):
    """Create small, realistic landmark variations for model training only.

    MediaPipe landmarks are already centered at each wrist.  We vary pose
    orientation, size, and landmark position slightly, which helps the model
    tolerate ordinary camera angle and lighting-detection variation without
    polluting the held-out validation samples.
    """
    rng = np.random.default_rng(seed)
    source = np.asarray(features, dtype=np.float32).reshape(-1)
    augmented = [source]

    for _ in range(copies):
        sample = source.reshape(2, 21, 3).copy()
        for hand in sample:
            if not np.any(hand):
                continue
            angle = np.deg2rad(rng.uniform(-12, 12))
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=np.float32,
            )
            scale = rng.uniform(0.90, 1.10)
            hand[:, :2] = (hand[:, :2] @ rotation.T) * scale
            hand[1:] += rng.normal(0, 0.008, size=(20, 3)).astype(np.float32)
            hand[0] = 0.0  # preserve the wrist origin
        augmented.append(normalize_feature_vector(sample.reshape(-1)))

    return np.asarray(augmented, dtype=np.float32)


def get_hand_features(frame):
    """BGR frame -> 126-dim landmark vector (left 63 + right 63), or None if no hand found."""
    if frame is None:
        return None
    if not USE_SERVER_MEDIAPIPE or hands_detector is None:
        # Server-side MediaPipe not available in this environment.
        # Return None to indicate caller should provide precomputed features instead.
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    left = [0.0] * 63
    right = [0.0] * 63
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Make samples robust to where the hand is in the camera frame and
            # to distance from the camera. The wrist is the origin and the
            # largest wrist-to-landmark distance is the scale.
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
            points -= points[0]
            scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
            if scale < 1e-6:
                continue
            coords = (points / scale).reshape(-1).tolist()
            label = handedness.classification[0].label
            if label == "Left":
                left = coords
            else:
                right = coords

    feat = np.array(left + right, dtype=np.float32)
    return feat if np.any(feat) else None


def predict_sign_from_b64(image_b64):
    if sign_model is None or label_encoder is None:
        return None, 0.0
    frame = decode_frame(image_b64)
    if not USE_SERVER_MEDIAPIPE:
        # Server cannot extract landmarks from images in this deployment.
        # Clients should send precomputed `features` to `/api/predict` instead of images.
        return None, 0.0
    feat = get_hand_features(frame)
    if feat is None:
        return None, 0.0
    return predict_sign_from_features(feat)


def predict_sign_from_features(features, normalize=False):
    """Return a prediction for one 126-value (left + right hand) vector."""
    try:
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        if feat.size != 126 or not np.all(np.isfinite(feat)) or not np.any(feat):
            return None, 0.0
        if normalize:
            feat = normalize_feature_vector(feat)
        with model_lock:
            model = sign_model
            encoder = label_encoder
        if model is None or encoder is None:
            return None, 0.0
        candidates = [feat]
        left_present = bool(np.any(feat[:63]))
        right_present = bool(np.any(feat[63:]))
        if left_present != right_present:
            candidates.append(np.concatenate((feat[63:], feat[:63])))
        candidate_matrix = np.asarray(candidates, dtype=np.float32)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(candidate_matrix)
            row_index, idx = np.unravel_index(int(np.argmax(probabilities)), probabilities.shape)
            conf = float(probabilities[row_index, idx]) * 100
        else:
            idx = int(model.predict(candidate_matrix)[0])
            conf = 100.0
        word = encoder.inverse_transform([idx])[0]
        return str(word), conf
    except Exception as e:
        print(f"[predict] error: {e}")
        return None, 0.0


# ------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------

def get_token_from_request():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    # The user dashboard sends the token directly while the admin dashboard
    # uses the conventional ``Bearer <token>`` form. Support both.
    return auth or request.args.get("token")


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = get_token_from_request()
        username = db.verify_session(token)
        if not username:
            return jsonify({"success": False, "message": "Unauthorized"}), 401
        request.username = username
        return f(*args, **kwargs)
    return wrapper


def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = get_token_from_request()
        username = db.verify_session(token)
        if not username or db.get_user_role(username) != "admin":
            return jsonify({"success": False, "message": "Admin access required"}), 403
        request.username = username
        return f(*args, **kwargs)
    return wrapper


# ------------------------------------------------------------------
# Auth routes
# ------------------------------------------------------------------

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True) or {}
    fullname = (data.get("fullname") or "").strip()
    email = (data.get("email") or "").strip().lower()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not all([fullname, email, username]) or len(password) < 6:
        return jsonify({"success": False, "message": "All fields are required (password min 6 chars)."}), 400

    ok, err = db.register_user(fullname, email, username, password)
    if not ok:
        return jsonify({"success": False, "message": err}), 409
    return jsonify({"success": True})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not db.verify_user(username, password):
        return jsonify({"success": False, "message": "Invalid credentials. Try again."}), 401

    token = db.create_session(username)
    info = db.get_user_info(username)
    return jsonify({"success": True, "token": token, "user": info})


@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    data = request.get_json(force=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not db.verify_user(username, password):
        return jsonify({"success": False, "message": "Invalid credentials."}), 401
    if db.get_user_role(username) != "admin":
        return jsonify({"success": False, "message": "This account does not have admin access."}), 403

    token = db.create_session(username)
    info = db.get_user_info(username)
    return jsonify({"success": True, "token": token, "user": info})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    token = get_token_from_request()
    if token:
        db.delete_session(token)
    return jsonify({"success": True})


@app.route("/api/session", methods=["GET"])
def api_session():
    token = get_token_from_request()
    username = db.verify_session(token)
    if not username:
        return jsonify({"success": False}), 401
    return jsonify({"success": True, "user": db.get_user_info(username)})


@app.route("/api/profile", methods=["GET"])
@require_auth
def api_profile():
    """Return the signed-in user's profile for the user dashboard."""
    info = db.get_user_info(request.username)
    if not info:
        return jsonify({"success": False, "message": "User not found"}), 404
    return jsonify({"success": True, **info})


# ------------------------------------------------------------------
# Call history routes
# ------------------------------------------------------------------

@app.route("/api/call-history", methods=["GET"])
@require_auth
def api_call_history():
    calls = db.get_call_history(request.username)
    # Map internal call record keys to the front-end's expected shape
    history = []
    for c in calls:
        participants = c.get('participants') or []
        caller = c.get('host_username')
        # callee: other participants (comma-separated)
        others = [p for p in participants if p != caller]
        callee = ', '.join(others) if others else ''
        history.append({
            'room_id': c.get('room_id'),
            'caller': caller,
            'callee': callee,
            'interpreter_mode': c.get('call_type') == 'interpreter',
            'duration_seconds': c.get('duration_secs') or 0,
            'created_at': c.get('started_at')
        })
    return jsonify({"success": True, "history": history})


# ------------------------------------------------------------------
# Admin — users / calls / stats
# ------------------------------------------------------------------

@app.route("/api/admin/users", methods=["GET"])
@require_admin
def api_admin_users():
    return jsonify({"success": True, "users": db.get_all_users()})


@app.route("/api/admin/calls", methods=["GET"])
@require_admin
def api_admin_calls():
    return jsonify({"success": True, "calls": db.get_all_calls()})


@app.route("/api/admin/stats", methods=["GET"])
@require_admin
def api_admin_stats():
    users = db.get_all_users()
    calls = db.get_all_calls()
    active_calls = [c for c in calls if c["ended_at"] is None]
    run = db.get_active_model_run()
    return jsonify({
        "success": True,
        "stats": {
            "total_users": len(users),
            "total_calls": len(calls),
            "active_calls": len(active_calls),
            "model_loaded": sign_model is not None,
            "total_samples": db.count_samples(),
            "active_run": run,
            "sign_counts": db.get_sample_counts(),
        }
    })


# ------------------------------------------------------------------
# Admin — browser-based sign recording
# ------------------------------------------------------------------

@app.route("/api/admin/samples", methods=["POST"])
@require_admin
def api_admin_add_sample():
    data = request.get_json(force=True) or {}
    label = (data.get("label") or "").strip().lower().replace(" ", "_")
    image_b64 = data.get("image")

    if not label or not image_b64:
        return jsonify({"success": False, "message": "label and image are required."}), 400
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", label):
        return jsonify({"success": False, "message": "Use 1–40 lowercase letters, numbers, _ or - for the sign label."}), 400
    if len(image_b64) > 1_500_000:
        return jsonify({"success": False, "message": "Image is too large. Please capture another frame."}), 413

    frame = decode_frame(image_b64)
    feat = get_hand_features(frame)
    if feat is None:
        return jsonify({"success": False, "message": "No hand detected in frame. Try again."}), 422

    # Prevent a held burst button from filling the dataset with near-identical
    # frames. A varied set trains substantially better than duplicated frames.
    if db.has_near_duplicate(label, feat.tolist(), threshold=0.035):
        return jsonify({"success": False, "message": "Frame is too similar to a recent sample. Move your hand slightly and capture again."}), 422

    db.insert_sample(label, feat.tolist())
    counts = db.get_sample_counts()
    return jsonify({"success": True, "label": label, "count": counts.get(label, 0), "counts": counts})


@app.route("/api/admin/samples/stats", methods=["GET"])
@require_admin
def api_admin_sample_stats():
    return jsonify({"success": True, "counts": db.get_sample_counts(), "total": db.count_samples()})


@app.route("/api/admin/samples/<label>", methods=["DELETE"])
@require_admin
def api_admin_delete_samples(label):
    label = label.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", label):
        return jsonify({"success": False, "message": "Invalid sign label."}), 400
    db.delete_samples_by_label(label)
    return jsonify({"success": True, "counts": db.get_sample_counts()})


# ------------------------------------------------------------------
# Admin — in-app training (replaces running train.py locally)
# ------------------------------------------------------------------

@app.route("/api/admin/train", methods=["POST"])
@require_admin
def api_admin_train():
    global sign_model, label_encoder

    if not training_lock.acquire(blocking=False):
        return jsonify({"success": False, "message": "Training is already in progress."}), 409

    try:
        return _train_model()
    finally:
        training_lock.release()


def _train_model():
    """Train and atomically activate a model from administrator samples."""
    global sign_model, label_encoder

    samples = db.get_all_samples()
    if len(samples) < 20:
        return jsonify({"success": False, "message": "Not enough samples yet (need at least 20 total)."}), 400

    labels = [s["label"] for s in samples]
    features = np.array([normalize_feature_vector(s["features"]) for s in samples], dtype=np.float32)
    counts = Counter(labels)

    if len(counts) < 2:
        return jsonify({"success": False, "message": "Need at least 2 different signs to train."}), 400
    if min(counts.values()) < 5:
        thin = [lbl for lbl, c in counts.items() if c < 5]
        return jsonify({"success": False,
                         "message": f"Record at least 5 varied samples for: {', '.join(thin)}"}), 400

    le_new = LabelEncoder()
    y = le_new.fit_transform(labels)

    test_count = max(len(counts), int(np.ceil(len(samples) * 0.2)))
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=test_count, random_state=42, stratify=y)

    # Expand only the training portion.  Keeping X_test untouched gives the
    # reported accuracy a meaningful measure of real, unseen samples.
    augmented_X = []
    augmented_y = []
    for feature, label in zip(X_train, y_train):
        variants = augment_landmark_features(feature)
        augmented_X.append(variants)
        augmented_y.extend([label] * len(variants))
    X_train = np.vstack(augmented_X)
    y_train = np.asarray(augmented_y)

    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample", min_samples_leaf=1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(
        y_test, preds, target_names=list(le_new.classes_),
        output_dict=True, zero_division=0
    )

    # Never leave a half-written model if the process is interrupted.
    for target, value in ((os.path.join(MODEL_DIR, "best_model.pkl"), clf),
                          (os.path.join(DATASET_DIR, "label_encoder.pkl"), le_new)):
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f)
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    db.save_model_run(
        epochs=0, batch_size=0, lr=0.0,
        best_accuracy=acc, final_loss=0.0,
        feature_size=int(features.shape[1]),
        classes=list(le_new.classes_),
        weights_path="models/best_model.pkl",
    )

    # Hot-reload — live call captions use the new model immediately, no restart.
    with model_lock:
        sign_model = clf
        label_encoder = le_new

    return jsonify({
        "success": True,
        "accuracy": acc,
        "num_samples": len(samples),
        "num_classes": len(le_new.classes_),
        "classes": list(le_new.classes_),
        "report": report,
    })


# ------------------------------------------------------------------
# Live sign prediction — REST endpoint (used by app.js during calls
# as an alternative/fallback to the "sign-frame" socket event below)
# ------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
@require_auth
def api_predict():
    data = request.get_json(force=True) or {}
    features = data.get("features")
    if features is not None:
        word, confidence = predict_sign_from_features(features, normalize=True)
        if word is None:
            return jsonify({"success": False, "message": "Invalid or empty hand landmarks."}), 400
        return jsonify({"success": True, "predicted_class": word, "confidence": confidence})

    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "message": "features or image is required."}), 400

    word, confidence = predict_sign_from_b64(image_b64)
    if word is None:
        return jsonify({"success": False, "message": "No hand detected or model not trained."}), 400

    return jsonify({"success": True, "predicted_class": word, "confidence": confidence})


# ------------------------------------------------------------------
# Static entry points
# ------------------------------------------------------------------

@app.route("/")
def root():
    return send_from_directory(STATIC_PATH, "login.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_PATH, path)


# ------------------------------------------------------------------
# Socket.IO — WebRTC signaling + live sign captions
# ------------------------------------------------------------------

sid_to_user = {}
room_members = {}
room_owners = {}


@socketio.on("connect")
def on_connect(auth):
    token = (auth or {}).get("token") if isinstance(auth, dict) else None
    username = db.verify_session(token)
    if not username:
        return False
    sid_to_user[request.sid] = username
    print(f"[socket] connected: {username} ({request.sid})")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    username = sid_to_user.pop(sid, None)
    for room_id, members in list(room_members.items()):
        if sid in members:
            del members[sid]
            emit("peer-left", {"sid": sid, "username": username}, room=room_id)
            if not members:
                db.end_call(room_id)
                del room_members[room_id]
    print(f"[socket] disconnected: {username} ({sid})")


@socketio.on("join-call")
def on_join_call(data):
    room_id = data.get("room")
    call_type = data.get("callType", "1on1")
    username = sid_to_user.get(request.sid, "unknown")

    if room_id not in room_members:
        room_members[room_id] = {}
        # record this SID as the room owner (host)
        room_owners[room_id] = request.sid
        db.create_call(room_id, username, call_type)
    else:
        db.add_participant(room_id, username)

    existing_peers = [{"sid": sid, "username": u} for sid, u in room_members[room_id].items()]

    join_room(room_id)
    room_members[room_id][request.sid] = username

    emit("existing-peers", {"peers": existing_peers})
    emit("peer-joined", {"sid": request.sid, "username": username}, room=room_id, include_self=False)


@socketio.on("join-room")
def on_join_room(data):
    # Compatibility handler for clients using 'join-room'
    room_id = data.get("room_id") or data.get("room")
    call_type = data.get("call_type", data.get("callType", "1on1"))
    username = sid_to_user.get(request.sid, None)
    if not username:
        # Try token fallback
        token = data.get('token')
        username = db.verify_session(token) if token else None
    if not username:
        return False

    if room_id not in room_members:
        room_members[room_id] = {}
        room_owners[room_id] = request.sid
        db.create_call(room_id, username, call_type)
    else:
        db.add_participant(room_id, username)

    existing_peers = [{"sid": sid, "username": u} for sid, u in room_members[room_id].items()]

    join_room(room_id)
    room_members[room_id][request.sid] = username

    # send caption history to the joining client only
    try:
        captions = db.get_captions(room_id)
    except Exception:
        captions = []
    emit("caption-history", {"captions": captions}, room=request.sid)
    # Emit room-joined to everyone in the room and log join for diagnostics
    app.logger.info(f"[socket] join-room: {username} -> {room_id}; peers={len(existing_peers)}")
    emit("room-joined", {"room_id": room_id, "peers": existing_peers, "room_owner": existing_peers[0]['username'] if existing_peers else username, "participants": [p['username'] for p in existing_peers]}, room=room_id)
    # notify other participants that someone joined
    emit('peer-joined', {'sid': request.sid, 'username': username, 'room_id': room_id}, room=room_id, include_self=False)


@socketio.on("signal")
def on_signal(data):
    to_sid = data.get("to")
    if not to_sid:
        return
    emit("signal", {
        "from": request.sid,
        "username": sid_to_user.get(request.sid, "unknown"),
        "signal": data.get("signal"),
    }, room=to_sid)


@socketio.on('offer')
def on_offer(data):
    # Route an SDP offer to a specific peer (target SID expected in 'to')
    to_sid = data.get('to') or data.get('target')
    if not to_sid:
        return
    payload = dict(data)
    payload['from'] = request.sid
    payload['username'] = sid_to_user.get(request.sid)
    app.logger.info(f"[socket] offer from {sid_to_user.get(request.sid)} ({request.sid}) -> {to_sid}")
    emit('offer', payload, room=to_sid)


@socketio.on('answer')
def on_answer(data):
    # Route an SDP answer to a specific peer
    to_sid = data.get('to') or data.get('target')
    if not to_sid:
        return
    payload = dict(data)
    payload['from'] = request.sid
    payload['username'] = sid_to_user.get(request.sid)
    app.logger.info(f"[socket] answer from {sid_to_user.get(request.sid)} ({request.sid}) -> {to_sid}")
    emit('answer', payload, room=to_sid)


@socketio.on('ice-candidate')
def on_ice_candidate(data):
    # Route ICE candidate to specific peer
    to_sid = data.get('to') or data.get('target')
    if not to_sid:
        return
    payload = dict(data)
    payload['from'] = request.sid
    payload['username'] = sid_to_user.get(request.sid)
    app.logger.info(f"[socket] ice-candidate from {sid_to_user.get(request.sid)} ({request.sid}) -> {to_sid}")
    emit('ice-candidate', payload, room=to_sid)


@socketio.on('camera-toggle')
def on_camera_toggle(data):
    room_id = data.get('room_id')
    enabled = bool(data.get('enabled'))
    username = sid_to_user.get(request.sid, 'unknown')
    # ignore attempts that try to claim another SID
    if 'sid' in data and data.get('sid') != request.sid:
        return
    if room_id:
        emit('camera-changed', {'room_id': room_id, 'enabled': enabled, 'username': username, 'sid': request.sid}, room=room_id)


@socketio.on('mic-toggle')
def on_mic_toggle(data):
    room_id = data.get('room_id')
    enabled = bool(data.get('enabled'))
    username = sid_to_user.get(request.sid, 'unknown')
    # security: do not accept crafted requests claiming to toggle another SID
    if 'sid' in data and data.get('sid') != request.sid:
        return
    if room_id:
        emit('mic-changed', {'room_id': room_id, 'enabled': enabled, 'username': username, 'sid': request.sid}, room=room_id)


@socketio.on('publish-caption')
def on_publish_caption(data):
    """Accept a client-side caption (predicted sign) and persist/broadcast it.

    Clients may send predictions directly after local model inference. This
    handler records the caption (using `db.insert_caption`) and emits the
    `sign-caption` event to the room so all participants (and joining users)
    receive a consistent caption stream.
    """
    room_id = data.get('room') or data.get('room_id')
    text = data.get('text') or data.get('caption')
    confidence = float(data.get('confidence') or 0.0)
    username = sid_to_user.get(request.sid, 'unknown')

    if not room_id or not text:
        return

    try:
        db.insert_caption(room_id, username, text, confidence)
    except Exception:
        pass

    emit('sign-caption', {
        'username': username,
        'text': text,
        'confidence': round(confidence, 1),
    }, room=room_id)


@socketio.on('end-call')
def on_end_call(data):
    room_id = data.get('room_id')
    username = sid_to_user.get(request.sid, 'unknown')
    # Only room owner may end the call for everyone
    owner_sid = room_owners.get(room_id)
    if owner_sid and request.sid != owner_sid:
        emit('end-call-denied', {'room_id': room_id, 'reason': 'only host may end call'}, room=request.sid)
        return
    if room_id:
        db.end_call(room_id)
        emit('call-ended', {'room_id': room_id, 'ended_by': username}, room=room_id)
        # cleanup server-side room state
        if room_id in room_members:
            for sid in list(room_members[room_id].keys()):
                try:
                    leave_room(room_id, sid=sid)
                except Exception:
                    pass
            del room_members[room_id]
    # remove owner record
    if room_id in room_owners:
        try:
            del room_owners[room_id]
        except Exception:
            pass


@socketio.on("leave-call")
def on_leave_call(data):
    room_id = data.get("room")
    username = sid_to_user.get(request.sid, "unknown")
    leave_room(room_id)
    if room_id in room_members and request.sid in room_members[room_id]:
        del room_members[room_id][request.sid]
        emit("peer-left", {"sid": request.sid, "username": username}, room=room_id)
        if not room_members[room_id]:
            db.end_call(room_id)
            del room_members[room_id]


@socketio.on("sign-frame")
def on_sign_frame(data):
    room_id = data.get("room")
    image_b64 = data.get("image")
    username = sid_to_user.get(request.sid, "unknown")

    if not image_b64:
        return

    word, confidence = predict_sign_from_b64(image_b64)
    if word and confidence >= 55:
        # persist caption (encrypted at rest if configured)
        try:
            db.insert_caption(room_id, username, word, float(confidence))
        except Exception:
            pass
        emit("sign-caption", {
            "username": username,
            "text": word,
            "confidence": round(confidence, 1),
        }, room=room_id)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
