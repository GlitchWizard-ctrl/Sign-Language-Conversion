"""
db.py — SQLite data access layer for the ISL Sign Recognition Platform.
"""

import os
import json
import secrets
import sqlite3
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from werkzeug.security import generate_password_hash, check_password_hash

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "datasets")
DB_PATH = os.path.join(DATASET_PATH, "isl_data.db")

os.makedirs(DATASET_PATH, exist_ok=True)


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname      TEXT NOT NULL DEFAULT '',
                email         TEXT UNIQUE NOT NULL DEFAULT '',
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'user',
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL,
                token      TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS samples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sign_label  TEXT NOT NULL,
                features    TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS model_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                epochs         INTEGER,
                batch_size     INTEGER,
                learning_rate  REAL,
                best_accuracy  REAL,
                final_loss     REAL,
                feature_size   INTEGER,
                classes        TEXT NOT NULL,
                weights_path   TEXT NOT NULL,
                is_active      INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS call_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id         TEXT NOT NULL,
                call_type       TEXT NOT NULL DEFAULT '1on1',
                host_username   TEXT NOT NULL,
                participants    TEXT NOT NULL DEFAULT '[]',
                started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at        TIMESTAMP,
                duration_secs   INTEGER
            );
        """)

        # Early versions stored calls with caller/callee/start_time columns.
        # Preserve those records while upgrading to the room-based call schema.
        call_columns = {row["name"] for row in conn.execute("PRAGMA table_info(call_history)")}
        required_call_columns = {"room_id", "call_type", "host_username", "participants", "started_at", "ended_at", "duration_secs"}
        if not required_call_columns.issubset(call_columns):
            legacy_rows = conn.execute("SELECT * FROM call_history").fetchall()
            conn.execute("ALTER TABLE call_history RENAME TO call_history_legacy")
            conn.execute("""
                CREATE TABLE call_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id         TEXT NOT NULL,
                    call_type       TEXT NOT NULL DEFAULT '1on1',
                    host_username   TEXT NOT NULL,
                    participants    TEXT NOT NULL DEFAULT '[]',
                    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at        TIMESTAMP,
                    duration_secs   INTEGER
                )
            """)
            for row in legacy_rows:
                keys = set(row.keys())
                host = row["caller"] if "caller" in keys else "unknown"
                callee = row["callee"] if "callee" in keys else host
                started = row["start_time"] if "start_time" in keys else row["created_at"]
                ended = row["end_time"] if "end_time" in keys else None
                duration = row["duration_seconds"] if "duration_seconds" in keys else None
                conn.execute(
                    """INSERT INTO call_history
                       (room_id, call_type, host_username, participants, started_at, ended_at, duration_secs)
                       VALUES (?, '1on1', ?, ?, ?, ?, ?)""",
                    (row["room_id"], host, json.dumps([host, callee]), started, ended, duration)
                )

        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        if row["c"] == 0:
            conn.execute(
                """INSERT INTO users (fullname, email, username, password_hash, role)
                   VALUES (?, ?, ?, ?, ?)""",
                ("Administrator", "admin@isl.local", "admin",
                 generate_password_hash("password"), "admin")
            )


# -----------------------------------------------------------------------
# USERS / AUTH
# -----------------------------------------------------------------------

def register_user(fullname, email, username, password):
    with get_connection() as conn:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            return False, "Username already taken. Please choose another."
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            return False, "An account with this email already exists."

        conn.execute(
            """INSERT INTO users (fullname, email, username, password_hash, role)
               VALUES (?, ?, ?, ?, 'user')""",
            (fullname, email, username, generate_password_hash(password))
        )
        return True, None


def verify_user(username, password):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        return bool(row and check_password_hash(row["password_hash"], password))


def get_user_info(username):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT fullname, email, role FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return {"fullname": row["fullname"], "email": row["email"], "role": row["role"], "username": username}
        return None


def get_user_role(username):
    with get_connection() as conn:
        row = conn.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
        return row["role"] if row else None


def create_session(username):
    token = secrets.token_hex(24)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (username, token) VALUES (?, ?)",
            (username, token)
        )
    return token


def verify_session(token):
    if not token:
        return None
    with get_connection() as conn:
        row = conn.execute(
            "SELECT username FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        return row["username"] if row else None


def delete_session(token):
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# -----------------------------------------------------------------------
# SAMPLES
# -----------------------------------------------------------------------

def insert_sample(sign_label, features):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO samples (sign_label, features) VALUES (?, ?)",
            (str(sign_label), json.dumps(features))
        )


def has_near_duplicate(sign_label, features, threshold=0.035):
    """Check recent samples only; this keeps capture responsive as data grows."""
    candidate = np.asarray(features, dtype=float)
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT features FROM samples WHERE sign_label = ? ORDER BY id DESC LIMIT 40", (sign_label,)
        ).fetchall()
    for row in rows:
        previous = np.asarray(json.loads(row["features"]), dtype=float)
        if previous.shape == candidate.shape and float(np.mean((previous - candidate) ** 2) ** 0.5) < threshold:
            return True
    return False


def get_all_samples():
    with get_connection() as conn:
        rows = conn.execute("SELECT sign_label, features FROM samples").fetchall()
        return [{"label": r["sign_label"], "features": json.loads(r["features"])} for r in rows]


def count_samples():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM samples").fetchone()["c"]


def get_sample_counts():
    """Returns {label: count} for every recorded sign, sorted by label."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT sign_label, COUNT(*) AS c FROM samples GROUP BY sign_label ORDER BY sign_label"
        ).fetchall()
        return {r["sign_label"]: r["c"] for r in rows}


def delete_samples_by_label(label):
    with get_connection() as conn:
        conn.execute("DELETE FROM samples WHERE sign_label = ?", (label,))


def clear_all_samples():
    with get_connection() as conn:
        conn.execute("DELETE FROM samples")


# -----------------------------------------------------------------------
# MODEL RUNS
# -----------------------------------------------------------------------

def save_model_run(epochs, batch_size, lr, best_accuracy, final_loss,
                   feature_size, classes, weights_path):
    with get_connection() as conn:
        conn.execute("UPDATE model_runs SET is_active = 0")
        conn.execute("""
            INSERT INTO model_runs
                (epochs, batch_size, learning_rate, best_accuracy, final_loss,
                 feature_size, classes, weights_path, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (epochs, batch_size, lr, best_accuracy, final_loss,
              feature_size, json.dumps(classes), weights_path))


def get_active_model_run():
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM model_runs WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "epochs": row["epochs"],
            "batch_size": row["batch_size"],
            "learning_rate": row["learning_rate"],
            "best_accuracy": row["best_accuracy"],
            "final_loss": row["final_loss"],
            "feature_size": row["feature_size"],
            "classes": json.loads(row["classes"]),
            "weights_path": row["weights_path"],
        }


# -----------------------------------------------------------------------
# CALL HISTORY
# -----------------------------------------------------------------------

def create_call(room_id, host_username, call_type="1on1"):
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM call_history WHERE room_id = ? AND ended_at IS NULL", (room_id,)
        ).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute(
            """INSERT INTO call_history (room_id, call_type, host_username, participants)
               VALUES (?, ?, ?, ?)""",
            (room_id, call_type, host_username, json.dumps([host_username]))
        )
        return cur.lastrowid


def add_participant(room_id, username):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, participants FROM call_history WHERE room_id = ? AND ended_at IS NULL",
            (room_id,)
        ).fetchone()
        if not row:
            return
        parts = json.loads(row["participants"])
        if username not in parts:
            parts.append(username)
        conn.execute(
            "UPDATE call_history SET participants = ? WHERE id = ?",
            (json.dumps(parts), row["id"])
        )


def end_call(room_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, started_at FROM call_history WHERE room_id = ? AND ended_at IS NULL",
            (room_id,)
        ).fetchone()
        if not row:
            return
        started = datetime.fromisoformat(row["started_at"])
        duration = int((datetime.utcnow() - started).total_seconds())
        conn.execute(
            "UPDATE call_history SET ended_at = CURRENT_TIMESTAMP, duration_secs = ? WHERE id = ?",
            (duration, row["id"])
        )


def get_call_history(username):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM call_history ORDER BY started_at DESC LIMIT 200"
        ).fetchall()
        result = []
        for r in rows:
            parts = json.loads(r["participants"])
            if username in parts or r["host_username"] == username:
                result.append({
                    "room_id": r["room_id"],
                    "call_type": r["call_type"],
                    "host_username": r["host_username"],
                    "participants": parts,
                    "started_at": r["started_at"],
                    "ended_at": r["ended_at"],
                    "duration_secs": r["duration_secs"],
                })
        return result


def get_all_calls():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM call_history ORDER BY started_at DESC LIMIT 500").fetchall()
        return [{
            "room_id": r["room_id"],
            "call_type": r["call_type"],
            "host_username": r["host_username"],
            "participants": json.loads(r["participants"]),
            "started_at": r["started_at"],
            "ended_at": r["ended_at"],
            "duration_secs": r["duration_secs"],
        } for r in rows]


def get_all_users():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, fullname, email, username, role, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
