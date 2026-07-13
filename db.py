"""
db.py — SQLite data access layer for the ISL Sign Recognition Platform.

Tables:
    users        - login credentials (hashed passwords, email, full name)
    sessions     - active login tokens
    samples      - recorded hand-landmark training samples
    model_runs   - metadata for each trained model
"""

import os
import json
import secrets
import sqlite3
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
    """Create all tables and seed a default admin user on first run."""
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
        """)

        # Seed default admin user only if users table is empty
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
    """
    Register a new user. Returns (True, None) on success or
    (False, error_message) if the username or email already exists.
    """
    with get_connection() as conn:
        # Check for duplicate username
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            return False, "Username already taken. Please choose another."
        # Check for duplicate email
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            return False, "An account with this email already exists."

        conn.execute(
            """INSERT INTO users (fullname, email, username, password_hash, role)
               VALUES (?, ?, ?, ?, 'user')""",
            (fullname, email, username, generate_password_hash(password))
        )
        return True, None


def verify_user(username, password):
    """Return True if username + password are correct."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        return bool(row and check_password_hash(row["password_hash"], password))


def get_user_info(username):
    """Return basic user info dict or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT fullname, email, role FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return {"fullname": row["fullname"], "email": row["email"], "role": row["role"]}
        return None


def create_session(username):
    """Issue a new session token for a logged-in user."""
    token = secrets.token_hex(24)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (username, token) VALUES (?, ?)",
            (username, token)
        )
    return token


def verify_session(token):
    """Return the username if the session token is valid, else None."""
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


def get_all_samples():
    with get_connection() as conn:
        rows = conn.execute("SELECT sign_label, features FROM samples").fetchall()
        return [{"label": r["sign_label"], "features": json.loads(r["features"])} for r in rows]


def count_samples():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM samples").fetchone()["c"]


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
