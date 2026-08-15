# ISL Sign Recognition Platform — Setup

## Correct folder structure (already set up in this zip)
```
isl-project/
├── app.py             ← Flask server + training/inference routes
├── db.py               ← SQLite data access layer (replaces old pickle files)
├── requirements.txt
└── static/
    ├── index.html
    ├── login.html
    ├── app.js
    ├── login.js
    └── style.css
```
Flask is configured to serve everything from `static/`, so the frontend
files MUST live in that subfolder.

## What changed: pickle → SQL (SQLite)

| Before | Now |
|---|---|
| `datasets/raw_data.pkl` | `samples` table in `datasets/isl_data.db` |
| `datasets/label_encoder.pkl` | `classes` (JSON) column on `model_runs` table |
| Hardcoded `admin` / `password` check | `users` table, password stored as a hash (`werkzeug.security`) |
| `Authorization` header sent but never checked | `sessions` table — `/api/login` issues a real token; `/api/record`, `/api/predict`, `/api/train`, `/api/status`, `/api/logout` all verify it |

The `.db` file is created automatically on first run at `datasets/isl_data.db`
(SQLite needs no separate server — it's just a file). Model **weights**
(`.pth` files) still live on disk in `models/`, since large tensors don't
belong in SQL rows — but the **database tracks which weights file is
"active"** via the `model_runs` table, including accuracy, loss, epochs,
and the exact class list used for that run.

## Run it in VS Code

1. Open the `isl-project` folder in VS Code.
2. Use Python 3.12 and create a virtual environment (recommended). MediaPipe's Hands API used by this project is not compatible with Python 3.14:
   ```
   py -3.12 -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the server:
   ```
   py -3.12 app.py
   ```
5. Open your browser at: http://127.0.0.1:5000

## Login credentials
- Username: `admin`
- Password: `password`

This default user is seeded into the `users` table automatically the
first time `isl_data.db` is created. To change it, either:
- delete `datasets/isl_data.db` and edit the seed call in `db.init_db()`
  in `db.py`, or
- add a small admin script that calls `werkzeug.security.generate_password_hash`
  and inserts/updates a row in `users`.

## Administrator workflow

Sign in with an administrator account to open the separate `admin.html`
console. It is protected on both the browser and API sides. From there an
administrator can:

- record sign samples directly from the browser camera;
- use burst capture while the backend rejects near-duplicate frames;
- review or remove samples per label; and
- train a new Random Forest model without leaving the site.

Training uses a held-out, stratified test set and requires at least two signs,
five varied samples for every sign, and 20 samples overall. When training
finishes, the model and label encoder are written atomically and loaded into
live calls immediately—no server restart is needed. For useful real-world
recognition, capture many examples of each sign with different hand positions,
angles, distances, and lighting.

## Inspecting the database
Since it's SQLite, you can open `datasets/isl_data.db` directly with:
- The **SQLite Viewer** or **SQLTools** extension in VS Code, or
- the `sqlite3` CLI: `sqlite3 datasets/isl_data.db` then `.tables`, `SELECT * FROM samples;`, etc.

## What was fixed (from the original version)
- `styles.css` → `style.css` typo in both `index.html` and `login.html`.
- Moved all frontend files into a `static/` subfolder to match Flask's
  static file configuration.
- Replaced pickle-file storage with SQLite (`db.py`).
- Added real session-token verification on protected API routes.
- Added this `requirements.txt` (flask, torch, numpy, scikit-learn, werkzeug).

