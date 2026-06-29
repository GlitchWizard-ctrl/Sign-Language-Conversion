# ISL Sign Recognition Platform — Setup

## Correct folder structure (already set up in this zip)
```
isl-project/
├── app.py
├── requirements.txt
└── static/
    ├── index.html
    ├── login.html
    ├── app.js
    ├── login.js
    └── style.css
```
Flask is configured to serve everything from `static/`, so the frontend
files MUST live in that subfolder — that was the main reason things
weren't loading before.

## Run it in VS Code

1. Open the `isl-project` folder in VS Code.
2. Open a terminal and create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the server:
   ```
   python app.py
   ```
5. Open your browser at: http://127.0.0.1:5000
   (It will redirect you to the login page automatically.)

## Login credentials (mock auth)
- Username: `admin`
- Password: `password`

These are hardcoded in `app.py` (`api_login` route) — change them there
if you want different credentials.

## What was fixed
- `styles.css` → `style.css` typo in both `index.html` and `login.html`
  (this was the reason the dashboard had no visual styling).
- Moved all frontend files into a `static/` subfolder, since `app.py`
  is configured to serve static assets from there.
- Added this `requirements.txt` (flask, torch, numpy, scikit-learn).

## Note on auth
The `Authorization` header sent by the frontend (`login.js`/`app.js`)
is currently not validated by any `/api/...` route in `app.py` — only
`/api/login` checks credentials. This is fine for local testing/demo,
but if you deploy this anywhere, you'll want to add real token
verification in the backend before trusting that header.
