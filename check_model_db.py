import sqlite3
import os

DB = os.path.join('datasets', 'isl_data.db')
print('DB path:', DB)
print('Exists:', os.path.exists(DB))
if not os.path.exists(DB):
    raise SystemExit(1)
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute('SELECT id, trained_at, epochs, batch_size, learning_rate, best_accuracy, final_loss, feature_size, classes, weights_path, is_active FROM model_runs ORDER BY id DESC LIMIT 5')
rows = cur.fetchall()
print('rows:', len(rows))
for row in rows:
    print({k: row[k] for k in row.keys()})
cur.close()
conn.close()
