import os
import pickle
import numpy as np

import db  # must be run from the same folder as your existing db.py

SAMPLES_PER_CLASS_TO_MIGRATE = 10  # keep small; app.py will re-augment 100x per sample


def main():
    db.init_db()

    dataset_path = db.DATASET_PATH

    x_train_path = os.path.join(dataset_path, "X_train_landmarks.npy")
    x_test_path = os.path.join(dataset_path, "X_test_landmarks.npy")
    y_train_path = os.path.join(dataset_path, "y_train.npy")
    y_test_path = os.path.join(dataset_path, "y_test.npy")
    encoder_path = os.path.join(dataset_path, "label_encoder.pkl")

    for p in (x_train_path, x_test_path, y_train_path, y_test_path, encoder_path):
        if not os.path.exists(p):
            print(f"✗ Missing expected file: {p}")
            print("  Make sure this script sits in the same folder as app.py/db.py,")
            print("  and that record.py has already been run.")
            return

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    labels = le.inverse_transform(y)  # back to original strings like "1", "2", ...

    print("=" * 55)
    print(f"Loaded {len(X)} total samples across {len(set(labels))} classes:")
    print(f"  {sorted(set(labels))}")
    print("=" * 55)

    rng = np.random.default_rng(42)
    by_class = {}
    for idx, lbl in enumerate(labels):
        by_class.setdefault(lbl, []).append(idx)

    inserted = 0
    for lbl, idxs in sorted(by_class.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        take = idxs[:SAMPLES_PER_CLASS_TO_MIGRATE]
        for idx in take:
            feat = X[idx].astype(float).tolist()
            db.insert_sample(str(lbl), feat)
            inserted += 1
        print(f"  '{lbl}': inserted {len(take)} of {len(idxs)} available samples")

    total_in_db = db.count_samples()

    print("\n" + "=" * 55)
    print(f"Done. Inserted {inserted} new sample rows this run.")
    print(f"Total samples now in isl_data.db: {total_in_db}")
    print("=" * 55)
    print("\nNext steps:")
    print("  1. Start app.py if it isn't already running.")
    print("  2. GET /api/status -> raw_samples should now be > 0.")
    print("  3. POST /api/train to train the LandmarkMLP.")


if __name__ == "__main__":
    main()