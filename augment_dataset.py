"""Expand landmark samples with conservative, label-preserving variations.

The project stores MediaPipe landmark vectors, not camera images.  Therefore
this script augments geometry (camera angle, size and detector noise), rather
than attempting to fabricate photorealistic lighting changes.
"""

import argparse
import json
import sqlite3

import numpy as np

import db


def normalize(vector):
    hands = np.asarray(vector, dtype=np.float32).reshape(2, 21, 3).copy()
    for hand in hands:
        if not np.any(hand):
            continue
        hand -= hand[0]
        scale = float(np.max(np.linalg.norm(hand[:, :2], axis=1)))
        if scale > 1e-6:
            hand /= scale
    return hands.reshape(-1)


def vary(vector, rng):
    hands = normalize(vector).reshape(2, 21, 3)
    for hand in hands:
        if not np.any(hand):
            continue
        angle = np.deg2rad(rng.uniform(-15, 15))
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        hand[:, :2] = (hand[:, :2] @ rotation.T) * rng.uniform(0.88, 1.12)
        hand[1:] += rng.normal(0, 0.010, size=(20, 3)).astype(np.float32)
        hand[0] = 0.0
    return normalize(hands.reshape(-1)).tolist()


def main():
    parser = argparse.ArgumentParser(description="Create geometric landmark variations for every recorded sign.")
    parser.add_argument("--copies", type=int, default=4, help="Synthetic samples to create per existing sample (default: 4).")
    parser.add_argument("--force", action="store_true", help="Allow another expansion after one has already run.")
    args = parser.parse_args()
    if args.copies < 1 or args.copies > 12:
        parser.error("--copies must be between 1 and 12")

    rng = np.random.default_rng(20260815)
    with db.get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS dataset_augmentations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_samples INTEGER NOT NULL,
            copies_per_sample INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        previous = conn.execute("SELECT COUNT(*) FROM dataset_augmentations").fetchone()[0]
        if previous and not args.force:
            raise SystemExit("Dataset was already expanded. Use --force only if you deliberately want more variants.")

        originals = conn.execute("SELECT sign_label, features FROM samples").fetchall()
        additions = []
        for row in originals:
            features = json.loads(row["features"])
            for _ in range(args.copies):
                additions.append((row["sign_label"], json.dumps(vary(features, rng))))

        conn.executemany("INSERT INTO samples (sign_label, features) VALUES (?, ?)", additions)
        conn.execute(
            "INSERT INTO dataset_augmentations (source_samples, copies_per_sample) VALUES (?, ?)",
            (len(originals), args.copies),
        )

    print(f"Added {len(additions)} augmented samples. Dataset now contains {len(originals) + len(additions)} samples.")


if __name__ == "__main__":
    main()
