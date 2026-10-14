from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "two_tower"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KS = (5, 10, 20, 50)
SAMPLE_SIZE = 500
SEED = 42
BATCH_SIZE = 32


def main():
    print("Loading interaction data...")
    train = pd.read_csv(
        DATA_DIR / "train_interactions.csv",
        usecols=["customer_id", "article_id"],
        dtype=str,
    )
    valid = pd.read_csv(
        DATA_DIR / "validation_interactions.csv",
        usecols=["customer_id", "article_id"],
        dtype=str,
    )

    print("Loading saved embeddings...")
    # These are the project's own locally generated ID files.
    user_ids = np.load(MODEL_DIR / "user_ids.npy", allow_pickle=True)
    item_ids = np.load(MODEL_DIR / "item_ids.npy", allow_pickle=True)
    user_embeddings = np.load(
        MODEL_DIR / "user_embeddings.npy", mmap_mode="r"
    )
    item_embeddings = np.load(
        MODEL_DIR / "item_embeddings.npy", mmap_mode="r"
    )

    user_ids = [str(x) for x in user_ids]
    item_ids = [str(x) for x in item_ids]
    user_index = {value: i for i, value in enumerate(user_ids)}
    item_index = {value: i for i, value in enumerate(item_ids)}

    if len(user_index) != len(user_ids) or len(item_index) != len(item_ids):
        raise ValueError("Duplicate IDs found in saved embedding ID arrays.")

    # Normalize embeddings so dot products correspond to cosine similarity.
    user_norms = np.linalg.norm(user_embeddings, axis=1)
    item_norms = np.linalg.norm(item_embeddings, axis=1)

    # Drop zero-vector items from ranking by assigning them zero similarity.
    items = np.asarray(item_embeddings, dtype=np.float32).copy()
    item_norms = np.asarray(item_norms)
    nonzero_items = item_norms > 0
    items[nonzero_items] /= item_norms[nonzero_items, None]
    items[~nonzero_items] = 0

    # Normalize user vectors; zero-vector users are not evaluable.
    users = np.asarray(user_embeddings, dtype=np.float32).copy()
    user_norms = np.asarray(user_norms)
    nonzero_users = user_norms > 0
    users[nonzero_users] /= user_norms[nonzero_users, None]
    users[~nonzero_users] = 0

    print("Preparing training history and validation targets...")
    seen_by_user = {}
    for uid, aid in train.itertuples(index=False, name=None):
        if uid in user_index and aid in item_index:
            seen_by_user.setdefault(uid, set()).add(item_index[aid])

    # Only count validation articles that are in the catalog and were not
    # already purchased by that same user in the training period.
    relevant_by_user = {}
    for uid, aid in valid.itertuples(index=False, name=None):
        if uid not in user_index or aid not in item_index:
            continue
        idx = item_index[aid]
        if idx not in seen_by_user.get(uid, set()):
            relevant_by_user.setdefault(uid, set()).add(idx)

    candidates = [
        uid for uid, targets in relevant_by_user.items()
        if targets and user_norms[user_index[uid]] > 0
    ]
    if not candidates:
        raise RuntimeError(
            "No eligible validation users. Check ID alignment and data."
        )

    rng = np.random.default_rng(SEED)
    sample_size = min(SAMPLE_SIZE, len(candidates))
    sampled_users = rng.choice(candidates, size=sample_size, replace=False)

    print(f"Eligible validation users: {len(candidates):,}")
    print(f"Sampled users: {sample_size:,}")
    print(f"Catalog items: {len(item_ids):,}")
    print("Scoring recommendations...")

    metric_sums = {
        k: {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}
        for k in KS
    }
    evaluated = 0
    max_k = max(KS)

    for start in range(0, len(sampled_users), BATCH_SIZE):
        batch_users = list(sampled_users[start:start + BATCH_SIZE])
        row_indices = [user_index[uid] for uid in batch_users]

        # Cosine scores for this batch against the complete item catalog.
        scores = users[row_indices] @ items.T

        for row, uid in enumerate(batch_users):
            seen = seen_by_user.get(uid, set())
            if seen:
                scores[row, list(seen)] = -np.inf

            # Retrieve the highest-scoring max_k items.
            top_idx = np.argpartition(
                scores[row], -max_k
            )[-max_k:]
            top_idx = top_idx[np.argsort(scores[row, top_idx])[::-1]]

            relevant = relevant_by_user[uid]
            evaluated += 1

            for k in KS:
                ranked = top_idx[:k]
                hits = np.array(
                    [1.0 if idx in relevant else 0.0 for idx in ranked]
                )
                hit_count = float(hits.sum())

                recall = hit_count / len(relevant)
                precision = hit_count / k

                discounts = 1.0 / np.log2(
                    np.arange(2, len(hits) + 2)
                )
                dcg = float(np.sum(hits * discounts))
                ideal_count = min(len(relevant), k)
                idcg = float(np.sum(
                    1.0 / np.log2(np.arange(2, ideal_count + 2))
                ))
                ndcg = dcg / idcg if idcg else 0.0

                metric_sums[k]["recall"] += recall
                metric_sums[k]["precision"] += precision
                metric_sums[k]["ndcg"] += ndcg

        print(f"Evaluated {min(start + len(batch_users), sample_size)}"
              f"/{sample_size} users")

    rows = []
    for k in KS:
        rows.append({
            "K": k,
            "Recall@K": metric_sums[k]["recall"] / evaluated,
            "Precision@K": metric_sums[k]["precision"] / evaluated,
            "NDCG@K": metric_sums[k]["ndcg"] / evaluated,
            "evaluated_users": evaluated,
            "seed": SEED,
        })

    results = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "ranking_metrics.csv"
    results.to_csv(output_path, index=False)

    print("\nPRELIMINARY OFFLINE RANKING METRICS")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved results to: {output_path}")
    print(
        "\nCaution: confirm that the saved user embeddings were generated "
        "without validation-period information before treating these "
        "metrics as final."
    )


if __name__ == "__main__":
    main()
