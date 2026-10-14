from pathlib import Path
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


PROJECT_ROOT = Path("/opt/recommendation-engine")
MODEL_DIR = PROJECT_ROOT / "models" / "two_tower"


def validate_embedding_artifacts():
    required = [
        "user_ids.npy",
        "user_embeddings.npy",
        "item_ids.npy",
        "item_embeddings.npy",
    ]

    missing = [
        name for name in required
        if not (MODEL_DIR / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing embedding artifacts: {missing}"
        )

    user_ids = np.load(
        MODEL_DIR / "user_ids.npy",
        allow_pickle=True,
    )
    user_embeddings = np.load(
        MODEL_DIR / "user_embeddings.npy",
        mmap_mode="r",
    )
    item_ids = np.load(
        MODEL_DIR / "item_ids.npy",
        allow_pickle=True,
    )
    item_embeddings = np.load(
        MODEL_DIR / "item_embeddings.npy",
        mmap_mode="r",
    )

    if len(user_ids) != len(user_embeddings):
        raise ValueError("User IDs and embeddings do not match")

    if len(item_ids) != len(item_embeddings):
        raise ValueError("Item IDs and embeddings do not match")

    if user_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError("Embedding dimensions do not match")

    print({
        "users": len(user_ids),
        "items": len(item_ids),
        "dimensions": user_embeddings.shape[1],
        "status": "valid",
    })


with DAG(
    dag_id="recommendation_embedding_validation",
    description="Validate exported recommendation embeddings",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["recommendations", "validation"],
) as dag:

    validate_artifacts = PythonOperator(
        task_id="validate_embedding_artifacts",
        python_callable=validate_embedding_artifacts,
    )
