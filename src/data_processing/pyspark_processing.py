
from pathlib import Path
from datetime import timedelta

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    sum,
    min,
    max,
    lit,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAW_TRANSACTIONS_PATH = (
    RAW_DATA_DIR / "transactions_train.csv"
)

TRAIN_INTERACTIONS_PATH = (
    PROCESSED_DATA_DIR / "train_interactions.csv"
)

VALIDATION_INTERACTIONS_PATH = (
    PROCESSED_DATA_DIR / "validation_interactions.csv"
)


# ============================================================
# SPARK SESSION
# ============================================================

def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("HMRecommendationDataProcessing")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "800")
        .config("spark.sql.adaptive.enabled", "true")
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            "true",
        )
        .config(
            "spark.sql.adaptive.skewJoin.enabled",
            "true",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# DETERMINE TRAINING CUTOFF
# ============================================================

def get_training_cutoff_date():
    """
    Determine the final permissible date for feature aggregation.

    Prefer the day before the first validation date.
    If validation data is unavailable, use the maximum date
    in the processed training interactions.
    """

    if VALIDATION_INTERACTIONS_PATH.exists():

        dates = pd.read_csv(
            VALIDATION_INTERACTIONS_PATH,
            usecols=["t_dat"],
        )

        dates["t_dat"] = pd.to_datetime(
            dates["t_dat"],
            errors="coerce",
        )

        first_validation_date = dates["t_dat"].min()

        if pd.notna(first_validation_date):

            cutoff = (
                first_validation_date.date()
                - timedelta(days=1)
            )

            print("First validation date:", first_validation_date.date())
            print("Training feature cutoff:", cutoff)

            return cutoff

    if TRAIN_INTERACTIONS_PATH.exists():

        dates = pd.read_csv(
            TRAIN_INTERACTIONS_PATH,
            usecols=["t_dat"],
        )

        dates["t_dat"] = pd.to_datetime(
            dates["t_dat"],
            errors="coerce",
        )

        last_training_date = dates["t_dat"].max()

        if pd.notna(last_training_date):

            cutoff = last_training_date.date()

            print("Training feature cutoff:", cutoff)

            return cutoff

    raise FileNotFoundError(
        "Cannot determine the training cutoff. "
        "Expected train_interactions.csv or "
        "validation_interactions.csv in data/processed."
    )


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

def load_transactions(spark):

    print("\n==============================================")
    print("       LOADING H&M TRANSACTIONS")
    print("==============================================")

    if RAW_TRANSACTIONS_PATH.exists():

        source_path = RAW_TRANSACTIONS_PATH

        print("Using full raw transaction dataset.")

    elif TRAIN_INTERACTIONS_PATH.exists():

        source_path = TRAIN_INTERACTIONS_PATH

        print(
            "WARNING: Raw transactions are unavailable.\n"
            "Using sampled training interactions as fallback."
        )

    else:

        raise FileNotFoundError(
            "Cannot find either:\n"
            f"  {RAW_TRANSACTIONS_PATH}\n"
            f"  {TRAIN_INTERACTIONS_PATH}"
        )

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(source_path))
        .select(
            "t_dat",
            "customer_id",
            "article_id",
            "price",
        )
        .withColumn(
            "t_dat",
            col("t_dat").cast("date"),
        )
        .withColumn(
            "article_id",
            col("article_id").cast("long"),
        )
        .withColumn(
            "price",
            col("price").cast("double"),
        )
        .filter(
            col("t_dat").isNotNull()
            & col("customer_id").isNotNull()
            & col("article_id").isNotNull()
            & col("price").isNotNull()
        )
    )

    print("Transaction source:", source_path)

    return transactions


# ============================================================
# FILTER TRAINING-PERIOD TRANSACTIONS
# ============================================================

def filter_training_transactions(
    transactions,
    cutoff_date,
):

    print("\n==============================================")
    print("   FILTERING TRAINING-PERIOD TRANSACTIONS")
    print("==============================================")

    training_transactions = transactions.filter(
        col("t_dat")
        <= lit(cutoff_date.isoformat()).cast("date")
    )

    print("Cutoff date:", cutoff_date)
    print("Validation-period transactions excluded.")

    return training_transactions


# ============================================================
# USER FEATURES
# ============================================================

def create_user_features(transactions):

    print("\n==============================================")
    print("       CREATING USER FEATURES")
    print("==============================================")

    user_features = (
        transactions
        .select(
            "customer_id",
            "t_dat",
            "price",
        )
        .groupBy("customer_id")
        .agg(
            count("*").alias("purchase_count"),
            avg("price").alias("avg_purchase_price"),
            sum("price").alias("total_spend"),
            min("t_dat").alias("first_purchase_date"),
            max("t_dat").alias("last_purchase_date"),
        )
    )

    return user_features


# ============================================================
# ITEM FEATURES
# ============================================================

def create_item_features(transactions):

    print("\n==============================================")
    print("       CREATING ITEM FEATURES")
    print("==============================================")

    item_features = (
        transactions
        .select(
            "article_id",
            "t_dat",
            "price",
        )
        .groupBy("article_id")
        .agg(
            count("*").alias("purchase_count"),
            avg("price").alias("avg_price"),
            sum("price").alias("total_revenue"),
            min("t_dat").alias("first_purchase_date"),
            max("t_dat").alias("last_purchase_date"),
        )
    )

    return item_features


# ============================================================
# SAVE AGGREGATED DATASETS
# ============================================================

def save_small_dataframe(df, name):

    output_file = (
        PROCESSED_DATA_DIR / f"{name}.csv"
    )

    temporary_file = output_file.with_suffix(".tmp.csv")

    print(f"\nSaving {name}...")

    pandas_df = df.toPandas()

    try:

        pandas_df.to_csv(
            temporary_file,
            index=False,
        )

        # Replace the existing file only after the new CSV
        # has been written successfully.
        temporary_file.replace(output_file)

    finally:

        if temporary_file.exists():
            temporary_file.unlink()

    print("Saved:", output_file)
    print(f"Rows: {len(pandas_df):,}")


# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Establish the cutoff before starting the expensive work.
    cutoff_date = get_training_cutoff_date()

    spark = create_spark_session()

    try:

        # ----------------------------------------------------
        # STEP 1: LOAD TRANSACTIONS
        # ----------------------------------------------------

        transactions = load_transactions(spark)

        # ----------------------------------------------------
        # STEP 2: REMOVE VALIDATION-PERIOD TRANSACTIONS
        # ----------------------------------------------------

        training_transactions = filter_training_transactions(
            transactions,
            cutoff_date,
        )

        # ----------------------------------------------------
        # STEP 3: BUILD USER FEATURES
        # ----------------------------------------------------

        user_features = create_user_features(
            training_transactions
        )

        save_small_dataframe(
            user_features,
            "user_features",
        )

        print("\nUSER FEATURES COMPLETE")

        # ----------------------------------------------------
        # STEP 4: BUILD ITEM FEATURES
        # ----------------------------------------------------

        item_features = create_item_features(
            training_transactions
        )

        save_small_dataframe(
            item_features,
            "item_features",
        )

        print("\nITEM FEATURES COMPLETE")

        # ----------------------------------------------------
        # COMPLETION SUMMARY
        # ----------------------------------------------------

        print("\n==============================================")
        print("      DATA PREPROCESSING COMPLETE")
        print("==============================================")

        print("Feature cutoff:", cutoff_date)
        print("Generated:")
        print("  data/processed/user_features.csv")
        print("  data/processed/item_features.csv")

        if RAW_TRANSACTIONS_PATH.exists():

            print(
                "Feature source: Full raw transactions, "
                "restricted to the training period."
            )

        else:

            print(
                "Feature source: Sampled training interactions."
            )

            print(
                "CAVEAT: Feature statistics are based on the "
                "sampled training interactions, not the "
                "complete original transaction history."
            )

        print(
            "\nNext: rerun feature engineering to regenerate "
            "the enriched feature CSVs."
        )

    finally:

        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
