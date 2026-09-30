from pathlib import Path
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    sum,
    min,
    max
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


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
            "true"
        )
        .config(
            "spark.sql.adaptive.skewJoin.enabled",
            "true"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

def load_transactions(spark):

    transaction_path = (
        RAW_DATA_DIR / "transactions_train.csv"
    )

    print("\n==============================================")
    print("       LOADING H&M TRANSACTIONS")
    print("==============================================")

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(transaction_path))
        .select(
            "t_dat",
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id"
        )
        .withColumn(
            "t_dat",
            col("t_dat").cast("date")
        )
        .withColumn(
            "article_id",
            col("article_id").cast("long")
        )
        .withColumn(
            "price",
            col("price").cast("double")
        )
    )

    return transactions


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
            "price"
        )
        .groupBy("customer_id")
        .agg(
            count("*").alias("purchase_count"),
            avg("price").alias(
                "avg_purchase_price"
            ),
            sum("price").alias(
                "total_spend"
            ),
            min("t_dat").alias(
                "first_purchase_date"
            ),
            max("t_dat").alias(
                "last_purchase_date"
            )
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
            "price"
        )
        .groupBy("article_id")
        .agg(
            count("*").alias("purchase_count"),
            avg("price").alias(
                "avg_price"
            ),
            sum("price").alias(
                "total_revenue"
            ),
            min("t_dat").alias(
                "first_purchase_date"
            ),
            max("t_dat").alias(
                "last_purchase_date"
            )
        )
    )

    return item_features


# ============================================================
# SAVE SMALL AGGREGATED DATASETS
# ============================================================

def save_small_dataframe(df, name):

    output_file = (
        PROCESSED_DATA_DIR / f"{name}.csv"
    )

    # Remove old CSV if it exists.
    if output_file.exists():
        output_file.unlink()

    print(f"\nSaving {name} as CSV...")

    pandas_df = df.toPandas()

    pandas_df.to_csv(
        output_file,
        index=False
    )

    print("\nSaved successfully:")
    print(output_file)

    print(
        f"Rows: {len(pandas_df):,}"
    )
# ============================================================
# MAIN
# ============================================================

def main():

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    spark = create_spark_session()

    try:

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        transactions = load_transactions(spark)

        print("\nTransactions dataframe created.")

        print("\n==============================================")
        print("       TRANSACTIONS READY")
        print("==============================================")


        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        user_features = create_user_features(
            transactions
        )

        save_small_dataframe(
            user_features,
            "user_features"
        )

        print("\nUSER FEATURES COMPLETE")


        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        item_features = create_item_features(
            transactions
        )

        save_small_dataframe(
            item_features,
            "item_features"
        )

        print("\nITEM FEATURES COMPLETE")


        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        print("\n==============================================")
        print("       INTERACTIONS")
        print("==============================================")

        print(
            "The original transactions_train.csv "
            "already contains the interaction data."
        )

        print(
            "No duplicate 31.8M-row interactions "
            "dataset will be created."
        )


        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print("\n")
        print("==============================================")
        print("      DATA PREPROCESSING COMPLETE")
        print("==============================================")

        print("\nGenerated files:")

        print(
            "1. data/processed/user_features.csv"
        )

        print(
            "2. data/processed/item_features.csv"
        )

        print(
            "3. data/raw/transactions_train.csv "
            "(original interaction data)"
        )

        print("\nReady for feature engineering.")


    finally:

        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()