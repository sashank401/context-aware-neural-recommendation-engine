from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions_train.csv"

TRAIN_OUTPUT = PROCESSED_DATA_DIR / "train_interactions.csv"
VALIDATION_OUTPUT = PROCESSED_DATA_DIR / "validation_interactions.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Keep validation chronological.
# H&M transaction data ends on 2020-09-22.
VALIDATION_DAYS = 30

# Maximum number of training interactions used for local
# Two-Tower development.
#
# IMPORTANT:
# This does NOT modify or delete the original 31.8M records.
MAX_TRAIN_ROWS = 2_000_000

# Maximum validation rows.
MAX_VALIDATION_ROWS = 300_000


# ============================================================
# SPARK
# ============================================================

def create_spark_session():
    return (
        SparkSession.builder
        .appName("RecommendationEngineInteractionPreparation")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

def load_transactions(spark):
    print("\nLoading transactions...")

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(TRANSACTIONS_FILE))
    )

    transactions = (
        transactions
        .select(
            F.to_date("t_dat").alias("t_dat"),
            F.col("customer_id").cast("string").alias("customer_id"),
            F.col("article_id").cast("int").alias("article_id"),
            F.col("price").cast("double").alias("price"),
            F.col("sales_channel_id").cast("int").alias("sales_channel_id"),
        )
        .filter(
            F.col("customer_id").isNotNull()
            & F.col("article_id").isNotNull()
            & F.col("t_dat").isNotNull()
        )
    )

    print("Transaction schema:")
    transactions.printSchema()

    return transactions


# ============================================================
# FIND DATE RANGE
# ============================================================

def get_date_range(transactions):
    result = transactions.select(
        F.min("t_dat").alias("min_date"),
        F.max("t_dat").alias("max_date"),
    ).collect()[0]

    min_date = result["min_date"]
    max_date = result["max_date"]

    print(f"\nTransaction date range:")
    print(f"Start: {min_date}")
    print(f"End:   {max_date}")

    return min_date, max_date


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def create_split(transactions, max_date):
    validation_start = F.date_sub(
        F.lit(max_date),
        VALIDATION_DAYS - 1
    )

    training = transactions.filter(
        F.col("t_dat") < validation_start
    )

    validation = transactions.filter(
        F.col("t_dat") >= validation_start
    )

    return training, validation


# ============================================================
# PREPARE INTERACTIONS
# ============================================================

def prepare_interactions(df):
    """
    Keep the fields needed by the retrieval model.

    We retain:
    - customer_id
    - article_id
    - transaction date
    - price
    - sales channel
    """

    return (
        df.select(
            "customer_id",
            "article_id",
            "t_dat",
            "price",
            "sales_channel_id",
        )
        .dropDuplicates(
            [
                "customer_id",
                "article_id",
                "t_dat",
            ]
        )
    )


# ============================================================
# LIMIT TRAINING DATA
# ============================================================

def limit_training_data(training):
    """
    Limit local training size without changing the raw dataset.

    We use a deterministic hash so that the same customer/article
    interaction is selected consistently between runs.
    """

    if MAX_TRAIN_ROWS is None:
        return training

    estimated_count = (
        training
        .select(
            F.count("*").alias("count")
        )
        .collect()[0]["count"]
    )

    print(f"\nAvailable training interactions: {estimated_count:,}")

    if estimated_count <= MAX_TRAIN_ROWS:
        return training

    fraction = MAX_TRAIN_ROWS / estimated_count

    print(
        f"Sampling approximately {MAX_TRAIN_ROWS:,} "
        f"training interactions."
    )

    sampled = (
        training
        .withColumn(
            "_hash",
            F.pmod(
                F.xxhash64(
                    "customer_id",
                    "article_id",
                    "t_dat"
                ),
                F.lit(1_000_000)
            )
        )
        .filter(
            F.col("_hash") < int(fraction * 1_000_000)
        )
        .drop("_hash")
    )

    return sampled


# ============================================================
# LIMIT VALIDATION DATA
# ============================================================

def limit_validation_data(validation):
    if MAX_VALIDATION_ROWS is None:
        return validation

    estimated_count = (
        validation
        .select(
            F.count("*").alias("count")
        )
        .collect()[0]["count"]
    )

    print(f"Available validation interactions: {estimated_count:,}")

    if estimated_count <= MAX_VALIDATION_ROWS:
        return validation

    fraction = MAX_VALIDATION_ROWS / estimated_count

    print(
        f"Sampling approximately {MAX_VALIDATION_ROWS:,} "
        f"validation interactions."
    )

    sampled = (
        validation
        .withColumn(
            "_hash",
            F.pmod(
                F.xxhash64(
                    "customer_id",
                    "article_id",
                    "t_dat"
                ),
                F.lit(1_000_000)
            )
        )
        .filter(
            F.col("_hash") < int(fraction * 1_000_000)
        )
        .drop("_hash")
    )

    return sampled


# ============================================================
# SAVE WITHOUT SPARK WRITE
# ============================================================

def save_dataframe_as_csv(df, output_path):
    """
    Save through Python/Pandas instead of Spark's Windows
    Hadoop output committer.

    This avoids the winutils.exe permission problem encountered
    during local development.
    """

    print(f"\nCollecting data for:")
    print(output_path)

    pandas_df = df.toPandas()

    if output_path.exists():
        output_path.unlink()

    pandas_df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")
    print(f"Rows: {len(pandas_df):,}")
    print(f"Columns: {len(pandas_df.columns)}")


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

        print("=" * 70)
        print("INTERACTION PREPARATION")
        print("=" * 70)

        transactions = load_transactions(spark)

        min_date, max_date = get_date_range(
            transactions
        )

        print("\nCreating chronological train/validation split...")

        training, validation = create_split(
            transactions,
            max_date
        )

        training = prepare_interactions(
            training
        )

        validation = prepare_interactions(
            validation
        )

        training = limit_training_data(
            training
        )

        validation = limit_validation_data(
            validation
        )

        print("\nPreparing final datasets...")

        training = (
            training
            .orderBy("t_dat")
        )

        validation = (
            validation
            .orderBy("t_dat")
        )

        save_dataframe_as_csv(
            training,
            TRAIN_OUTPUT
        )

        save_dataframe_as_csv(
            validation,
            VALIDATION_OUTPUT
        )

        print("\n" + "=" * 70)
        print("INTERACTION PREPARATION COMPLETE")
        print("=" * 70)

        print("\nGenerated files:")
        print(TRAIN_OUTPUT)
        print(VALIDATION_OUTPUT)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()