from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    coalesce,
    lit
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
        .appName("HMRecommendationFeatureEngineering")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "400")
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
# LOAD PROCESSED USER / ITEM FEATURES
# ============================================================

def load_processed_data(spark):

    print(
        "\n========== LOADING PROCESSED DATA =========="
    )

    # ---------------- USER FEATURES ----------------

    user_features = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(
            str(
                PROCESSED_DATA_DIR /
                "user_features.csv"
            )
        )
    )

    print(
        f"User features: {user_features.count():,} rows"
    )

    # ---------------- ITEM FEATURES ----------------

    item_features = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(
            str(
                PROCESSED_DATA_DIR /
                "item_features.csv"
            )
        )
    )

    print(
        f"Item features: {item_features.count():,} rows"
    )

    return user_features, item_features


# ============================================================
# LOAD CUSTOMERS
# ============================================================

def load_customers(spark):

    path = RAW_DATA_DIR / "customers.csv"

    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )


# ============================================================
# LOAD ARTICLES
# ============================================================

def load_articles(spark):

    path = RAW_DATA_DIR / "articles.csv"

    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )


# ============================================================
# PREPARE CUSTOMER METADATA
# ============================================================

def prepare_customer_metadata(customers):

    selected_columns = [
        "customer_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age"
    ]

    customers = customers.select(
        *selected_columns
    )

    customers = (
        customers

        .withColumn(
            "FN",
            coalesce(
                col("FN"),
                lit(0)
            )
        )

        .withColumn(
            "Active",
            coalesce(
                col("Active"),
                lit(0)
            )
        )

        .withColumn(
            "club_member_status",
            coalesce(
                col("club_member_status"),
                lit("UNKNOWN")
            )
        )

        .withColumn(
            "fashion_news_frequency",
            coalesce(
                col("fashion_news_frequency"),
                lit("UNKNOWN")
            )
        )

        .withColumn(
            "age",
            coalesce(
                col("age"),
                lit(0)
            )
        )
    )

    return customers


# ============================================================
# PREPARE ARTICLE METADATA
# ============================================================

def prepare_article_metadata(articles):

    selected_columns = [
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name"
    ]

    articles = articles.select(
        *selected_columns
    )

    categorical_columns = [
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name"
    ]

    for column_name in categorical_columns:

        articles = articles.withColumn(
            column_name,
            coalesce(
                col(column_name),
                lit("UNKNOWN")
            )
        )

    return articles


# ============================================================
# CREATE ENRICHED USER FEATURES
# ============================================================

def create_enriched_user_features(
    user_features,
    customers
):

    print(
        "\n========== PREPARING USER FEATURES =========="
    )

    customers = prepare_customer_metadata(
        customers
    )

    enriched_users = (
        customers
        .join(
            user_features,
            on="customer_id",
            how="left"
        )
    )

    # Fill users with no transaction history.
    enriched_users = (
        enriched_users

        .withColumn(
            "purchase_count",
            coalesce(
                col("purchase_count"),
                lit(0)
            )
        )

        .withColumn(
            "avg_purchase_price",
            coalesce(
                col("avg_purchase_price"),
                lit(0.0)
            )
        )

        .withColumn(
            "total_spend",
            coalesce(
                col("total_spend"),
                lit(0.0)
            )
        )
    )

    return enriched_users


# ============================================================
# CREATE ENRICHED ITEM FEATURES
# ============================================================

def create_enriched_item_features(
    item_features,
    articles
):

    print(
        "\n========== PREPARING ITEM FEATURES =========="
    )

    articles = prepare_article_metadata(
        articles
    )

    enriched_items = (
        articles
        .join(
            item_features,
            on="article_id",
            how="left"
        )
    )

    # Fill items with no transaction history.
    enriched_items = (
        enriched_items

        .withColumn(
            "purchase_count",
            coalesce(
                col("purchase_count"),
                lit(0)
            )
        )

        .withColumn(
            "avg_price",
            coalesce(
                col("avg_price"),
                lit(0.0)
            )
        )

        .withColumn(
            "total_revenue",
            coalesce(
                col("total_revenue"),
                lit(0.0)
            )
        )
    )

    return enriched_items


# ============================================================
# SAVE SMALL / MEDIUM DATASETS AS CSV
# ============================================================

def save_dataframe(df, name):

    output_file = (
        PROCESSED_DATA_DIR /
        f"{name}.csv"
    )

    if output_file.exists():
        output_file.unlink()

    print(
        f"\nSaving {name} as CSV..."
    )

    pandas_df = df.toPandas()

    pandas_df.to_csv(
        output_file,
        index=False
    )

    print(
        "Saved successfully:"
    )

    print(
        output_file
    )

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
        # LOAD PROCESSED FEATURES
        # ----------------------------------------------------

        user_features, item_features = (
            load_processed_data(spark)
        )

        # ----------------------------------------------------
        # LOAD RAW METADATA
        # ----------------------------------------------------

        print(
            "\n========== LOADING RAW METADATA =========="
        )

        customers = load_customers(
            spark
        )

        articles = load_articles(
            spark
        )

        print(
            f"Customers: {customers.count():,}"
        )

        print(
            f"Articles: {articles.count():,}"
        )

        # ----------------------------------------------------
        # ENRICH USERS
        # ----------------------------------------------------

        print(
            "\n========== ENRICHING USER FEATURES =========="
        )

        enriched_users = (
            create_enriched_user_features(
                user_features,
                customers
            )
        )

        print(
            f"Enriched users: "
            f"{enriched_users.count():,}"
        )

        enriched_users.printSchema()

        # ----------------------------------------------------
        # ENRICH ITEMS
        # ----------------------------------------------------

        print(
            "\n========== ENRICHING ITEM FEATURES =========="
        )

        enriched_items = (
            create_enriched_item_features(
                item_features,
                articles
            )
        )

        print(
            f"Enriched items: "
            f"{enriched_items.count():,}"
        )

        enriched_items.printSchema()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        print(
            "\n========== SAVING ENRICHED FEATURES =========="
        )

        save_dataframe(
            enriched_users,
            "enriched_user_features"
        )

        save_dataframe(
            enriched_items,
            "enriched_item_features"
        )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print(
            "\n=============================================="
        )

        print(
            "      FEATURE ENGINEERING COMPLETE"
        )

        print(
            "=============================================="
        )

        print(
            "\nGenerated files:"
        )

        print(
            "1. data/processed/enriched_user_features.csv"
        )

        print(
            "2. data/processed/enriched_item_features.csv"
        )

        print(
            "\nReady for Two-Tower feature preparation."
        )

    finally:

        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()