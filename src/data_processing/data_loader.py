from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("HMRecommendationDataAudit")
        .master("local[*]")
        .getOrCreate()
    )


def load_csv(spark, filename):
    path = RAW_DATA_DIR / filename

    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(path))
    )


if __name__ == "__main__":

    spark = create_spark_session()

    print("\n========== LOADING DATASETS ==========")

    customers = load_csv(spark, "customers.csv")
    articles = load_csv(spark, "articles.csv")
    transactions = load_csv(spark, "transactions_train.csv")

    print("\n========== DATASET COUNTS ==========")

    customer_count = customers.count()
    article_count = articles.count()
    transaction_count = transactions.count()

    print("Customers:", customer_count)
    print("Articles:", article_count)
    print("Transactions:", transaction_count)

    print("\n========== UNIQUE IDS ==========")

    unique_transaction_customers = (
        transactions.select("customer_id")
        .distinct()
        .count()
    )

    unique_transaction_articles = (
        transactions.select("article_id")
        .distinct()
        .count()
    )

    print(
        "Unique customers in transactions:",
        unique_transaction_customers
    )

    print(
        "Unique articles in transactions:",
        unique_transaction_articles
    )

    print("\n========== TRANSACTION DATE RANGE ==========")

    transactions.select(
        min("t_dat").alias("earliest_date"),
        max("t_dat").alias("latest_date")
    ).show()

    print("\n========== MISSING VALUES ==========")

    print("Customers:")
    customers.select([
        col(c).isNull().cast("int").alias(c)
        for c in customers.columns
    ]).groupBy().sum().show()

    print("Articles:")
    articles.select([
        col(c).isNull().cast("int").alias(c)
        for c in articles.columns
    ]).groupBy().sum().show()

    print("Transactions:")
    transactions.select([
        col(c).isNull().cast("int").alias(c)
        for c in transactions.columns
    ]).groupBy().sum().show()

    print("\n========== CUSTOMER SCHEMA ==========")
    customers.printSchema()

    print("\n========== ARTICLE SCHEMA ==========")
    articles.printSchema()

    print("\n========== TRANSACTION SCHEMA ==========")
    transactions.printSchema()

    print("\n========== SAMPLE CUSTOMERS ==========")
    customers.show(5, truncate=False)

    print("\n========== SAMPLE ARTICLES ==========")
    articles.show(5, truncate=False)

    print("\n========== SAMPLE TRANSACTIONS ==========")
    transactions.show(5, truncate=False)

    spark.stop()

    print("\n========== DATA AUDIT COMPLETE ==========")