import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


JAVA_HOME = r"C:\Program Files\Java\jdk-17.0.20.1"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_HOME + r"\bin;" + os.environ["PATH"]


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TRANSACTIONS_PATH = r"C:\Users\SASHANK VARMA K\recommendation-engine-ml-backup\data\raw\transactions_train.csv"

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "eda",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


spark = (
    SparkSession.builder
    .appName("HM Recommendation Engine - Large Scale EDA")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("Spark version:", spark.version)
print("Reading transactions...")

transactions = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(TRANSACTIONS_PATH)
)

print("Transaction schema:")
transactions.printSchema()


# ---------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------

overview = transactions.select(
    F.count("*").alias("transaction_count"),
    F.countDistinct("customer_id").alias("unique_customers"),
    F.countDistinct("article_id").alias("unique_articles"),
    F.min("t_dat").alias("first_transaction_date"),
    F.max("t_dat").alias("last_transaction_date"),
    F.min("price").alias("minimum_price"),
    F.max("price").alias("maximum_price"),
    F.avg("price").alias("average_price"),
)

overview.show(truncate=False)

overview.toPandas().to_csv(
    os.path.join(OUTPUT_DIR, "transaction_overview.csv"),
    index=False,
)


# ---------------------------------------------------------
# 2. Transactions by sales channel
# ---------------------------------------------------------

channel_distribution = (
    transactions
    .groupBy("sales_channel_id")
    .count()
    .orderBy("sales_channel_id")
)

channel_distribution.show()

channel_distribution.toPandas().to_csv(
    os.path.join(OUTPUT_DIR, "sales_channel_distribution.csv"),
    index=False,
)


# ---------------------------------------------------------
# 3. Monthly transaction volume
# ---------------------------------------------------------

monthly_transactions = (
    transactions
    .withColumn(
        "month",
        F.date_format("t_dat", "yyyy-MM")
    )
    .groupBy("month")
    .count()
    .orderBy("month")
)

monthly_transactions.show(30)

monthly_transactions.toPandas().to_csv(
    os.path.join(OUTPUT_DIR, "monthly_transactions.csv"),
    index=False,
)


# ---------------------------------------------------------
# 4. Top articles by transaction count
# ---------------------------------------------------------

top_articles = (
    transactions
    .groupBy("article_id")
    .count()
    .orderBy(F.desc("count"))
    .limit(20)
)

top_articles.show()

top_articles.toPandas().to_csv(
    os.path.join(OUTPUT_DIR, "top_articles.csv"),
    index=False,
)


# ---------------------------------------------------------
# 5. Customer purchase frequency
# ---------------------------------------------------------

customer_frequency = (
    transactions
    .groupBy("customer_id")
    .count()
    .orderBy(F.desc("count"))
    .limit(20)
)

customer_frequency.show()

customer_frequency.toPandas().to_csv(
    os.path.join(OUTPUT_DIR, "top_customers.csv"),
    index=False,
)


# ---------------------------------------------------------
# Finish
# ---------------------------------------------------------

print("\nLarge-scale EDA completed successfully.")
print("Results saved to:")
print(OUTPUT_DIR)

spark.stop()