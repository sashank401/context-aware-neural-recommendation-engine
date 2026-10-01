import os
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


TRANSACTIONS_PATH = (
    r"C:\Users\SASHANK VARMA K\recommendation-engine-ml-backup"
    r"\data\raw\transactions_train.csv"
)

OUTPUT_DIR = "data/processed/eda"

VALIDATION_DAYS = 30


# ---------------------------------------------------------
# START SPARK
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("ColdStartEDA")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# ---------------------------------------------------------
# LOAD RAW TRANSACTIONS
# ---------------------------------------------------------

print("Loading raw H&M transactions...")

transactions = spark.read.csv(
    TRANSACTIONS_PATH,
    header=True,
    inferSchema=True
).select(
    "t_dat",
    "customer_id",
    "article_id"
)


transactions = transactions.withColumn(
    "t_dat",
    F.to_date("t_dat")
)


# ---------------------------------------------------------
# FIND VALIDATION PERIOD
# ---------------------------------------------------------

max_date = transactions.select(
    F.max("t_dat")
).collect()[0][0]

validation_start = F.date_sub(
    F.lit(max_date),
    VALIDATION_DAYS - 1
)

print()
print("Maximum transaction date:", max_date)
print("Validation starts:", max_date - pd.Timedelta(days=VALIDATION_DAYS - 1))


# ---------------------------------------------------------
# TRAIN / VALIDATION
# ---------------------------------------------------------

train_df = transactions.filter(
    F.col("t_dat") < validation_start
)

validation_df = transactions.filter(
    F.col("t_dat") >= validation_start
)


# ---------------------------------------------------------
# DISTINCT USERS
# ---------------------------------------------------------

train_users = train_df.select(
    "customer_id"
).distinct()

validation_users = validation_df.select(
    "customer_id"
).distinct()


cold_start_users = validation_users.join(
    train_users,
    on="customer_id",
    how="left_anti"
)


validation_user_count = validation_users.count()
cold_start_user_count = cold_start_users.count()


# ---------------------------------------------------------
# DISTINCT ARTICLES
# ---------------------------------------------------------

train_articles = train_df.select(
    "article_id"
).distinct()

validation_articles = validation_df.select(
    "article_id"
).distinct()


cold_start_articles = validation_articles.join(
    train_articles,
    on="article_id",
    how="left_anti"
)


validation_article_count = validation_articles.count()
cold_start_article_count = cold_start_articles.count()


# ---------------------------------------------------------
# PERCENTAGES
# ---------------------------------------------------------

cold_start_user_percentage = (
    cold_start_user_count /
    validation_user_count *
    100
)

cold_start_article_percentage = (
    cold_start_article_count /
    validation_article_count *
    100
)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print()
print("=" * 60)
print("===== COLD-START ANALYSIS =====")
print("=" * 60)

print()
print("VALIDATION PERIOD")
print("-" * 40)

print(
    f"Validation start date : {max_date - pd.Timedelta(days=VALIDATION_DAYS - 1)}"
)

print(
    f"Validation end date   : {max_date}"
)

print()
print("USER COLD START")
print("-" * 40)

print(
    f"Validation users      : {validation_user_count:,}"
)

print(
    f"Cold-start users      : {cold_start_user_count:,}"
)

print(
    f"Cold-start user rate  : {cold_start_user_percentage:.2f}%"
)

print()
print("ARTICLE COLD START")
print("-" * 40)

print(
    f"Validation articles   : {validation_article_count:,}"
)

print(
    f"Cold-start articles   : {cold_start_article_count:,}"
)

print(
    f"Cold-start article rate: {cold_start_article_percentage:.2f}%"
)


# ---------------------------------------------------------
# SAVE RESULT
# ---------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

result = pd.DataFrame([
    {
        "validation_users": validation_user_count,
        "cold_start_users": cold_start_user_count,
        "cold_start_user_percentage": cold_start_user_percentage,
        "validation_articles": validation_article_count,
        "cold_start_articles": cold_start_article_count,
        "cold_start_article_percentage": cold_start_article_percentage
    }
])

output_path = os.path.join(
    OUTPUT_DIR,
    "cold_start_summary.csv"
)

result.to_csv(
    output_path,
    index=False
)

print()
print(f"Saved: {output_path}")

spark.stop()