import os

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    countDistinct,
    max as spark_max,
    min as spark_min,
    sum as spark_sum,
)


TRANSACTIONS_PATH = (
    r"C:\Users\SASHANK VARMA K\recommendation-engine-ml-backup"
    r"\data\raw\transactions_train.csv"
)

OUTPUT_DIR = r"data\processed\eda"


def create_spark_session():
    os.environ["JAVA_HOME"] = (
        r"C:\Program Files\Java\jdk-17.0.20.1"
    )

    spark = (
        SparkSession.builder
        .appName("RecommendationInteractionEDA")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


def save_small_dataframe(dataframe, filename):
    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spark = create_spark_session()

    print("Loading transactions...")

    transactions = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(TRANSACTIONS_PATH)
        .select(
            "t_dat",
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id",
        )
    )

    print("Transaction data loaded.")

    # =========================================================
    # 1. OVERALL INTERACTION STATISTICS
    # =========================================================

    overall = (
        transactions
        .agg(
            count("*").alias("transaction_count"),
            countDistinct("customer_id").alias(
                "unique_customers"
            ),
            countDistinct("article_id").alias(
                "unique_articles"
            ),
            spark_min("t_dat").alias("first_date"),
            spark_max("t_dat").alias("last_date"),
        )
        .collect()[0]
    )

    overall_df = pd.DataFrame(
        [
            {
                "transaction_count": overall["transaction_count"],
                "unique_customers": overall["unique_customers"],
                "unique_articles": overall["unique_articles"],
                "first_date": overall["first_date"],
                "last_date": overall["last_date"],
            }
        ]
    )

    print("\n===== OVERALL INTERACTION STATISTICS =====")
    print(overall_df.to_string(index=False))

    save_small_dataframe(
        overall_df,
        "interaction_overview.csv",
    )

    # =========================================================
    # 2. CUSTOMER INTERACTION DISTRIBUTION
    # =========================================================

    print("\nCalculating customer interaction statistics...")

    customer_interactions = (
        transactions
        .groupBy("customer_id")
        .agg(
            count("*").alias("interaction_count")
        )
    )

    customer_stats = (
        customer_interactions
        .agg(
            count("*").alias("customers"),
            spark_min("interaction_count").alias(
                "min_interactions"
            ),
            spark_max("interaction_count").alias(
                "max_interactions"
            ),
            avg("interaction_count").alias(
                "avg_interactions"
            ),
        )
        .collect()[0]
    )

    customer_stats_df = pd.DataFrame(
        [
            {
                "customers": customer_stats["customers"],
                "min_interactions": customer_stats[
                    "min_interactions"
                ],
                "max_interactions": customer_stats[
                    "max_interactions"
                ],
                "avg_interactions": customer_stats[
                    "avg_interactions"
                ],
            }
        ]
    )

    print("\n===== CUSTOMER INTERACTION STATISTICS =====")
    print(customer_stats_df.to_string(index=False))

    save_small_dataframe(
        customer_stats_df,
        "interaction_customer_stats.csv",
    )

    # =========================================================
    # 3. ARTICLE INTERACTION DISTRIBUTION
    # =========================================================

    print("\nCalculating article interaction statistics...")

    article_interactions = (
        transactions
        .groupBy("article_id")
        .agg(
            count("*").alias("interaction_count")
        )
    )

    article_stats = (
        article_interactions
        .agg(
            count("*").alias("articles"),
            spark_min("interaction_count").alias(
                "min_interactions"
            ),
            spark_max("interaction_count").alias(
                "max_interactions"
            ),
            avg("interaction_count").alias(
                "avg_interactions"
            ),
        )
        .collect()[0]
    )

    article_stats_df = pd.DataFrame(
        [
            {
                "articles": article_stats["articles"],
                "min_interactions": article_stats[
                    "min_interactions"
                ],
                "max_interactions": article_stats[
                    "max_interactions"
                ],
                "avg_interactions": article_stats[
                    "avg_interactions"
                ],
            }
        ]
    )

    print("\n===== ARTICLE INTERACTION STATISTICS =====")
    print(article_stats_df.to_string(index=False))

    save_small_dataframe(
        article_stats_df,
        "interaction_article_stats.csv",
    )

    # =========================================================
    # 4. CUSTOMER-ITEM INTERACTION SPARSITY
    # =========================================================

    print("\nCalculating customer-item sparsity...")

    unique_customers = (
        transactions
        .select("customer_id")
        .distinct()
        .count()
    )

    unique_articles = (
        transactions
        .select("article_id")
        .distinct()
        .count()
    )

    unique_customer_article_pairs = (
        transactions
        .select(
            "customer_id",
            "article_id",
        )
        .distinct()
        .count()
    )

    possible_pairs = (
        unique_customers *
        unique_articles
    )

    interaction_density = (
        unique_customer_article_pairs
        / possible_pairs
        * 100
    )

    interaction_sparsity = (
        100 -
        interaction_density
    )

    sparsity_df = pd.DataFrame(
        [
            {
                "unique_customers": unique_customers,
                "unique_articles": unique_articles,
                "unique_customer_article_pairs":
                    unique_customer_article_pairs,
                "possible_customer_article_pairs":
                    possible_pairs,
                "interaction_density_percent":
                    interaction_density,
                "interaction_sparsity_percent":
                    interaction_sparsity,
            }
        ]
    )

    print("\n===== INTERACTION SPARSITY =====")
    print(sparsity_df.to_string(index=False))

    save_small_dataframe(
        sparsity_df,
        "interaction_sparsity.csv",
    )

    # =========================================================
    # 5. ARTICLE LONG-TAIL ANALYSIS
    # =========================================================

    print("\nCalculating article long-tail statistics...")

    popularity_summary = (
        article_interactions
        .agg(
            count("*").alias("total_articles"),

            spark_sum(
                (col("interaction_count") == 1)
                .cast("int")
            ).alias(
                "articles_with_1_interaction"
            ),

            spark_sum(
                (col("interaction_count") <= 5)
                .cast("int")
            ).alias(
                "articles_with_1_to_5_interactions"
            ),

            spark_sum(
                (col("interaction_count") <= 10)
                .cast("int")
            ).alias(
                "articles_with_1_to_10_interactions"
            ),

            spark_sum(
                (col("interaction_count") <= 50)
                .cast("int")
            ).alias(
                "articles_with_1_to_50_interactions"
            ),
        )
        .collect()[0]
    )

    popularity_df = pd.DataFrame(
        [
            {
                "total_articles":
                    popularity_summary["total_articles"],

                "articles_with_1_interaction":
                    popularity_summary[
                        "articles_with_1_interaction"
                    ],

                "articles_with_1_to_5_interactions":
                    popularity_summary[
                        "articles_with_1_to_5_interactions"
                    ],

                "articles_with_1_to_10_interactions":
                    popularity_summary[
                        "articles_with_1_to_10_interactions"
                    ],

                "articles_with_1_to_50_interactions":
                    popularity_summary[
                        "articles_with_1_to_50_interactions"
                    ],
            }
        ]
    )

    print("\n===== ARTICLE LONG-TAIL ANALYSIS =====")
    print(popularity_df.to_string(index=False))

    save_small_dataframe(
        popularity_df,
        "article_popularity_summary.csv",
    )

    # =========================================================
    # 6. REPEAT CUSTOMER-ITEM INTERACTIONS
    # =========================================================

    print("\nCalculating repeat interactions...")

    pair_counts = (
        transactions
        .groupBy(
            "customer_id",
            "article_id",
        )
        .agg(
            count("*").alias("interaction_count")
        )
    )

    repeat_summary = (
        pair_counts
        .agg(
            count("*").alias(
                "unique_customer_article_pairs"
            ),

            spark_sum(
                (col("interaction_count") > 1)
                .cast("int")
            ).alias(
                "repeat_customer_article_pairs"
            ),
        )
        .collect()[0]
    )

    repeat_pairs = repeat_summary[
        "unique_customer_article_pairs"
    ]

    repeat_count = repeat_summary[
        "repeat_customer_article_pairs"
    ]

    repeat_percentage = (
        repeat_count / repeat_pairs * 100
    )

    repeat_df = pd.DataFrame(
        [
            {
                "unique_customer_article_pairs":
                    repeat_pairs,

                "repeat_customer_article_pairs":
                    repeat_count,

                "repeat_pair_percentage":
                    repeat_percentage,
            }
        ]
    )

    print("\n===== REPEAT INTERACTIONS =====")
    print(repeat_df.to_string(index=False))

    save_small_dataframe(
        repeat_df,
        "repeat_interaction_summary.csv",
    )

    # =========================================================
    # FINISH
    # =========================================================

    spark.stop()

    print("\n========================================")
    print("Interaction EDA completed successfully.")
    print("========================================")


if __name__ == "__main__":
    main()