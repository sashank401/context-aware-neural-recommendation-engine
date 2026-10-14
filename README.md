# Context-Aware Neural Recommendation Engine

A context-aware neural recommendation system for personalized fashion product retrieval, built using the H&M Personalized Fashion Recommendations dataset.

The project combines customer behavior, customer metadata, product metadata, feature engineering, and a Two-Tower Neural Network to generate user and item embeddings. A similarity-based retrieval pipeline and FastAPI service provide product recommendations.

The implementation covers data analysis, model training, embedding generation, offline ranking evaluation, recommendation retrieval, API integration, and automated testing.

---

## 🚀 Project Overview

Recommendation systems help users discover products that match their interests. However, e-commerce recommendation presents several challenges:

- Sparse customer–item interactions.
- Large product catalogs.
- Differences in customer preferences.
- Products with limited historical interactions.
- New users and products with limited interaction history.
- Efficient retrieval of relevant products from a large catalog.

This project explores a neural retrieval approach using:

- **Apache Spark / PySpark** for large-scale data processing.
- **Feature engineering** for customer and product representations.
- **TensorFlow and TensorFlow Recommenders** for neural recommendation modeling.
- **Two-Tower Neural Network** architecture for user and item embeddings.
- **NumPy embedding storage** and cosine-similarity-based retrieval.
- **FastAPI** for recommendation serving.
- **Pytest** for automated testing.

Approximate nearest-neighbor retrieval, Redis integration, Airflow orchestration, and containerized deployment are potential extensions to the current implementation.

---

## 🎯 Project Objectives

The main objectives are to:

1. Analyze a large-scale e-commerce transaction dataset.
2. Explore customer behavior, product popularity, and interaction sparsity.
3. Construct customer and product features.
4. Prepare training and validation interactions chronologically.
5. Train a Two-Tower neural retrieval model.
6. Generate 64-dimensional user and item embeddings.
7. Retrieve candidate products using vector similarity.
8. Evaluate recommendation quality using Recall@K, Precision@K, and NDCG@K.
9. Expose recommendations through a FastAPI service.
10. Validate the engineering components through automated tests.
11. Identify opportunities for better ranking quality and scalable retrieval.

---

## 🏗️ System Architecture

```text
                 H&M Fashion Dataset
             Customers | Articles | Transactions
                            |
                            v
                  Data Processing
                        PY Spark
                            |
                            v
                  Feature Engineering
                User Features / Item Features
                            |
                            v
                 Two-Tower Neural Model
                    /             \
                   v               v
              User Tower       Item Tower
                   |               |
                   v               v
             User Embedding   Item Embedding
                    \             /
                     \           /
                      v         v
                  Similarity Retrieval
                     Cosine Similarity
                            |
                            v
                  Top-K Recommendations
                            |
                            v
                   Recommendation Service
                            |
                            v
                         FastAPI
                            |
                            v
                          Client
```

The user tower represents a customer, while the item tower represents a product. Both produce vectors in a shared embedding space.

At inference time, the retrieval component compares the selected customer's embedding with available item embeddings and returns the highest-scoring candidates.

---

## 📊 Dataset

This project uses the **H&M Personalized Fashion Recommendations** dataset available through Kaggle.

The dataset contains three primary sources.

### Customers

Customer-level information, including:

- Customer identifier.
- Age.
- Active status and membership-related attributes.
- Fashion news subscription information.
- Other available customer metadata.

### Articles

Product-level information, including:

- Article identifier.
- Product code and product type.
- Product group and colour.
- Department and index.
- Section and garment group.
- Other product metadata.

### Transactions

Historical customer–item interactions containing:

- Transaction date.
- Customer identifier.
- Article identifier.
- Purchase price.
- Sales channel.

---

## 📈 Dataset Scale

The full transaction dataset contains approximately 31.8 million transactions.

| Metric | Value |
|---|---:|
| Total transactions | 31,788,324 |
| Unique customers in transactions | 1,362,281 |
| Customer records | 1,371,980 |
| Unique articles in transactions | 104,547 |
| Article records | 105,542 |
| First transaction date | 2018-09-20 |
| Last transaction date | 2020-09-22 |
| Approximate raw transaction file size | 3.5 GB |

### Model training data

The current model experiment uses a prepared subset of interactions rather than all transactions in the original dataset.

| Dataset component | Rows |
|---|---:|
| Training interactions | 2,002,124 |
| Validation interactions | 300,304 |

These figures describe the current training and validation files. They should not be confused with the full historical transaction count.

---

## 🔍 Exploratory Data Analysis

Exploratory data analysis was performed to understand the scale and characteristics of the recommendation problem.

### Transaction analysis

- Total transaction volume.
- Unique customers and articles.
- Transaction date range.
- Minimum, maximum, and average transaction prices.
- Monthly transaction trends.
- Sales channel distribution.

### Customer analysis

- Customer interaction counts.
- Customer metadata completeness.
- Purchase behavior and frequency.
- Distribution of customer activity.

### Article analysis

- Most frequently interacted articles.
- Product type and category distributions.
- Department and colour distributions.
- Article popularity and interaction frequency.

### Interaction sparsity

The recorded interaction analysis produced the following results:

```text
Unique customers:                 1,362,281
Unique articles:                    104,547
Unique customer-item pairs:       27,306,439
Possible customer-item pairs: 142,422,391,707

Interaction density:                 0.019173%
Interaction sparsity:               99.980827%
```

The interaction matrix is extremely sparse. This motivates an embedding-based retrieval approach that learns representations of customers and products instead of constructing and processing a dense customer–item matrix.

---

## 📉 Long-Tail Analysis

The article interaction distribution demonstrates that many products receive relatively few interactions.

```text
Total articles:                       104,547
Articles with exactly 1 interaction:    4,491
Articles with <= 5 interactions:       14,669
Articles with <= 10 interactions:      22,477
Articles with <= 50 interactions:      47,851
```

This long-tail behavior highlights the limitations of recommending products based only on popularity. Product metadata and learned representations can provide additional information, although their ability to improve recommendations must be verified through evaluation.

---

## 🔄 Repeat Interaction Analysis

The recorded interaction analysis identified repeat customer–item pairs.

```text
Unique customer-item pairs: 27,306,439
Repeat customer-item pairs:   3,515,413
Repeat pair percentage:          12.87%
```

Repeat interactions can indicate recurring preferences. They also make it important to define the evaluation target clearly—for example, whether the system should predict future purchases of previously purchased products or prioritize new products.

---

## 🧊 Cold-Start Analysis

Cold-start behavior was analyzed using a chronological validation period.

```text
Validation period: 2020-08-24 to 2020-09-22
```

| Metric | Value |
|---|---:|
| Validation users | 245,945 |
| Cold-start users | 24,252 |
| Cold-start user rate | 9.86% |
| Validation articles | 28,889 |
| Cold-start articles | 3,790 |
| Cold-start article rate | 13.12% |

A user is considered cold-start in this analysis when they appear during validation but have no interaction history in the preceding training period. An article is treated similarly when it appears during validation but not in the earlier training interactions.

Metadata-aware representations are a potential way to support cold-start scenarios. The current metrics should not be interpreted as proof that cold-start recommendations are already solved.

---

## 🧠 Feature Engineering

The feature engineering pipeline constructs user and item feature representations from the available source data and historical interactions.

### User features

Examples include:

- Customer identifier.
- Age and available customer metadata.
- Purchase count.
- Average purchase price.
- Total spend.
- Other available interaction statistics.

### Item features

Examples include:

- Article identifier.
- Product type and category information.
- Purchase count.
- Average price.
- Total revenue.
- Other available article metadata.

### Numerical preprocessing

Numerical features are converted to appropriate numeric types, missing values are handled using defined fallback values, and continuous features are scaled where required by the model pipeline.

Consistent feature preprocessing matters because numerical values with very different scales can dominate a neural network's context projection.

---

## ⚙️ Data Processing Pipeline

Apache Spark / PySpark is used for large-scale processing of transaction data and aggregation tasks.

```text
Raw CSV Files
      |
      v
Schema and Data Validation
      |
      v
Transaction Processing
      |
      v
Customer / Article Aggregations
      |
      v
Feature Engineering
      |
      v
Prepared Interaction and Feature Tables
      |
      v
Model Training and Evaluation
```

Spark supports large-scale transformations, while pandas is used for appropriate downstream data analysis and prepared tables.

The raw transaction dataset is approximately 3.5 GB, so processing it as a whole in an ordinary spreadsheet is impractical.

---

## 🧪 Training and Validation Strategy

The project uses a chronological train/validation approach to model a future-interaction recommendation task.

Historical interactions are used for training, while later interactions are used for validation.

```text
Historical Interactions                 Later Interactions
|--------------------------------------|-------------------|
                 Training                   Validation
```

A chronological split helps reduce temporal leakage and better reflects the scenario in which historical behavior is used to recommend products for a later period.

The training and validation interaction counts for the current experiment are 2,002,124 and 300,304, respectively.

---

## 🧠 Two-Tower Neural Network

The core recommendation model uses a Two-Tower Neural Network.

### User tower

The user tower generates a representation of the customer using the available user identifier and contextual features.

```text
Customer ID + User Features
             |
             v
      Embedding Layer
             |
             v
     Context Projection
             |
             v
      64-D User Vector
```

### Item tower

The item tower generates a representation of a product using its identifier and available product context.

```text
Article ID + Item Features
             |
             v
      Embedding Layer
             |
             v
     Context Projection
             |
             v
      64-D Item Vector
```

### Retrieval

The model learns representations that can be compared in a shared vector space.

```text
User Embedding
       |
       v
Similarity with Item Embeddings
       |
       v
Rank Candidate Products
       |
       v
Top-K Recommendations
```

The model is implemented using TensorFlow/Keras and TensorFlow Recommenders. The current embedding dimension is 64.

---

## 📦 Generated Embedding Artifacts

The current model pipeline generates NumPy embedding arrays and corresponding ID arrays.

| Artifact | Purpose |
|---|---|
| `user_embeddings.npy` | User embedding vectors |
| `item_embeddings.npy` | Item embedding vectors |
| `user_ids.npy` | User IDs aligned with user vectors |
| `item_ids.npy` | Article IDs aligned with item vectors |
| `context_user_tower.keras` | Saved user tower |
| `context_item_tower.keras` | Saved item tower |

Recorded embedding shapes:

| Artifact | Shape |
|---|---|
| User embeddings | `(1371980, 64)` |
| Item embeddings | `(105542, 64)` |

These generated model artifacts are local runtime assets and are excluded from Git version control.

The ID arrays must remain aligned with their corresponding embedding rows. Incorrect alignment can produce recommendations for the wrong users or articles even when the API executes successfully.

---

## 🔎 Embedding-Based Retrieval

The current retrieval implementation includes:

- User embedding lookup.
- Item embedding lookup.
- Cosine similarity.
- Candidate scoring and top-K retrieval.
- Recommendation service integration.

The retrieval component compares the selected user's embedding with the stored item embeddings and ranks candidates by similarity.

This provides a straightforward retrieval implementation for the current catalog. A full-catalog similarity scan may become expensive as catalog size and request volume increase, motivating a future approximate nearest-neighbor index.

---

## 📏 Offline Ranking Evaluation

Recommendation quality is measured using ranking metrics rather than relying only on API responses or model training loss.

The current neural evaluation uses a deterministic sample of 500 users with random seed 42.

### Final recorded neural results

| Metric | Result |
|---|---:|
| Recall@5 | 0.83% |
| Recall@10 | 1.22% |
| Recall@20 | 1.96% |
| Recall@50 | 3.30% |
| Recall@100 | 5.21% |
| Precision@5 | 0.24% |
| Precision@10 | 0.20% |
| NDCG@10 | 0.0073 |
| NDCG@50 | 0.0125 |
| Evaluated users | 500 |
| Random seed | 42 |

### Recall@K

Recall@K measures the proportion of relevant validation items retrieved within the top K recommendations.

\[
\operatorname{Recall@K} =
\frac{\text{Relevant items retrieved in top K}}
{\text{Total relevant items}}
\]

### Precision@K

Precision@K measures the proportion of the top K recommendations that are relevant.

### NDCG@K

Normalized Discounted Cumulative Gain measures ranking quality and gives greater weight to relevant items appearing nearer the top of the recommendation list.

### Interpreting the results

The current evaluation reports positive Recall@K, Precision@K, and NDCG values. This confirms that the evaluation retrieves relevant items for some sampled users under its current evaluation rules.

The results also indicate that recommendation quality remains an area for improvement.

### Baseline comparison limitation

The recorded popularity baseline was evaluated on 99 users, while the neural model was evaluated on 500 users.

Because the evaluation populations differ, these results do not establish whether the neural model outperforms the popularity baseline. A fair comparison requires both methods to use the same users, relevant targets, candidate catalog, and seen-item filtering rules.

### Evaluation artifacts

```text
reports/
├── ranking_metrics.csv
└── ranking_metrics_fast_sample.csv

notebooks/
└── 02_two_tower_training.ipynb
```

---

## 🌐 FastAPI Recommendation Service

The project includes a FastAPI service that exposes recommendation functionality through HTTP endpoints.

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Health check |
| GET | `/recommend/{customer_id}?k=5` | Retrieve recommendations for a customer |

### Start the API

Activate the virtual environment and run the following command from the repository root:

```powershell
$env:PYTHONPATH = "$PWD"
python -m uvicorn src.api.main:app --reload
```

The local service is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

### Example recommendation request

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/recommend/PASTE_REAL_CUSTOMER_ID?k=5"
```

Replace `PASTE_REAL_CUSTOMER_ID` with an actual customer ID from the saved user ID array.

The endpoint returns article IDs for the selected customer. The recommendation limit is restricted to values from 1 through 100.

The current API and retrieval implementation have been exercised successfully. The availability of an endpoint does not, by itself, guarantee high recommendation relevance or production-scale latency.

---

## 🧪 Automated Testing

The engineering test suite covers the implemented API and retrieval components.

Examples include:

- API root and health endpoints.
- Recommendation endpoint behavior.
- Request validation.
- Recommendation limit validation.
- Embedding loading and lookup.
- Cosine similarity.
- Retriever behavior.
- Recommendation service behavior.
- Edge cases and invalid inputs.

### Run the tests

```powershell
$env:PYTHONPATH = "$PWD"
python -m pytest -q
```

### Latest recorded result

```text
35 passed
1 warning
```

The warning was a Starlette/AnyIO deprecation warning. The recorded run had no test failures.

Test results establish that the covered software behaviors pass; they should be considered separately from offline recommendation-quality metrics.

---

## 🛠️ Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Data processing | Apache Spark / PySpark |
| Data analysis | Pandas |
| Visualization | Matplotlib / Seaborn |
| Numerical computing | NumPy |
| Machine learning | Scikit-learn |
| Deep learning | TensorFlow / Keras |
| Recommendation modeling | TensorFlow Recommenders |
| API | FastAPI |
| API server | Uvicorn |
| Retrieval | Cosine similarity |
| Testing | Pytest |
| Version control | Git / GitHub |

Redis, approximate nearest-neighbor indexing, Airflow, and Docker are intended extensions unless separately configured and verified in the deployed environment.

---

## 📁 Project Structure

```text
context-aware-neural-recommendation-engine/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_two_tower_training.ipynb
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── data_processing/
│   │   ├── data_loader.py
│   │   ├── pyspark_processing.py
│   │   ├── interaction_preparation.py
│   │   ├── eda_analysis.py
│   │   ├── interaction_eda.py
│   │   └── cold_start_eda.py
│   │
│   ├── feature_engineering/
│   │   └── features.py
│   │
│   ├── models/
│   │   ├── embeddings.py
│   │   └── embedding_loader.py
│   │
│   ├── recommendation/
│   │   └── service.py
│   │
│   └── retrieval/
│       ├── retriever.py
│       └── similarity.py
│
├── scripts/
│   └── evaluate_ranking_metrics.py
│
├── tests/
│
├── reports/
│   ├── ranking_metrics.csv
│   └── ranking_metrics_fast_sample.csv
│
├── config/
├── deployment/
├── docs/
├── pyproject.toml
└── requirements.txt
```

This is a representative project layout. Individual files may vary according to the current repository contents.

---

## 💻 Installation

### 1. Clone the repository

```powershell
git clone https://github.com/sashank401/context-aware-neural-recommendation-engine.git
cd context-aware-neural-recommendation-engine
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Use a compatible Python, TensorFlow/Keras, Java, and PySpark environment. The exact setup may depend on the installed package versions.

### 4. Environment configuration

For a Windows environment using Spark, configure `JAVA_HOME` to point to the installed Java 17 directory.

Example:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

If the installed TensorFlow Recommenders setup requires legacy Keras compatibility, configure the environment before importing TensorFlow:

```powershell
$env:TF_USE_LEGACY_KERAS = "1"
```

Adjust the Java path and compatibility settings to match your environment.

---

## 🔐 Data and Git Policy

Large raw datasets, processed interaction files, and generated model artifacts should generally remain outside Git version control.

The repository uses ignore rules for large data and model outputs so that source code, tests, documentation, and configuration remain manageable.

Local model artifacts must be generated or obtained separately before running model-backed recommendations.

---

## 🧭 Development Status

### Implemented

- [x] Dataset exploration and transaction analysis.
- [x] Customer and article analysis.
- [x] Interaction sparsity and long-tail analysis.
- [x] Repeat-interaction analysis.
- [x] Cold-start analysis.
- [x] PySpark data-processing pipeline.
- [x] Prepared training and validation interactions.
- [x] User and item feature engineering.
- [x] Two-Tower model training.
- [x] User and item embedding generation.
- [x] NumPy embedding loading and lookup.
- [x] Cosine-similarity retrieval.
- [x] Recommendation service.
- [x] FastAPI root, health, and recommendation endpoints.
- [x] Offline ranking evaluation.
- [x] Automated engineering tests.
- [x] Evaluation reports.

### Potential extensions

- [ ] Compare neural retrieval and popularity using the same evaluation users.
- [ ] Improve Recall@K and NDCG@K.
- [ ] Benchmark retrieval latency across catalog sizes.
- [ ] Add approximate nearest-neighbor indexing.
- [ ] Integrate Redis for low-latency embedding access.
- [ ] Configure Airflow for scheduled pipeline orchestration.
- [ ] Add Docker-based deployment.
- [ ] Add monitoring and automated model-quality checks.
- [ ] Evaluate dedicated cold-start recommendation strategies.
- [ ] Conduct broader performance and load testing.

These extensions are not presented as completed functionality.

---

## 🎓 Learning Outcomes

This project demonstrates practical work in:

- Large-scale data processing.
- Exploratory data analysis.
- Feature engineering.
- Neural recommendation systems.
- Two-Tower architectures.
- Dense embedding generation.
- Similarity-based retrieval.
- Temporal model evaluation.
- Cold-start and long-tail analysis.
- API development.
- Automated software testing.
- Git-based engineering workflows.

It brings together machine learning and software engineering by connecting a trained retrieval model to an application interface and evaluating the results with ranking metrics.

---

## 📚 Dataset Attribution

This project uses the H&M Personalized Fashion Recommendations dataset provided through Kaggle.

The dataset is used for educational and research-oriented recommendation-system development. Refer to the original dataset license and terms before redistributing or publishing the underlying data.

---

## 👨‍💻 Author

**Sashank Varma K**

Computer Science Engineering  
Machine Learning · Artificial Intelligence · Data Science

---

## ⚠️ Project Status Disclaimer

This project includes a trained neural retrieval model, generated embeddings, similarity-based retrieval, a FastAPI service, and offline evaluation.

The recorded ranking metrics demonstrate measurable retrieval performance, but they do not establish that the model outperforms a popularity baseline. That comparison requires identical evaluation populations and rules.

Production-scale ANN retrieval, Redis-backed embedding storage, automated Airflow retraining, and Docker deployment should be treated as future extensions unless their integration is independently verified.

The project is a foundation for continued improvement of personalized retrieval quality, scalability, and deployment reliability.
