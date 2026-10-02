# Context-Aware Neural Recommendation Engine

A large-scale, context-aware neural recommendation system designed for e-commerce and content recommendation. The system uses **user behavior, user metadata, item metadata, and contextual signals** to generate personalized recommendations through a **Two-Tower Neural Network architecture**.

The project is built around the **H&M Personalized Fashion Recommendations dataset**, containing more than **31 million transactions**, over **1.3 million customers**, and more than **100,000 articles**.

---

## 🚀 Project Overview

Traditional recommendation systems often struggle with:

* New users with little or no interaction history
* New products with limited interaction data
* Changing user interests
* Extremely sparse customer-item interactions
* Large-scale retrieval across hundreds of thousands of products
* Combining user, item, behavioral, and contextual information

This project addresses these challenges through a scalable recommendation architecture based on:

* Large-scale data processing with **Apache Spark**
* Feature engineering from customer, article, and transaction data
* A **Two-Tower Neural Network**
* Dense user and item embeddings
* Similarity-based candidate retrieval
* ANN-based retrieval for scalable inference
* Redis-based embedding storage
* FastAPI serving
* Airflow-based pipeline orchestration
* Docker-based deployment

---

# 🎯 Objectives

The main objectives of this project are:

1. Process millions of e-commerce interactions efficiently.
2. Perform large-scale exploratory data analysis.
3. Build meaningful user and item features.
4. Handle sparse customer-item interactions.
5. Analyze user and item cold-start behavior.
6. Train a Two-Tower neural retrieval model.
7. Generate user and item embeddings.
8. Retrieve personalized items using embedding similarity.
9. Evaluate recommendation quality using Recall@K and NDCG@K.
10. Integrate the ML model with a production-style recommendation API.
11. Build a foundation for scalable retraining and deployment.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │    H&M Raw Data     │
                         │                     │
                         │ Customers           │
                         │ Articles            │
                         │ Transactions        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Apache Spark       │
                         │ Large-Scale ETL     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │     Feature Engineering      │
                    │                              │
                    │ User Features                │
                    │ Item Features                │
                    │ Behavioral Features          │
                    │ Contextual Features          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                     ┌────────────────────────┐
                     │  Two-Tower Neural Net  │
                     │                        │
                     │ ┌────────┐ ┌────────┐ │
                     │ │  User  │ │  Item  │ │
                     │ │ Tower  │ │ Tower  │ │
                     │ └────┬───┘ └───┬────┘ │
                     │      │           │      │
                     │      ▼           ▼      │
                     │   User Emb.   Item Emb. │
                     └───────────┬────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Embedding Retrieval│
                       │                    │
                       │ Cosine / ANN       │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │ Recommendation     │
                       │ Service            │
                       └─────────┬──────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    FastAPI    │
                         └───────────────┘
```

---

# 📊 Dataset

This project uses the **H&M Personalized Fashion Recommendations** dataset.

The dataset contains three primary sources:

### Customers

Customer-level information such as:

* Customer ID
* Age
* Fashion News subscription
* Active status
* Club membership
* Fashion news frequency
* Postal code

### Articles

Product-level information such as:

* Article ID
* Product code
* Product type
* Product group
* Colour
* Department
* Index
* Section
* Garment group
* Product description

### Transactions

Historical customer-item interactions containing:

* Transaction date
* Customer ID
* Article ID
* Price
* Sales channel

---

# 📈 Dataset Scale

The full transaction dataset contains approximately:

| Metric                           |                   Value |
| -------------------------------- | ----------------------: |
| Transactions                     |              31,788,324 |
| Unique customers in transactions |               1,362,281 |
| Total customer records           |               1,371,980 |
| Unique articles in transactions  |                 104,547 |
| Total article records            |                 105,542 |
| Transaction date range           | 2018-09-20 → 2020-09-22 |
| Raw transaction file size        |                 ~3.5 GB |

The scale of the transaction data makes conventional Excel-based analysis impractical, so **Apache Spark** is used for large-scale processing.

---

# 🔍 Exploratory Data Analysis

Large-scale EDA has been completed using Spark-based aggregation and Python visualization.

The analysis includes:

### Transaction Analysis

* Total transaction volume
* Unique customers
* Unique articles
* Transaction date range
* Price statistics
* Monthly transaction trends
* Sales channel distribution

### Customer Analysis

* Customer interaction counts
* Minimum/maximum interactions
* Average interactions per customer
* Customer metadata completeness

### Article Analysis

* Article interaction frequency
* Most frequently interacted articles
* Product category distribution
* Product type distribution
* Department distribution
* Colour distribution
* Article popularity

### Interaction Analysis

The customer-item interaction matrix is extremely sparse.

Observed results:

```text
Unique customers                  1,362,281
Unique articles                     104,547
Unique customer-item pairs       27,306,439
Possible customer-item pairs   142,422,391,707

Interaction density                  0.019173%
Interaction sparsity                 99.980827%
```

This high sparsity motivates the use of embedding-based neural retrieval rather than a dense matrix-based recommendation approach.

---

# 📉 Long-Tail Analysis

The article interaction distribution exhibits a long-tail pattern.

Results include:

```text
Total articles                 104,547
Articles with exactly 1 interaction     4,491
Articles with <= 5 interactions        14,669
Articles with <= 10 interactions       22,477
Articles with <= 50 interactions       47,851
```

This demonstrates that a substantial portion of the catalog receives relatively few interactions.

The system therefore needs to consider item metadata and learned representations rather than relying exclusively on historical popularity.

---

# 🔄 Repeat Interaction Analysis

Historical interactions were analyzed at the unique customer-item pair level.

```text
Unique customer-item pairs       27,306,439
Repeat customer-item pairs        3,515,413
Repeat pair percentage               12.87%
```

This provides evidence that both repeated preferences and exploration of different products are present in the interaction history.

---

# 🧊 Cold-Start Analysis

Cold-start behavior was evaluated using a chronological validation period.

### Validation period

```text
2020-08-24 → 2020-09-22
```

Results:

| Metric                  |   Value |
| ----------------------- | ------: |
| Validation users        | 245,945 |
| Cold-start users        |  24,252 |
| Cold-start user rate    |   9.86% |
| Validation articles     |  28,889 |
| Cold-start articles     |   3,790 |
| Cold-start article rate |  13.12% |

A user is treated as cold-start in this analysis when the user appears in the validation period but has no interaction history in the preceding training period.

An article is treated similarly when it appears during validation without appearing in the earlier training period.

These results motivate the use of metadata-aware representations and dedicated cold-start handling.

---

# 🧠 Feature Engineering

The feature engineering pipeline creates separate user and item feature representations.

## User Features

Examples include:

* Customer ID
* Age
* Club membership
* Fashion news frequency
* Active status
* Purchase count
* Average purchase price
* Total spend
* First purchase date
* Last purchase date

## Item Features

Examples include:

* Article ID
* Product code
* Product name
* Product type
* Product group
* Colour
* Department
* Index
* Section
* Garment group
* Purchase count
* Average price
* Total revenue
* First purchase date
* Last purchase date

Categorical missing values are handled using an explicit `UNKNOWN` representation where appropriate, while numerical missing values are handled using defined fallback values.

---

# ⚙️ Data Processing Pipeline

The large transaction dataset is processed using Apache Spark.

```text
Raw Transactions
       │
       ▼
Spark DataFrame
       │
       ▼
Schema Validation
       │
       ▼
Transaction Aggregation
       │
       ├──────────────► User Features
       │
       └──────────────► Item Features
                              │
                              ▼
                       Feature Engineering
```

The pipeline avoids loading the entire 3.5 GB transaction file into pandas memory.

Spark is used for large-scale computation, while pandas is used only for small aggregated outputs where appropriate.

---

# 🧪 Train / Validation Strategy

The recommendation problem is treated as a temporal prediction problem.

Instead of randomly splitting interactions, the data is divided chronologically.

```text
Historical Interactions
───────────────────────────────────────┬──────────────
                                       │
                                  Validation
                                       │
                                       ▼
                                 Future period
```

This reduces temporal leakage and better represents the real-world recommendation scenario where historical behavior is used to predict future interactions.

---

# 🧠 Two-Tower Neural Network

The core ML model uses a Two-Tower architecture.

## User Tower

The User Tower learns a dense representation of a customer using features such as:

* Customer identity
* Demographic information
* Historical purchase behavior
* Interaction statistics
* Contextual features

```text
User Features
      │
      ▼
Embedding Layers
      │
      ▼
Dense Neural Layers
      │
      ▼
User Embedding
```

## Item Tower

The Item Tower learns a dense representation of an article using:

* Article identity
* Product metadata
* Category information
* Colour
* Department
* Historical interaction statistics

```text
Item Features
      │
      ▼
Embedding Layers
      │
      ▼
Dense Neural Layers
      │
      ▼
Item Embedding
```

## Retrieval

The model learns to place users close to items they are likely to interact with in embedding space.

```text
User Embedding
      │
      │ similarity
      ▼
Item Embeddings
      │
      ▼
Top-K Candidates
```

TensorFlow and TensorFlow Recommenders are used for the neural retrieval model.

---

# 📏 Model Evaluation

Recommendation quality will be evaluated using ranking metrics rather than classification accuracy alone.

Planned metrics include:

### Recall@K

Measures whether relevant items appear within the top K recommendations.

```text
Recall@K =
Relevant items retrieved in top K
---------------------------------
Total relevant items
```

### NDCG@K

Measures ranking quality while giving higher importance to relevant items appearing near the top of the recommendation list.

Evaluation will be performed on the temporal validation set.

---

# 🔎 Embedding Retrieval

After training, the Two-Tower model will generate:

```text
User Embeddings
       +
Item Embeddings
       ↓
Similarity Search
       ↓
Top-K Recommendations
```

The engineering layer already contains interfaces for:

* Embedding storage
* User embedding lookup
* Item embedding lookup
* Cosine similarity
* Top-K retrieval
* Recommendation service

The initial retrieval implementation provides a foundation that can later be replaced with an ANN index for large-scale serving.

---

# ⚡ Approximate Nearest Neighbor Retrieval

For production-scale retrieval, comparing every user embedding against every item embedding is inefficient.

The planned architecture is:

```text
User Embedding
      │
      ▼
ANN Index
      │
      ▼
Candidate Items
      │
      ▼
Ranking
      │
      ▼
Recommendations
```

The ANN layer is intended to support fast retrieval across the full item catalog.

---

# 🗄️ Redis Embedding Store

Redis is planned as a low-latency storage layer for generated embeddings.

Example conceptual structure:

```text
user:{customer_id}  → user embedding
item:{article_id}   → item embedding
```

This separates model training from online recommendation serving.

---

# 🌐 FastAPI Recommendation Service

A FastAPI service has already been implemented as the serving foundation.

Current endpoints include:

```text
GET /
GET /health
GET /recommend/{customer_id}?k=10
```

Example:

```text
GET /recommend/customer_001?k=5
```

Example response structure:

```json
{
  "customer_id": "customer_001",
  "recommendations": [
    "article_001",
    "article_002",
    "article_003"
  ]
}
```

The current service uses the retrieval abstraction, allowing the underlying retrieval implementation to evolve without changing the API contract.

---

# 🧪 Testing

Automated tests have been added for the engineering components.

Current test coverage includes:

* Recommendation service validation
* Retriever behavior
* Cosine similarity
* API endpoints
* Invalid `k` values
* Recommendation limits
* Retrieval ranking metadata
* Edge cases

The current engineering test suite has passed successfully during development.

---

# 🐳 Deployment Architecture

The planned production-style architecture is:

```text
                    ┌───────────────┐
                    │    Client     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    FastAPI    │
                    └───────┬───────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Recommendation     │
                  │ Service            │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ ANN Retrieval      │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Redis Embeddings   │
                  └────────────────────┘
```

Training pipeline:

```text
Raw Data
   │
   ▼
Spark Processing
   │
   ▼
Feature Engineering
   │
   ▼
Two-Tower Training
   │
   ▼
Embedding Generation
   │
   ▼
ANN Index / Redis
   │
   ▼
FastAPI
```

---

# ⏱️ Automated Retraining

Apache Airflow is planned for workflow orchestration.

The intended pipeline is:

```text
Data Update
    ↓
Data Validation
    ↓
Feature Generation
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Embedding Generation
    ↓
ANN Index Update
    ↓
Redis Update
```

This will provide a foundation for scheduled model and embedding updates.

---

# 🛠️ Technology Stack

| Area             | Technology              |
| ---------------- | ----------------------- |
| Programming      | Python                  |
| Data Processing  | Apache Spark / PySpark  |
| Data Analysis    | Pandas                  |
| Visualization    | Matplotlib / Seaborn    |
| Machine Learning | Scikit-learn            |
| Deep Learning    | TensorFlow / Keras      |
| Recommendation   | TensorFlow Recommenders |
| API              | FastAPI                 |
| API Server       | Uvicorn                 |
| Retrieval        | Cosine Similarity / ANN |
| Cache / Storage  | Redis                   |
| Orchestration    | Apache Airflow          |
| Containerization | Docker                  |
| Testing          | Pytest                  |
| Version Control  | Git / GitHub            |

---

# 📁 Project Structure

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
│   │   └── embeddings.py
│   │
│   ├── recommendation/
│   │   └── service.py
│   │
│   └── retrieval/
│       ├── retriever.py
│       └── similarity.py
│
├── tests/
│   ├── test_api.py
│   ├── test_recommendation_service.py
│   ├── test_retriever.py
│   └── test_similarity.py
│
├── config/
│
├── deployment/
│   ├── docker/
│   └── airflow/
│
├── docs/
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# 💻 Installation

## 1. Clone the repository

```bash
git clone https://github.com/sashank401/context-aware-neural-recommendation-engine.git
cd context-aware-neural-recommendation-engine
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# 🔧 Environment Configuration

For the Spark environment on Windows, Java 17 is required.

Example:

```powershell
$env:JAVA_HOME="C:\Program Files\Java\jdk-17.0.20.1"
$env:Path="$env:JAVA_HOME\bin;$env:Path"
```

TensorFlow Recommenders may require the legacy Keras compatibility setting depending on the TensorFlow/Keras environment:

```powershell
$env:TF_USE_LEGACY_KERAS="1"
```

The exact environment configuration may vary depending on the installed TensorFlow version.

---

# ▶️ Running the API

Start the FastAPI service:

```powershell
uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 🧪 Running Tests

Run the test suite:

```powershell
pytest
```

The project uses `pyproject.toml` to configure the test path and Python import path.

---

# 📚 Data Processing

The transaction dataset should not be opened directly in Excel because the raw transaction file is approximately 3.5 GB.

The recommended processing approach is:

```text
Raw CSV
  ↓
PySpark
  ↓
Aggregations
  ↓
Feature Tables
  ↓
ML Pipeline
```

Raw datasets are intentionally excluded from Git version control.

---

# 🔐 Data and Git Policy

Large raw datasets and generated model artifacts should not be committed to GitHub.

The repository excludes:

```text
data/raw/
data/processed/
*.h5
*.keras
*.pkl
*.joblib
```

except for required directory placeholders.

This keeps the Git repository manageable while allowing the complete processing pipeline to be reproduced locally.

---

# 🧭 Development Roadmap

## Phase 1 — Data Understanding

* [x] Dataset inspection
* [x] Large-scale transaction analysis
* [x] Customer analysis
* [x] Article analysis
* [x] Interaction sparsity analysis
* [x] Long-tail analysis
* [x] Repeat interaction analysis
* [x] Cold-start analysis

## Phase 2 — Data Processing

* [x] PySpark transaction processing
* [x] User aggregation
* [x] Item aggregation
* [x] Feature tables
* [x] Temporal validation strategy

## Phase 3 — Feature Engineering

* [x] User feature construction
* [x] Item feature construction
* [x] Missing-value handling
* [x] Feature enrichment

## Phase 4 — Two-Tower Model

* [ ] User Tower
* [ ] Item Tower
* [ ] Training pipeline
* [ ] Negative sampling
* [ ] Model validation
* [ ] Model checkpointing

## Phase 5 — Evaluation

* [ ] Recall@K
* [ ] NDCG@K
* [ ] Cold-start evaluation
* [ ] Retrieval latency evaluation

## Phase 6 — Embeddings and Retrieval

* [ ] User embedding generation
* [ ] Item embedding generation
* [ ] ANN index
* [ ] Top-K retrieval
* [ ] Retrieval benchmarking

## Phase 7 — Serving

* [x] FastAPI foundation
* [x] Recommendation service
* [x] Retrieval interface
* [x] Similarity interface
* [ ] Redis integration
* [ ] Model-backed recommendations

## Phase 8 — MLOps and Deployment

* [ ] Airflow pipeline
* [ ] Automated retraining
* [ ] Embedding refresh
* [ ] Docker deployment
* [ ] End-to-end testing
* [ ] Stress testing
* [ ] Monitoring

---

# 📌 Current Project Status

The project has completed the **large-scale data understanding, EDA, Spark processing, feature engineering, and initial recommendation-engine engineering foundation**.

The next major milestone is the **Two-Tower Neural Retrieval model**.

Current high-level pipeline status:

```text
Large-Scale EDA              ✅
PySpark Processing           ✅
Feature Engineering          ✅
Cold-Start Analysis          ✅
FastAPI Foundation           ✅
Retrieval Foundation         ✅

Two-Tower Model              🚧
Model Evaluation             🚧
Embedding Generation         ⏳
ANN Retrieval                ⏳
Redis Integration            ⏳
Airflow                      ⏳
Docker Deployment            ⏳
End-to-End Production Flow  ⏳
```

---

# 🎓 Learning Outcomes

This project provides practical experience with:

* Large-scale data processing
* Recommendation systems
* Neural retrieval
* Two-Tower architectures
* Embedding-based search
* Feature engineering
* Temporal validation
* Cold-start analysis
* Sparse interaction modeling
* TensorFlow Recommenders
* PySpark
* FastAPI
* Redis
* ANN retrieval
* Airflow
* Docker
* Automated testing
* Git-based software engineering

---

# 👨‍💻 Project Focus

The project combines **Machine Learning Engineering and Software Engineering** rather than treating the recommendation model as an isolated notebook.

The intended final workflow is:

```text
Data
 ↓
Spark Processing
 ↓
Feature Engineering
 ↓
Two-Tower Training
 ↓
Evaluation
 ↓
Embedding Generation
 ↓
ANN Retrieval
 ↓
Redis
 ↓
Recommendation Service
 ↓
FastAPI
 ↓
Client
```

---

# 📜 Dataset Attribution

This project uses the **H&M Personalized Fashion Recommendations** dataset provided through Kaggle.

The dataset is used for educational and research-oriented recommendation-system development.

Please refer to the original dataset license and terms before redistributing or publishing the underlying data.

---

# ⚠️ Project Status Disclaimer

This repository is an actively developed company internship project.

Some components of the final production architecture are still under development. Results, model performance, and deployment characteristics should therefore be interpreted according to the current implementation stage.

---

# ⭐ Future Enhancements

Potential future improvements include:

* Context-aware user representations
* Session-based recommendations
* Transformer-based sequence modeling
* Hybrid retrieval and ranking
* Popularity-aware negative sampling
* Better cold-start strategies
* ANN benchmarking
* Online feature updates
* Model monitoring
* Recommendation explanations
* A/B testing infrastructure
* Real-time recommendation updates

---

## Author

**Sashank Varma K**

Computer Science Engineering
Machine Learning / AI / Data Science

---

