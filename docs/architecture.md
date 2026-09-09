# System Architecture

## High-Level Flow

H&M Dataset
↓
Distributed Data Processing
↓
Feature Engineering
↓
User & Item Features
↓
Two-Tower Neural Network
↓
User & Item Embeddings
↓
ANN Candidate Retrieval
↓
Redis Feature Store / Cache
↓
FastAPI Recommendation Service
↓
Top-K Personalized Recommendations

## Major Components

### 1. Data Layer
Handles customer, article, and transaction data from the H&M Personalized Fashion Recommendations dataset.

### 2. Data Processing Layer
Uses PySpark to process and transform large-scale transaction data.

### 3. Feature Engineering Layer
Creates user, item, and contextual features such as recency, purchase frequency, popularity, and time-related information.

### 4. ML Layer
Uses TensorFlow and TensorFlow Recommenders to train a Two-Tower Neural Network.

The Query Tower represents the user.

The Candidate Tower represents the item.

### 5. Embedding Layer
The trained model generates numerical vector representations for users and items.

### 6. Retrieval Layer
ANN search is used to quickly find relevant items from the item embedding space.

### 7. Feature Store
Redis stores frequently accessed user features and precomputed item embeddings for low-latency retrieval.

### 8. API Layer
FastAPI provides an endpoint that accepts a User ID and returns personalized Top-K recommendations.

### 9. Orchestration Layer
Apache Airflow schedules data processing, model retraining, and embedding updates.

## Technology Stack

- Python
- PySpark
- TensorFlow
- TensorFlow Recommenders
- Redis
- FastAPI
- Apache Airflow
- Docker

## Project Goal

The final system should demonstrate an end-to-end production-oriented recommendation pipeline, from large-scale data processing to real-time recommendation serving.