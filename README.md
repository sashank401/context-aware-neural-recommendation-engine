# Context-Aware Neural Recommendation Engine

A deep learning based personalized fashion recommendation system
that uses user context, historical interactions, and item information
to generate relevant product recommendations.

## Project Objective

Build an end-to-end recommendation system capable of:

- Processing large-scale e-commerce transaction data
- Learning user and item representations
- Generating personalized recommendations
- Performing fast candidate retrieval
- Serving recommendations through an API

## Architecture

H&M Dataset
→ Data Processing
→ Feature Engineering
→ Two-Tower Neural Network
→ User & Item Embeddings
→ ANN Retrieval
→ Redis
→ FastAPI
→ Top-K Recommendations

## Technology Stack

- Python
- PySpark
- TensorFlow
- TensorFlow Recommenders
- Redis
- FastAPI
- Apache Airflow
- Docker

## Dataset

H&M Personalized Fashion Recommendations

## Team

- Sashank Varma — Project Engineering Lead
- Ganesh - ML

## Project Status

Active development — Week 1

## Current Development Status

### Completed
- Project repository and Git workflow initialized
- FastAPI service foundation created
- Application configuration loader implemented
- Recommendation service layer implemented
- Embedding storage interface created
- Cosine similarity-based retrieval implemented
- Top-K recommendation retrieval interface implemented
- Initial automated tests added
- Health and recommendation API endpoints validated

### In Progress
- Large-scale H&M dataset processing with PySpark
- Data cleaning and missing-value handling
- User, item, and contextual feature engineering
- Two-Tower neural recommendation model
- Model training and evaluation
- User and item embedding generation

### Upcoming
- Approximate Nearest Neighbor retrieval
- Redis feature/embedding store
- FastAPI integration with trained recommendation model
- Docker deployment
- Apache Airflow pipeline
- End-to-end testing and performance evaluation

## Key Capabilities

- Large-scale transaction processing using Apache PySpark
- User and item feature engineering
- Context-aware recommendation generation
- Two-Tower deep learning architecture
- User and item embedding generation
- Top-K candidate retrieval using embedding similarity
- Approximate Nearest Neighbor (ANN) retrieval
- Redis-based low-latency feature and embedding access
- REST API for real-time recommendations
- Automated pipeline orchestration using Apache Airflow
- Containerized deployment using Docker
- Evaluation using Recall@K and NDCG
- Cold-start handling for new users and products