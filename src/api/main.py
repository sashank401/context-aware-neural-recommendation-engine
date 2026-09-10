from fastapi import FastAPI

app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "Recommendation Engine API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }