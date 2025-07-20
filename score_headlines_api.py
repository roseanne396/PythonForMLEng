from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Load model and transformer
try:
    MODEL_PATH = "svm_model.pkl" 
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded SVM model from %s", MODEL_PATH)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded SentenceTransformer model")
except Exception as e:
    logger.critical("Failed to load model or transformer: %s", e)
    raise

# Request schema
class HeadlinesRequest(BaseModel):
    headlines: List[str]

# GET /status
@app.get("/status")
def get_status():
    return {"status": "OK"}

# POST score_headlines
@app.post("/score_headlines")
def score_headlines(request: HeadlinesRequest):
    try:
        logger.info("Received scoring request with %d headlines", len(request.headlines))
        embeddings = encoder.encode(request.headlines)
        predictions = model.predict(embeddings)
        labels = predictions.tolist()
        return {"labels": labels}
    except Exception as e:
        logger.error("Error scoring headlines: %s", e)
        raise HTTPException(status_code=500, detail="Failed to score headlines")
