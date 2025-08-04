from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.predictor import get_prediction
from app.training.train_model import train_and_save
import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Data leakage Classifier API")

class TextRequest(BaseModel):
    text: str


class TrainingRequest(BaseModel):
    normal_text: list[str]
    leaked_text: list[str]


@app.post("/detect")
def predict(request: TextRequest):
    logging.info(f"Detection request received: {request.text[:50]}...")
    try:
        result = get_prediction(request.text)
        return result
    except Exception as e:
        logging.error(f"Detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(request: TrainingRequest):
    logging.info(f"Training request received")
    try:
        train_and_save(request.normal_text, request.leaked_text)
        return {"status": "Model trained and saved successfully."}
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise logging.info(HTTPException(status_code=500, detail=str(e)))
