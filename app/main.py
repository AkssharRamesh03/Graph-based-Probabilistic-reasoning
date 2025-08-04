from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.predictor import get_prediction
from app.training.train_model import train_and_save

app = FastAPI(title="LLM Data leakage Classifier API")


class TextRequest(BaseModel):
    text: str


class TrainingRequest(BaseModel):
    normal_text: list[str]
    leaked_text: list[str]


@app.post("/predict")
def predict(request: TextRequest):
    try:
        result = get_prediction(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train(request: TrainingRequest):
    try:
        train_and_save(request.normal_text, request.leaked_text)
        return {"status": "Model trained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
