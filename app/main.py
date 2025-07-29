from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.predictor import get_prediction

app = FastAPI(title="Fake News Classifier API")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    try:
        result = get_prediction(request.text)
        return {"prediction": "fake" if result == 1 else "real"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))