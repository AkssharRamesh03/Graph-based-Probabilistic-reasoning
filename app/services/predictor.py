from typing import Dict, Any

from app.detector.graph_probabilistic_detector import GraphProbabilisticInformationLeakageDetector

model = GraphProbabilisticInformationLeakageDetector.load("app/model/probabilistic_model.pkl")

def get_prediction(text: str) -> dict[str, Any]:
    return {"class": model.predict(text)}