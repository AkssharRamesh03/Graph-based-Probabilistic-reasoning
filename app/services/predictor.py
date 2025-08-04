from app.detector.graph_probabilistic_detector import GraphProbabilisticInformationLeakageDetector

model = GraphProbabilisticInformationLeakageDetector.load("model/probabilistic_model.pkl")

def get_prediction(text: str) -> int:
    return model.predict(text)