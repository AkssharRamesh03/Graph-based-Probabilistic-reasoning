from app.detector.graph_probabilistic_detector import GraphProbabilisticInformationLeakageDetector
from app.utils.data_loader import load_data
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def train_and_save(normal_data=None, leaked_data=None):
    if normal_data is None or leaked_data is None:
        normal_data, leaked_data = load_data()
    train_normal = normal_data[:int(0.8 * len(normal_data))]
    train_leaked = leaked_data[:int(0.8 * len(leaked_data))]
    model = GraphProbabilisticInformationLeakageDetector(p1=2, p2=2)
    # saving the model
    model.train(train_normal, train_leaked)
    model_dir = os.path.join(base_dir, "..", "model")
    model_path = os.path.join(model_dir, "probabilistic_model.pkl")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)


if __name__ == "__main__":
    train_and_save()
