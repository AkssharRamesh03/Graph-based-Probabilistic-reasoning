from app.models.graph_probabilistic_detector import GraphProbabilisticInformationLeakageDetector
from app.utils.data_loader import load_data
import os

def train_and_save():
    real_news, fake_news = load_data()
    train_real = real_news[:int(0.8 * len(real_news))]
    train_fake = fake_news[:int(0.8 * len(fake_news))]
    model = GraphProbabilisticInformationLeakageDetector(p1=2, p2=2)
    model.train(train_real, train_fake)
    os.makedirs("model", exist_ok=True)
    model.save("model/markov_model.pkl")

if __name__ == "__main__":
    train_and_save()