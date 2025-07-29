import json
import os

def load_data():
    with open("app/data/train_normal.json") as f:
        normal_data = json.load(f)
    with open("app/data/train_fake.json") as f:
        leaked_data = json.load(f)
    return normal_data, leaked_data