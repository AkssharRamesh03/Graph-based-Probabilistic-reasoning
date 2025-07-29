import json


def load_data():
    with open("app/data/train_normal.json") as f:
        normal_data = json.load(f)
    with open("app/data/train_leaked.json") as f:
        leaked_data = json.load(f)
    return normal_data, leaked_data
