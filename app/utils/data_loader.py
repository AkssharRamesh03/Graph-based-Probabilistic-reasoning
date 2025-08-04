import json


def load_data():
    with open("app/data/train_normal.json") as f:
        data = json.load(f)
        normal_data = data["data"]
    with open("app/data/train_leaked.json") as f:
        data = json.load(f)
        leaked_data = data["data"]
    return normal_data, leaked_data
