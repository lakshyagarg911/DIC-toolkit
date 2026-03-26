# dicUtils/config.py
import json


def load_config(path="configuration.json"):
    with open(path, "r") as f:
        return json.load(f)