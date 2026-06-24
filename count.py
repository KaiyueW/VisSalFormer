import json
import os
from collections import defaultdict

JSON_PATH   = "./data/ChartQA_data/test/test_augmented.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

print(f"Total: {len(data)} items")