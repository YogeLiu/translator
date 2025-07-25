import re
import json
from datasets import load_dataset, Dataset
import pandas as pd

# train_file_path = "filtered_data.json"

# raw_data = pd.read_json(train_file_path, lines=True)

# raw_data = raw_data.sample(n=60000)

# # convert to dataset, then put it into huggingface
# dataset = Dataset.from_pandas(raw_data)
# dataset.push_to_hub("YogeLiu/zh-en-translation-dataset-60K")

dataset = load_dataset("YogeLiu/zh-en-translation-dataset-60K", split="train")

# print the max length of the dataset
print(max(len(item["chinese"]) for item in dataset))
print(max(len(item["english"]) for item in dataset))

# print the min length of the dataset
print(min(len(item["chinese"]) for item in dataset))
print(min(len(item["english"]) for item in dataset))

# print the average length of the dataset
print(sum(len(item["chinese"]) for item in dataset) / len(dataset))
print(sum(len(item["english"]) for item in dataset) / len(dataset))
