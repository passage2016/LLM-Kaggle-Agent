
import os
import json

def submit_to_kaggle(message):
    os.system(f"kaggle competitions submit -f ./data/result.csv -m \"{message}\"")
