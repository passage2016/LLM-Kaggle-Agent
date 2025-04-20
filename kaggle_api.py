
import os
import json

def submit_to_kaggle(filename, message):
    os.system(f"kaggle competitions submit -f {filename} -m \"{message}\"")
