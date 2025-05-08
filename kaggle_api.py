
import os
import json

def submit_to_kaggle(task_name, filename, message):
    os.system(f"kaggle competitions submit -c {task_name} -f {filename} -m \"{message}\"")
