
import yaml
from agent import LLMAgent
from data_manager import DataManager
from kaggle_api import submit_to_kaggle

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    task_name = "store-sales-time-series-forecasting"
    task = """\
In this “getting started” competition, you’ll use time-series forecasting to forecast store sales on data from Corporación Favorita, a large Ecuadorian-based grocery retailer.

Specifically, you'll build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores. You'll practice your machine learning skills with an approachable training dataset of dates, store, and item information, promotions, and unit sales.
"""
    
    dm = DataManager(task_name)
    model_name = ["deepseek-chat", "deepseek-reasoner", "chatgpt-4o-latest", "sft"]
    loss = {}
    device = 3
    for i in range(4):
        agent = LLMAgent(config, task, dm, model_name[i], device)
        result = agent.do_task(task_name)
        loss[result[0]] = result[1]
    max_items = heapq.nlargest(n, my_dict.items(), key=lambda item: item[1])
    max_key = max_items[0][0]
    max_value = max_items[0][1]
    print(f"Model '{max_key}' get max loss {max_value}")
    submit_to_kaggle(task_name, f"./code/{task_name}/{max_key}/submission.csv", "agent submition")

    os.system(f"cd MLAgentBench/ & python -u -m MLAgentBench.runner --python /data/env/mlab/bin/python --task spaceship-titanic --device {device} --log-dir log_titanic  --work-dir workspace --llm-name deepseek-chat --edit-script-llm-name deepseek-chat --fast-llm-name deepseek-chat")

    

if __name__ == "__main__":
    main()
