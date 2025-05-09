
import yaml
import heapq

from agent import LLMAgent
from data_manager import DataManager
from kaggle_api import submit_to_kaggle

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    task_name = "feedback-prize-english-language-learning"
    
    dm = DataManager(task_name)
    model_name = ["deepseek-chat", "deepseek-reasoner", "chatgpt-4o-latest", "sft"]
    loss = {}
    device = 0
    for i in range(len(model_name)):
        print(f"run with model {model_name[i]}")
        agent = LLMAgent(config, task_name, dm, model_name[i], device)
        loss_value = agent.do_task()
        loss[model_name[i]] = loss_value
    max_items = heapq.nlargest(1, loss.items(), key=lambda item: item[1])
    max_key = max_items[0][0]
    max_value = max_items[0][1]
    print(f"Model '{max_key}' get max loss {max_value}")
    submit_to_kaggle(task_name, f"./code/{task_name}/{max_key}/submission.csv", "agent submition")

    os.system(f"cd MLAgentBench/ & python -u -m MLAgentBench.runner --python /data/env/mlab/bin/python --task {task_name} --device {device} --log-dir ./log/log_{task_name}  --work-dir workspace --llm-name deepseek-chat --edit-script-llm-name deepseek-chat --fast-llm-name deepseek-chat")

    

if __name__ == "__main__":
    main()
