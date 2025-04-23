
import yaml
from agent import LLMAgent
from data_manager import DataManager

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    dm = DataManager()
    # https://www.kaggle.com/competitions/titanic
    task_name = "llms-you-cant-please-them-all"
    task = """\
Your task it to predict listening time of a podcast episode.
"""
    agent = LLMAgent(config, task, dm)
    agent.do_task(task_name)
    submit_to_kaggle("agent submition")

    

if __name__ == "__main__":
    main()
