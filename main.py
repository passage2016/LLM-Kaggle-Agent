
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
    task = """
    The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
    """
    agent = LLMAgent(config, task, dm)
    agent.do_task(task_text)
    submit_to_kaggle("agent submition")

    

if __name__ == "__main__":
    main()
