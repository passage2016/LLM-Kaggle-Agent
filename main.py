
import yaml
from agent import LLMAgent
from executor import execute_python_code

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    agent = LLMAgent(config)

    task_prompt = "Predict house prices using structured data."
    code = agent.do_task(task_prompt)

    print("Generated Code:", code)
    stdout, stderr = execute_python_code(code)
    print("Output:", stdout)
    print("Errors:", stderr)

if __name__ == "__main__":
    main()
