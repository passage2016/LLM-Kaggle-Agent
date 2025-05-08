
import os
import re

from prompts import load_prompt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser

from executor import execute_python_code
from data_manager import DataManager

class LLMAgent:
    def __init__(self, config, task, dm, model_name, device):
        self.deepseek_api_key = config['deepseek_api_key']
        self.deepseek_base_url = config['deepseek_base_url']
        self.gpt_api_key = config['gpt_api_key']
        self.gpt_base_url = config['gpt_base_url']
        self.sft_api_key = config['sft_api_key']
        self.sft_base_url = config['sft_base_url']
        self.model_name = model_name
        self.task = task
        self.data_manager = dm
        if model_name.startswith("deepseek"):
            os.environ["DEEPSEEK_API_KEY"] = self.deepseek_api_key
            os.environ["DEEPSEEK_BASE_URL"] = self.deepseek_base_url
            self.model = ChatDeepSeek(model=self.model_name)\
        elif model_name.startswith("gpt"):
            os.environ["OPENAI_API_KEY"] = self.gpt_api_key
            os.environ["OPENAI_BASE_URL"] = self.gpt_base_url
            self.model = ChatOpenAI(model=self.model_name)
        else:
            self.model = None
        
        
    
    def get_response(self, user_prompt, system_input = None):
        if not system_input:
            system_input = "You are an AI assistant, please answer user's question."
        if self.model = None:
            url = self.sft_base_url

            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.sft_api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "message": prompt,
                "mode": "chat"
            }

            response = requests.post(url, headers=headers, json=data)
            # print(response.json())
            return response.json()["textResponse"]
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_input),
                    ("user", "{input}")
                ]
            )
            chain = prompt | self.model | StrOutputParser()
            return chain.invoke({"input": user_prompt})
    
    def get_code_from_response(self, response):
        print(response)
        pattern = r'```python(.*?)```'
        code_blocks = re.findall(pattern, response, flags=re.DOTALL)
        return [block.strip() for block in code_blocks]

    def get_loss(stdout):
        matches = re.findall(r'loss\s*=\s*([\d.]+)', stdout)
        if matches:
            last_loss = matches[-1]
            return float(last_loss)
        else:
            return -1

    def do_task(self, task_name):
        system_input = f"""\
            You are a Kaggle grandmaster expert in machine learning and data science. Your task is to generate high quality python code for the given task
You are in a jupyter notebook environment, Generate python code for the notebook cell according to the provided task.
Pay attention to previous codes and for new cell continue integrity of code and solution, prefer to use gpu.
There are {self.data_manager.get_file_count()} file in ./data/
<sample_data>
{self.data_manager.get_sample_data_texts()}
</sample_data>
        """
        user_prompt = """\
please write code for this task.
Note : ** Please skip visualization and using plots**
"""
        response = self.get_response(user_prompt, system_input)
        code = "\n".join(self.get_code_from_response(response))
        # print("Generated Code:", code)
        stdout, stderr = execute_python_code(code, self.device, f"./code/{task_name}/{self.model_name}/temp.py")
        # print("Output:", stdout)
        # print("Errors:", stderr)
        while len(stderr) >0:
            system_input = f"""\
                you are a python code debugger and have expertise in ML code. debug code based on previous code and error message provided.
You are in a jupyter notebook environment, Generate python code for the notebook cell according to the provided task.
Pay attention to previous codes and for new cell continue integrity of code and solution, prefer to use gpu.
            """
            prompt_text = f"""\
                this is my code
                {code}
                For the code, I got this error, please help me to fix the code, and return all of the code back
                {stderr}"""
            response = self.get_response(prompt_text, system_input)
            code = "\n".join(self.get_code_from_response(response))
            # print("Generated Code:", code)
            stdout, stderr = execute_python_code(code, self.device, f"./code/{task_name}/{self.model_name}/temp.py")
            # print("Output:", stdout)
            # print("Errors:", stderr)
        with open(f"./data/{self.model_name}_result.csv", "w", encoding="utf-8") as f:
            f.write(stdout)
        return [self.model_name, get_loss(stdout)]
