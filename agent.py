
from prompts import load_prompt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from executor import execute_python_code
from data_manager import DataManager

class LLMAgent:
    def __init__(self, config, task, dm):
        self.deepseek_api_key = config['deepseek_api_key']
        self.deepseek_base_url = config['deepseek_base_url']
        self.model_name = config['model_name']
        self.task = task
        self.data_manager = dm
        os.environ["DEEPSEEK_API_KEY"] = self.deepseek_api_key
        os.environ["DEEPSEEK_BASE_URL"] = self.deepseek_base_url
        
        
        
        self.model = ChatDeepSeek(model=self.model_name)
        chain = prompt | model | StrOutputParser()
        
    
    def get_response(prompt_text, system_input = None):
        if not system_input:
            system_input = "You are an AI assistant, please answer user's question."
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_input),
                MessagesPlaceholder("history"),
                ("user", "{input}")
            ]
        )
        return text =  chain.invoke(prompt_text)

    def do_task(self, task_text):
        system_input = f"""\
            You are a Kaggle grandmaster expert in machine learning and data science. Your task is to generate high quality python code for the given task
You are in a jupyter notebook environment, Generate python code for the notebook cell according to the provided task.
Pay attention to previous codes and for new cell continue integrity of code and solution.
There are {self.data_manager.get_file_count()} file in ./data/
<sample_data>
{self.data_manager.get_sample_data_texts()}
</sample_data>
        """
        response = get_response(task_text, system_input)
        code = self.get_code_from_response(response)
        print("Generated Code:", code)
        stdout, stderr = execute_python_code(code)
        print("Output:", stdout)
        print("Errors:", stderr)
        while len(stderr) >0:
            system_input = f"""\
                you are a python code debugger and have expertise in ML code. debug code based on previous code and error message provided.
You are in a jupyter notebook environment, Generate python code for the notebook cell according to the provided task.
Pay attention to previous codes and for new cell continue integrity of code and solution.
            """
            prompt_text = f"""\
                For the code, I got this error, please help me to fix the code
                {stderr}"""
            response = get_response(prompt_text, system_input)
            code = self.get_code_from_response(response)
            print("Generated Code:", code)
            stdout, stderr = execute_python_code(code)
            print("Output:", stdout)
            print("Errors:", stderr)
        with open("./data/result.csv", "w", encoding="utf-8") as f:
            f.write(stdout)
