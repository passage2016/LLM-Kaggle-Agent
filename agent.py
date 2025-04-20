
from prompts import load_prompt

class LLMAgent:
    def __init__(self, config):
        self.deepseek_api_key = config['deepseek_api_key']
        self.deepseek_base_url = config['deepseek_base_url']
        self.model_name = config['model_name']
        os.environ["DEEPSEEK_API_KEY"] = self.api_key
        os.environ["DEEPSEEK_BASE_URL"] = self.url
        
        
        
        self.model = ChatDeepSeek(model=self.model_name)
        chain = prompt | model | StrOutputParser()
        
    
    def get_response(prompt_text, system_input = None):
        if not system_input:
            system_input = "You are an AI assistant, please answer user's question."
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_input),
                ("user", "{input}")
            ]
        )
        return text =  chain.invoke(prompt_text)

    def do_task(self, task_text):
        system_input = "你是一个数据分析专家，我需要完成一个任务，请帮我将这个任务进行拆解几部分。"
        response = get_response(task_text, system_input)
        # TODO(Ray): 按部分进行拆分，分别输出每部分的代码，并对于每部分的代码
