import os
from html.parser import HTMLParser



def get_task_description(task_name):
    os.system("mkdir temp")
    os.system(f"curl -A \"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3\" https://www.kaggle.com/competitions/{task_name} > ./temp/{task_name}.txt")
    f = open(f"./temp/{task_name}.txt")
    lines = list(set(f.readlines()))
    html_text = "\n".join(lines)
    return html_text