import os
import subprocess
from kaggle_analysis import get_task_description

    
def _get_sample_data_text(filename, sampledata):
    return f"{filename}:\n{sampledata}"

class DataManager:
    def __init__(self, task_name):
        self.task = get_task_description(task_name)
        sample_data_dict = {}

        path_to_dir = "./data/"
        os.system(f"rm -rf {path_to_dir}")
        os.system(f"mkdir -p {path_to_dir}")
        subprocess.run(["kaggle", "competitions", "download", "-c", task_name], capture_output=True, text=True, cwd=path_to_dir)
        subprocess.run(["unzip", f"{task_name}.zip"], capture_output=True, text=True, cwd=path_to_dir)
        root, dirs, files = list(os.walk(path_to_dir))[0]
        for file in files:
            if str(file).endswith("csv") or str(file).endswith("tsv"):
                result = subprocess.run(["head", f"./data/{file}"], capture_output=True, text=True)
                sample_data_dict[str(file)] = result.stdout[:200]
            elif str(file).endswith("txt"):
                result = subprocess.run(["cat", f"./data/{file}"], capture_output=True, text=True)
            elif str(file).endswith("zip"):
                pass
            elif str(file).endswith("gz"):
                pass
            else:
                print(file)
                raise Exception("Unknow file type.")
        self.sample_data_dict = sample_data_dict
        
    def get_sample_data_dict(self):
        return self.sample_data_dict
        
    def get_file_count(self):
        return len(self.sample_data_dict)
        
    def get_sample_data_texts(self):
        return "\n".join([_get_sample_data_text(filename, self.sample_data_dict[filename]) for filename  in self.sample_data_dict])