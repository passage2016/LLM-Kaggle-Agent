import os
import subprocess

    
def _get_sample_data_text(filename, sampledata):
    return f"{filename}:\n{sampledata}"

class DataManager:
    def __init__(self):
        sample_data_dict = {}

        path_to_dir = f"./data/"
        root, dirs, files = list(os.walk(path_to_dir))[0]
        for file in files:
            if str(file).endswith("csv") or str(file).endswith("tsv"):
                result = subprocess.run(["head", f"./data/{file}"], capture_output=True, text=True)
                sample_data_dict[str(file)] = result.stdout[:200]
            elif str(file).endswith("txt"):
                result = subprocess.run(["cat", f"./data/{file}"], capture_output=True, text=True)
            elif str(file).endswith("zip"):
                pass
            else:
                raise Exception("Unknow file type.")
        self.sample_data_dict = sample_data_dict
        
    def get_sample_data_dict(self):
        return self.sample_data_dict
        
    def get_file_count(self):
        return len(self.sample_data_dict)
        
    def get_sample_data_texts(self):
        return "\n".join([_get_sample_data_text(filename, self.sample_data_dict[filename]) for filename  in self.sample_data_dict])