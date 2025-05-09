import subprocess
import os

def execute_python_code(code, device, filename="temp_exec.py"):
    path = filename.split("/")
    dir_path = "/".join(path[:-1])
    os.system(f"mkdir -p {dir_path}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = device
    result = subprocess.run(["python", path[-1]], capture_output=True, text=True, cwd=dir_path)
    return result.stdout, result.stderr
