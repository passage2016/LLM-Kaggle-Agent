
import subprocess

def execute_python_code(code, filename="temp_exec.py"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    result = subprocess.run(["python", filename], capture_output=True, text=True)
    return result.stdout, result.stderr
