import subprocess
import ones

def execute_python_code(code, device, filename="temp_exec.py"):
    path = filename.split("/")
    dir_path = "/".join(path[:-1])
    os.system(f"mkdir -p {dir_path}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    subprocess.run([f"cd", dir_path], capture_output=True, text=True)
    result = subprocess.run([f"CUDA_VISIBLE_DEVICES={device}", "python", filename], capture_output=True, text=True)
    return result.stdout, result.stderr
