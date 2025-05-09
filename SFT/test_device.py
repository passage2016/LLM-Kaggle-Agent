import torch

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())

if torch.cuda.is_available():
    cur = torch.cuda.current_device()
    print("当前 CUDA device 索引:", cur)
    print("当前 CUDA 设备名称:", torch.cuda.get_device_name(cur))
    # 检查默认张量类型所在设备
    x = torch.tensor([0.])
    print("默认张量设备:", x.device)
    # 检查模型或 Trainer 用的设备
    # 假设你有个 model 或 trainer 对象：
    # print("模型所在设备:", next(model.parameters()).device)
    # 或者 HuggingFace Trainer:
    # print("Trainer 所在设备:", trainer.args.device)
