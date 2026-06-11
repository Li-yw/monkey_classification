import torch

print("=" * 50)
print("PyTorch 版本信息")
print("=" * 50)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cudnn 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("当前使用 CPU")
print("=" * 50)
