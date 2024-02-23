import torch
import time
import cpuinfo

# https://stackoverflow.com/questions/57238344/i-have-a-gpu-and-cuda-installed-in-windows-10-but-pytorchs-torch-cuda-is-availa
# https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef
# https://www.educative.io/answers/how-to-resolve-torch-not-compiled-with-cuda-enabled
# https://developer.nvidia.com/cuda-toolkit
# https://developer.nvidia.com/cuda-gpus

# Get CPU info
info = cpuinfo.get_cpu_info()

# Print CPU model name
print("CPU Model:", info["brand_raw"])
if torch.cuda.is_available():
    # If a GPU is available, print its model name
    gpu_model = torch.cuda.get_device_name(0)
    print("GPU Model:", gpu_model)
else:
    print("No GPU available.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

matrix_size = 32 * 512
x = torch.randn(matrix_size, matrix_size)
y = torch.randn(matrix_size, matrix_size)

print("CPU SPEED")
start = time.time()
result = torch.matmul(x, y)
print(time.time() - start)
print(f"Verify device: {result.device}")

x_gpu = x.to(device)
y_gpu = y.to(device)
torch.cuda.synchronize()

for i in range(10):
    print("GPU SPEED")
    start = time.time()
    result_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    print(time.time() - start)
    print(f"Verify device: {result_gpu.device}")
