import numpy as np
import torch
import time

# Matrix dimension (N x N)
N = 300

# 1. Pure Python Implementation (Nested Loops)
def pure_python_matmul(A, B):
    C = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

# Initialize data
# We'll use a smaller N for Python if you don't want to wait 10 minutes,
# but for the sake of the exercise, let's assume N=1000.
A_list = np.random.rand(N, N).tolist()
B_list = np.random.rand(N, N).tolist()
A_np = np.array(A_list)
B_np = np.array(B_list)

print(f"Matrix Size: {N}x{N}\n" + "-"*30)

# --- Execution: Pure Python ---
# Note: For N=1000, Python loops can take ~100-200 seconds.
start = time.time()
# Uncomment the line below to actually run the Python version:
_ = pure_python_matmul(A_list, B_list) 
python_time = time.time() - start
print(f"1. Pure Python: ~{python_time:.4f} sec (Estimated)")

# --- Execution: NumPy (CPU) ---
start = time.time()
res_np = np.dot(A_np, B_np)
numpy_time = time.time() - start
print(f"2. NumPy (CPU):  {numpy_time:.4f} sec")

# --- Execution: PyTorch (GPU) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    A_pt = torch.tensor(A_np).to(device)
    B_pt = torch.tensor(B_np).to(device)
    
    # Warm-up (GPU kernels need to initialize)
    _ = torch.matmul(A_pt, B_pt)
    torch.cuda.synchronize() 
    
    start = time.time()
    res_pt = torch.matmul(A_pt, B_pt)
    torch.cuda.synchronize() # Wait for GPU to finish
    pytorch_time = time.time() - start
    print(f"3. PyTorch (GPU): {pytorch_time:.4f} sec")
else:
    pytorch_time = None
    print("3. PyTorch (GPU): CUDA not available.")

# --- Results ---
print("\n" + "Speedup Factor (relative to Python):")
print(f"NumPy Speedup:   {python_time / numpy_time:.2f}x")
if pytorch_time:
    print(f"PyTorch Speedup: {python_time / pytorch_time:.2f}x")