# GPU Check Script (save as check_gpu.py):
import torch
import sys
import subprocess
import os

def check_nvidia_smi():
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'])
        return nvidia_smi.decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nvidia-smi not found. NVIDIA driver might not be installed."

def check_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"\nCUDA is available: {torch.cuda.is_available()}")
    
    # Check NVIDIA driver
    print("\nNVIDIA Driver Check:")
    print(check_nvidia_smi())
    
    # Check CUDA environment variables
    print("\nCUDA Environment Variables:")
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME: {cuda_home}")
    
    if torch.cuda.is_available():
        print(f"\nCUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB")
            print(f"Memory cached: {torch.cuda.memory_reserved(i)/1e9:.2f} GB")
            props = torch.cuda.get_device_properties(i)
            print(f"Total memory: {props.total_memory/1e9:.2f} GB")
            print(f"GPU capability: {props.major}.{props.minor}")
    else:
        print("\nPossible issues:")
        print("1. PyTorch CPU-only version installed (should end with +cu###)")
        print("2. NVIDIA drivers not installed or outdated")
        print("3. CUDA toolkit not installed or not in PATH")
        print("4. Incompatible CUDA version with PyTorch")
        print("\nTry running: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu()