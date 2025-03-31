import torch

def check_gpu_pytorch():
    # Check if CUDA is available
    is_available = torch.cuda.is_available()
    
    # Print detailed GPU information if available
    if is_available:
        print(torch.cuda.version)
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        
        print(f"GPU is available: {is_available}")
        print(f"Number of GPUs: {device_count}")
        print(f"GPU names: {device_names}")
        
        # Current device information
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device} ({torch.cuda.get_device_name(current_device)})")
    else:
        print("No GPU available with PyTorch")
    
    return is_available

check_gpu_pytorch()