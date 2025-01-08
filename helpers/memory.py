import torch
import gc
import time 

def check_memory():
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found.")
        return

    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024/1024/1024:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(i)/1024/1024/1024:.2f} GB")
        print(f"  Total: {torch.cuda.get_device_properties(i).total_memory/1024/1024/1024:.2f} GB")
        print()


def clear_all_cuda_memory():
    # Ensure all CUDA operations are complete
    torch.cuda.synchronize()
    
    # Empty the cache on all devices
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    # Clear references to any tensors and force garbage collection
    gc.collect()
    
    # Optionally, reset the CUDA context (commented out as it's more drastic and may not always be necessary)
    # for device_id in range(torch.cuda.device_count()):
    #     torch.cuda.reset()
        
    print("All CUDA memory cleared on all devices.")


def profile_memory(func, *args, **kwargs):
    """
    Profile peak CUDA memory usage of a torch function.

    Params
    @func: The function to test 
    @args, kwarsg: Arguments to pass to the function
    
    Examples:
        profile_memory(np.diff, [0, 2, 5, 10, 12], n = 1)
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated()
    
    # Run function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    
    return {
        'name': func.__name__,
        'start_memory': f"{start_mem/1e6:.1f}MB",
        'end_memory': f"{end_mem/1e6:.1f}MB", 
        'peak_memory': f"{peak_mem/1e6:.1f}MB",
        'memory_increase': f"{(end_mem - start_mem)/1e6:.1f}MB",
        'time': f"{end_time - start_time:.3f}s"
    }