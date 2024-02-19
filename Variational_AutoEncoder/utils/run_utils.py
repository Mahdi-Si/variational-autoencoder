import psutil
import GPUtil
import time

def log_resource_usage():
    # CPU and RAM usage
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    ram_percent = memory.percent

    # GPU usage - assuming you have NVIDIA GPUs
    gpu_logs = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpu_name = gpu.name
        gpu_load = f"{gpu.load*100}%"
        gpu_memory_util = f"{gpu.memoryUtil*100}%"
        gpu_logs.append((gpu_name, gpu_load, gpu_memory_util))

    # Construct a log message
    log_message = f"CPU Usage: {cpu_percent}%, RAM Usage: {ram_percent}%, " + ", ".join([f"{name} Load: {load}, Memory Utilization: {memory_util}" for name, load, memory_util in gpu_logs])

    # Print or save log message
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {log_message}\n")
