import psutil
import GPUtil
import time
import logging


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


class StreamToLogger:
    """
    Stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger_, log_level=logging.INFO):
        self.logger = logger_
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# Setup your logging
def setup_logging(log_file_setup=None):
    # log_file = os.path.join(log_dir, 'log.txt')
    logger_s = logging.getLogger('my_app')
    logger_s.setLevel(logging.INFO)

    # Create handlers for both file and console
    file_handler = logging.FileHandler(log_file_setup, mode='w')
    console_handler = logging.StreamHandler()

    # Optional: add a formatter to include more details
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('- %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger_s.addHandler(file_handler)
    logger_s.addHandler(console_handler)

    return logger_s


# Function to update the learning rate of the optimizer
def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
