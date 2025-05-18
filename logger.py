import os
import sys
from datetime import datetime

def setup_logging(log_dir='logs'):
    """
    设置日志系统，将终端输出同时保存到日志文件
    
    Args:
        log_dir (str): 日志文件保存的目录路径
    
    Returns:
        str: 创建的日志文件路径
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    
    # 创建日志文件
    log_file_handler = open(log_file, 'w')
    
    # 保存原始的标准输出
    original_stdout = sys.stdout
    
    # 定义一个新的输出类，同时输出到终端和文件
    class Logger:
        def __init__(self, file, stdout):
            self.file = file
            self.stdout = stdout
        
        def write(self, message):
            self.file.write(message)
            self.stdout.write(message)
            # 确保实时写入文件
            self.file.flush()
        
        def flush(self):
            self.file.flush()
            self.stdout.flush()
    
    # 替换标准输出为自定义的Logger
    sys.stdout = Logger(log_file_handler, original_stdout)
    
    print(f"日志记录已启动，日志文件: {log_file}")
    return log_file