import os
import logging
from logging.handlers import RotatingFileHandler

# 自动创建logs目录
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger():
    logger = logging.getLogger("fastapi_rag")
    logger.setLevel(logging.INFO)
    # 避免重复handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # 文件输出：单个文件最大10MB，最多保留5个轮转日志
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding="utf‑8"
    )
    file_handler.setFormatter(formatter)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()
