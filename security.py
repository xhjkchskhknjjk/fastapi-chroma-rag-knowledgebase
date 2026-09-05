from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------token鉴权配置----------------------
# 在.env文件添加 API_ACCESS_TOKEN=my_secret_2026
EXPECTED_TOKEN = os.getenv("API_ACCESS_TOKEN", "")

def verify_token(request: Request):
    """从请求头 X-Access-Token 获取token做校验"""
    header_token = request.headers.get("X-Access-Token", "")
    if not EXPECTED_TOKEN:
        # 环境变量未配置token，关闭鉴权，开发调试方便
        return True
    if header_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="无权访问，token错误")
    return True

# ----------------------限流配置 内存版----------------------
limiter = Limiter(key_func=get_remote_address)
RATE_LIMIT = "20/minute"   # 同一个IP每分钟最多20次请求

