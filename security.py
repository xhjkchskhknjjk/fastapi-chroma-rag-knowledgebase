import hmac
import ipaddress
import os

from dotenv import load_dotenv
from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# 先在模块级加载一次.env，保证鉴权配置在import阶段就可读
# 这样无论用 python main.py 还是 uvicorn main:app 启动，都不会漏读.env
load_dotenv()

# ----------------------运行环境判定----------------------
ENV_DEV = "dev"
ENV_PROD = "prod"
APP_ENV = os.getenv("APP_ENV", ENV_DEV).strip().lower()
IS_PROD = APP_ENV == ENV_PROD

# ----------------------token鉴权配置----------------------
# 在.env文件添加 API_ACCESS_TOKEN=my_secret_2026
EXPECTED_TOKEN = os.getenv("API_ACCESS_TOKEN", "").strip()
TOKEN_HEADER = "X-Access-Token"
MIN_TOKEN_LENGTH = 16

# 生产环境强制要求配置token：缺失或强度不足直接让进程启动失败
# 宁可部署时暴露配置错误，也不要线上静默免鉴权
if IS_PROD and len(EXPECTED_TOKEN) < MIN_TOKEN_LENGTH:
    raise RuntimeError(
        f"APP_ENV=prod 时必须配置 API_ACCESS_TOKEN 且长度不少于{MIN_TOKEN_LENGTH}位，"
        "当前未配置或强度不足，请检查.env或系统环境变量"
    )

# 开发环境允许不配token，但仅放行本机回环地址，方便本地调试
DEV_NO_TOKEN = (not IS_PROD) and (not EXPECTED_TOKEN)

# 启动日志文案：只说明当前鉴权策略，不打印token本身
AUTH_STARTUP_WARNING = ""
if DEV_NO_TOKEN:
    AUTH_STARTUP_WARNING = (
        "APP_ENV=dev 且未配置 API_ACCESS_TOKEN：仅本机回环地址(127.0.0.1/::1)免鉴权，"
        "其他来源一律返回401；对外提供服务前请配置 API_ACCESS_TOKEN 并设置 APP_ENV=prod"
    )


def is_loopback_request(request: Request) -> bool:
    """判断请求来源是否为本机回环地址

    注意：这里依赖 request.client.host。若服务部署在反向代理后面，
    需在代理层透传真实IP，否则代理自身地址会被当成本机地址。
    """
    host = request.client.host if request.client else ""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # 无法识别的地址按非本机处理，采取保守策略
        return False


def resolve_client_ip(request: Request) -> str:
    """获取客户端IP，供中间件日志与限流共用，保证取值口径一致"""
    return get_remote_address(request)


def verify_token(request: Request):
    """校验请求头 X-Access-Token

    1) 使用恒定时间比较，避免时序侧信道
    2) 未配置token时仅放行本机回环请求
    3) 失败时不区分“缺失”与“错误”，统一返回401
    """
    if DEV_NO_TOKEN and is_loopback_request(request):
        return True

    if not EXPECTED_TOKEN:
        # 未配置token且来源不是本机，直接拒绝并给出明确提示
        raise HTTPException(
            status_code=401,
            detail="服务端未启用访问令牌，仅允许本机访问，请联系管理员配置 API_ACCESS_TOKEN",
        )

    header_token = request.headers.get(TOKEN_HEADER, "").strip()
    # compare_digest要求两侧同为bytes或同为纯ASCII的str，统一编码为bytes更稳妥
    if not hmac.compare_digest(
        header_token.encode("utf-8"), EXPECTED_TOKEN.encode("utf-8")
    ):
        # 不打印token内容，只记录来源，避免日志泄露与辅助探测
        raise HTTPException(status_code=401, detail="无权访问，token错误")
    return True


# ----------------------限流配置 内存版----------------------
limiter = Limiter(key_func=resolve_client_ip)
# 同一个IP每分钟最多20次请求；可用环境变量覆盖，便于测试与不同环境调参
RATE_LIMIT = os.getenv("RAG_RATE_LIMIT", "20/minute")

# 预留：如需多实例共享限流计数，改这里即可
# limiter = Limiter(key_func=resolve_client_ip, storage_uri="redis://localhost:6379")
