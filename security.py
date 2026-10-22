import hmac
import ipaddress
import os

# 必须最先导入：在 transformers / sentence_transformers 被导入之前设置 HF 离线变量，
# 否则加载 embedding 模型时会向 huggingface.co 发请求并在超时后重试，导致启动卡数分钟
import model_config  # noqa: F401

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


# ----------------------大模型与业务配置----------------------
# 这些项必须真正从环境变量读取：此前 DEEPSEEK_URL / DEEPSEEK_MODEL 在代码里写死，
# 导致 .env 中配置了也不生效，属于"看起来可配、实际不可配"的陷阱。

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv(
    "DEEPSEEK_URL", "https://api.deepseek.com/v1/chat/completions"
).strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# 生成参数：温度与超时此前分别硬编码为 0.1 与 60 秒
try:
    DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1"))
except ValueError:
    DEEPSEEK_TEMPERATURE = 0.1
try:
    DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "60"))
except ValueError:
    DEEPSEEK_TIMEOUT = 60

# 上传目录：此前写死 "./uploads"，导致容器内数据卷必须挂在固定路径才生效
UPLOAD_DIR = os.getenv("RAG_UPLOAD_DIR", "./uploads").strip() or "./uploads"

# 单文件大小上限（MB），缺省 50
try:
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
    if MAX_UPLOAD_MB <= 0:
        MAX_UPLOAD_MB = 50
except ValueError:
    MAX_UPLOAD_MB = 50

# 同时执行解析入库的任务数上限，缺省 4
try:
    MAX_CONCURRENT_INGEST = max(1, int(os.getenv("MAX_CONCURRENT_INGEST", "4")))
except ValueError:
    MAX_CONCURRENT_INGEST = 4

# 等待入库槽位的最长秒数，缺省 60，超时返回 503
try:
    INGEST_WAIT_TIMEOUT = float(os.getenv("RAG_INGEST_WAIT_TIMEOUT", "60"))
    if INGEST_WAIT_TIMEOUT <= 0:
        INGEST_WAIT_TIMEOUT = 60.0
except ValueError:
    INGEST_WAIT_TIMEOUT = 60.0

# OpenAPI 展示用的版本号：此前写死 "1.0.0"，现由环境变量提供
APP_VERSION = os.getenv("RAG_APP_VERSION", "1.0.0").strip() or "1.0.0"

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你是知识库问答助手，请严格依据提供的参考文档回答问题，如果文档没有相关信息如实告知用户。",
).strip()

# 启动期配置校验：把"运行时才发现"的配置错误提前到启动时暴露
CONFIG_STARTUP_PROBLEMS = []
CONFIG_STARTUP_WARNINGS = []

if not DEEPSEEK_API_KEY:
    msg = (
        "未配置 DEEPSEEK_API_KEY：/qa/chat 的所有请求都会失败"
        "（大模型返回401）。请在 .env 或系统环境变量中配置"
    )
    # 生产环境缺大模型密钥直接拒绝启动：核心功能完全不可用
    if IS_PROD:
        raise RuntimeError(msg)
    CONFIG_STARTUP_PROBLEMS.append(msg)

if not DEEPSEEK_URL.startswith(("http://", "https://")):
    msg = f"DEEPSEEK_URL 不是合法的 http(s) 地址: {DEEPSEEK_URL!r}"
    if IS_PROD:
        raise RuntimeError(msg)
    CONFIG_STARTUP_WARNINGS.append(msg)

if not DEEPSEEK_MODEL:
    CONFIG_STARTUP_WARNINGS.append("DEEPSEEK_MODEL 为空，将导致大模型调用失败")

if DEEPSEEK_TIMEOUT <= 0:
    CONFIG_STARTUP_WARNINGS.append(
        f"DEEPSEEK_TIMEOUT={DEEPSEEK_TIMEOUT} 不是正整数，已回退为60"
    )
    DEEPSEEK_TIMEOUT = 60

if not SYSTEM_PROMPT:
    CONFIG_STARTUP_WARNINGS.append("SYSTEM_PROMPT 为空，将不带系统提示词直接提问")

# 供启动日志展示：让"实际生效的配置"在日志里可见，而不是靠猜
EFFECTIVE_CONFIG_SUMMARY = {
    "APP_ENV": APP_ENV,
    "DEEPSEEK_URL": DEEPSEEK_URL,
    "DEEPSEEK_MODEL": DEEPSEEK_MODEL,
    "DEEPSEEK_TEMPERATURE": DEEPSEEK_TEMPERATURE,
    "DEEPSEEK_TIMEOUT": DEEPSEEK_TIMEOUT,
    "UPLOAD_DIR": UPLOAD_DIR,
    "MAX_UPLOAD_MB": MAX_UPLOAD_MB,
    "MAX_CONCURRENT_INGEST": MAX_CONCURRENT_INGEST,
    "API_KEY_CONFIGURED": bool(DEEPSEEK_API_KEY),  # 只标识是否配置，不输出密钥本身
    "API_ACCESS_TOKEN_CONFIGURED": bool(EXPECTED_TOKEN),
}


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
