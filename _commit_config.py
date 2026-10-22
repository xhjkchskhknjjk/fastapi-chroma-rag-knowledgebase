"""提交并推送本次配置化改造"""
import os
import subprocess

BASE = r"D:\code\projects\fastapi-chroma-rag-knowledgebase"

MESSAGE = """配置化改造：让 .env 里的配置真正生效，并修复启动期网络阻塞

问题
- DEEPSEEK_URL / DEEPSEEK_MODEL 在代码里写死，.env 中配置了却不生效
  （属于"看起来可配、实际不可配"的陷阱，比没有配置项更危险）
- temperature / timeout / 上传目录 / OpenAPI 版本号同样是硬编码
- DEEPSEEK_API_KEY 缺失时，要等用户提问、走完整条检索链路、
  被大模型返回 401 才能发现
- utils 模块级加载 embedding 模型时会向 huggingface.co 发请求检查更新，
  网络不通时重试 5 次逐个超时，服务启动卡数分钟（模型其实已在本地缓存）

改动
- security.py 集中配置区：DEEPSEEK_URL / MODEL / TEMPERATURE / TIMEOUT、
  UPLOAD_DIR、MAX_UPLOAD_MB、MAX_CONCURRENT_INGEST、INGEST_WAIT_TIMEOUT、
  APP_VERSION 全部改为读取环境变量，main.py 只引用不再赋值
- 启动期配置校验：prod 缺少 DEEPSEEK_API_KEY 或 URL 非法时拒绝启动；
  dev 下记录明确的配置问题与警告
- 启动日志打印生效配置（密钥只输出"是否已配置"的布尔值，不输出原文），
  让"配置到底有没有生效"在日志里可见
- call_llm 增加密钥前置守卫：缺密钥时立即返回可排查的错误，不再发出注定 401 的请求
- 新增 model_config.py：默认把 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 置为 1，
  避免启动时阻塞在与 huggingface.co 的网络检查上；需要检查更新时用 RAG_HF_OFFLINE=0
  （实测启动从数分钟降到约 6 秒）

测试
- 新增 test_config.py（13 个用例）：验证配置值真的进入请求体/请求URL/超时参数，
  prod 缺密钥拒绝启动，环境变量可覆盖，生效配置摘要不泄露密钥
- 增加静态回归守卫：扫描 main.py 源码，禁止出现对配置项的赋值语句
  （仅靠运行时比对无法发现"硬编码值与.env恰好相同"的遮蔽情况）
- 全量用例 60 -> 73，全部通过

文档
- .env.example 补全所有可配置项及缺省值说明
"""

r = subprocess.run(
    ["git", "add", "-A"], cwd=BASE, capture_output=True, text=True, encoding="utf-8"
)
print(r.stdout or r.stderr or "已暂存")

r = subprocess.run(
    ["git", "status", "--short"], cwd=BASE, capture_output=True, text=True, encoding="utf-8"
)
print("=== 暂存内容 ===")
print(r.stdout)

msg_file = os.path.join(BASE, "_msg.txt")
with open(msg_file, "w", encoding="utf-8") as f:
    f.write(MESSAGE)

try:
    r = subprocess.run(
        ["git", "commit", "-F", msg_file], cwd=BASE,
        capture_output=True, text=True, encoding="utf-8",
    )
    print("=== commit ===")
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        raise SystemExit("commit 失败")
finally:
    os.remove(msg_file)

r = subprocess.run(
    ["git", "push", "origin", "main"], cwd=BASE,
    capture_output=True, text=True, encoding="utf-8",
)
print("=== push ===")
print(r.stdout or r.stderr)
if r.returncode != 0:
    raise SystemExit("push 失败")

r = subprocess.run(
    ["git", "log", "--oneline", "-3"], cwd=BASE,
    capture_output=True, text=True, encoding="utf-8",
)
print("=== 最近提交 ===")
print(r.stdout)
