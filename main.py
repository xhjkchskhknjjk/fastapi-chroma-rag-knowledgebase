from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import os
import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv
from utils import process_pdf_to_vector, search_vector_db, get_history, append_history, is_pdf_already_exist, get_uploaded_doc_list, delete_document_by_unique_id
from logger import logger
from file_validation import (
    UploadValidationError,
    resolve_upload_path,
    sanitize_filename,
    validate_content_length,
    validate_pdf_content,
)
from security import (
    limiter,
    RATE_LIMIT,
    verify_token,
    _rate_limit_exceeded_handler,
    is_loopback_request,
    resolve_client_ip,
    AUTH_STARTUP_WARNING,
)
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool
import threading
import time

# 加载环境变量：本地开发读取.env；docker环境依靠系统环境变量，找不到文件不会报错
# 注意：security.py在导入时已自行加载一次，保证APP_ENV与API_ACCESS_TOKEN在启动校验前可读
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "你是知识库问答助手，请严格依据提供的参考文档回答问题，如果文档没有相关信息如实告知用户。")
APP_DESCRIPTION_VERSION = "1.0.0"

# 实例化app对象
app = FastAPI(title="RAG知识库服务")
# 注册限流
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 鉴权策略启动提示：token未配置这类风险必须在启动时就可见
if AUTH_STARTUP_WARNING:
    logger.warning(AUTH_STARTUP_WARNING)
else:
    logger.info("鉴权已启用：除文档接口外，所有请求需携带请求头 X-Access-Token")

# ==========OpenAPI文档增加token输入框（swagger右上角🔐按钮）==========
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RAG知识库服务",
        version=APP_DESCRIPTION_VERSION,
        description=(
            "RAG知识库问答服务。\n\n"
            "**鉴权**：除 `/docs`、`/openapi.json` 外，所有接口需在请求头携带 `X-Access-Token`，"
            "取值为服务端环境变量 `API_ACCESS_TOKEN`。"
        ),
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Access-Token"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# 白名单接口：文档与OpenAPI描述跳过token校验，其余接口一律需要token
# 说明：/redoc 是FastAPI默认开启的文档页，不纳入白名单会返回一坨401 JSON而不是页面
WHITE_LIST_PATH = {
    "/docs",
    "/docs/",
    "/docs/oauth2-redirect",
    "/redoc",
    "/redoc/",
    "/openapi.json",
}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # CORS预检请求不携带自定义头，交给后续CORS中间件处理，否则浏览器请求会先被401拦掉
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path not in WHITE_LIST_PATH:
        try:
            verify_token(request)
        except HTTPException as e:
            # 只记录来源IP与路径，不记录token内容，避免日志泄露与辅助探测
            client_ip = resolve_client_ip(request)
            extra = "（来源为本机回环地址）" if is_loopback_request(request) else ""
            logger.warning(
                f"token校验失败 path={request.url.path} client_ip={client_ip}{extra}"
            )
            return JSONResponse(
                status_code=e.status_code,
                content={"code": e.status_code, "msg": e.detail, "data": None},
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return await call_next(request)

@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    start_time = time.time()
    logger.info(f"收到请求: method={request.method}, path={request.url.path}")
    response = await call_next(request)
    cost_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"请求完成: path={request.url.path}, status={response.status_code}, cost={cost_ms}ms")
    return response

# 全局异常处理器，增加日志落盘
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局捕获异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "msg": f"服务异常：{str(exc)}",
            "data": None
        }
    )

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 单文件大小上限，可用环境变量覆盖，默认50MB
try:
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
except ValueError:
    MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# 读请求体的分块大小；此循环运行在事件循环里，块取大一些以减少 await 次数
UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024

# 同时执行"解析入库"的上限。PDF解析+embedding是CPU密集操作，
# 并发过高会把默认线程池(40)占满，连带让 /doc/list 等接口一起排队
try:
    MAX_CONCURRENT_INGEST = max(1, int(os.getenv("MAX_CONCURRENT_INGEST", "4")))
except ValueError:
    MAX_CONCURRENT_INGEST = 4
INGEST_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_INGEST)
# 等待入库槽位的最长秒数，超时返回503而不是无限占用请求
INGEST_WAIT_TIMEOUT = float(os.getenv("INGEST_WAIT_TIMEOUT", "60"))

# 同一个embedding模型会被多个线程并发调用：实测 torch 2.13 CPU 版并发4次调用正常
# （intra-op 16线程，总耗时与单次相同），故模型侧无需再加锁

# 请求体校验模型
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="用户提问，1‑500字符")
    k: int = Field(default=3, ge=1, le=10, description="检索片段数量1~10")
    session_id: str = Field(default="default", description="会话ID区分不同用户")

# Deepseek配置
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

def call_llm(messages: list):
    """
    调用大模型API
    :param messages: openai格式消息数组 [{"role":"system","content":"..."}, ...]
    :return: (content:str|None, err_msg:str)
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.1
    }
    try:
        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
        res_json = resp.json()
        if resp.status_code != 200:
            err = f"大模型http状态码异常:{resp.status_code}, detail:{res_json.get('error','')}"
            logger.error(err)
            return None, err
        if "choices" not in res_json or len(res_json["choices"]) == 0:
            err = f"大模型返回缺少choices字段，resp_body:{res_json}"
            logger.error(err)
            return None, err
        content = res_json["choices"][0]["message"]["content"].strip()
        return content, ""
    except RequestException as e:
        err = f"大模型网络请求异常：{str(e)}"
        logger.error(err, exc_info=True)
        return None, err
    except Exception as e:
        err = f"大模型调用未知异常：{str(e)}"
        logger.error(err, exc_info=True)
        return None, err

# 上传PDF接口
@app.post("/upload")
@limiter.limit(RATE_LIMIT)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """上传PDF并入库

    安全校验顺序：文件名消毒与扩展名 -> 分块落盘并限制大小 -> 文件头魔数校验 -> 入库。
    任何一步失败都会清理已写入的临时文件，校验不通过的请求不返回500。
    """
    filename = file.filename

    # 0) 先按 Content-Length 拦掉明显超限的请求，再进入文件名与内容校验
    try:
        validate_content_length(request.headers.get("content-length"), MAX_UPLOAD_BYTES)
    except UploadValidationError as e:
        logger.warning(f"上传被拒（请求体超限）：filename={filename!r}, 原因={e}")
        return JSONResponse(
            status_code=e.status_code,
            content={"code": e.status_code, "msg": str(e), "data": None},
        )

    # 1) 文件名消毒 + 扩展名白名单（同时防路径穿越）
    try:
        safe_name = sanitize_filename(filename)
    except UploadValidationError as e:
        logger.warning(f"上传被拒（文件名不合法）：原始文件名={filename!r}, 原因={e}")
        return JSONResponse(
            status_code=e.status_code,
            content={"code": e.status_code, "msg": str(e), "data": None},
        )

    # 2) 去重：用消毒后的名字，和写入向量库的 doc_name 保持同一口径
    #    查 Chroma 是同步阻塞调用（实测可达数百毫秒），同样不能跑在事件循环里
    if await run_in_threadpool(is_pdf_already_exist, safe_name):
        logger.info(f"重复上传PDF：{safe_name}")
        return JSONResponse(status_code=200, content={"code": 0, "msg": "该PDF已经上传过，无需重复导入"})

    save_path = resolve_upload_path(UPLOAD_DIR, safe_name)
    written = 0
    head_buffer = b""
    try:
        # 3) 分块落盘，边写边校验大小；同时在内存里留下文件头用于类型校验。
        #    刻意不"写完后重新打开读首块"：Windows 上杀软实时防护会短暂锁住新写入的文件，
        #    此时重新 open 会拿到 [Errno 22]/[Errno 13]，把一个本该400的校验失败变成500。
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise UploadValidationError(
                        f"文件超过大小上限 {MAX_UPLOAD_MB}MB，请压缩或拆分后再上传",
                        status_code=413,
                    )
                if len(head_buffer) < 1024:
                    head_buffer += chunk[: 1024 - len(head_buffer)]
                f.write(chunk)

        # 4) 内容校验：必须是真正的PDF，只改后缀的伪装文件在这里被挡下
        validate_pdf_content(head_buffer)

        logger.info(f"开始处理PDF入库：{safe_name}（{written} 字节）")
        # 5) PDF解析 + 切块 + embedding 是CPU密集操作，必须丢到线程池：
        #    否则会把事件循环卡住数十秒，期间 /doc/list、/docs 等全部无响应。
        #    同时用信号量限制并发入库数量，避免解析任务占满线程池。
        doc_unique_id, chunk_count = await run_in_threadpool(
            _ingest_with_limit, save_path
        )
        if chunk_count == 0:
            # 扫描件/纯图片PDF会走到这里：文件不该留下，否则重传会被误判成"已上传过"
            raise UploadValidationError(
                "该PDF没有解析出任何文本，可能是扫描件或纯图片，暂不支持"
            )
        logger.info(f"PDF入库完成：{safe_name}，切片数={chunk_count}")
        return {
            "code": 0,
            "msg": "上传并入库成功",
            "filename": safe_name,
            "doc_unique_id": doc_unique_id,
            "chunk_count": chunk_count,
        }
    except UploadValidationError as e:
        _remove_failed_upload(save_path)
        logger.warning(f"上传被拒（校验不通过）：filename={safe_name}, 原因={e}")
        return JSONResponse(
            status_code=e.status_code,
            content={"code": e.status_code, "msg": str(e), "data": None},
        )
    except Exception as e:
        _remove_failed_upload(save_path)
        logger.error(f"PDF处理失败：{safe_name}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "msg": "PDF处理失败", "error": str(e)},
        )


def _ingest_with_limit(save_path: str):
    """在线程池线程里执行入库，并用信号量限制并发数量

    acquire 带超时：入库是长任务（大PDF可达数十秒），
    长时间等不到槽位应快速失败，而不是把线程池线程一直占着。
    """
    if not INGEST_SEMAPHORE.acquire(timeout=INGEST_WAIT_TIMEOUT):
        raise UploadValidationError(
            f"当前入库任务较多（并发上限 {MAX_CONCURRENT_INGEST}），请稍后重试",
            status_code=503,
        )
    try:
        return process_pdf_to_vector(save_path)
    finally:
        INGEST_SEMAPHORE.release()


def _remove_failed_upload(path: str):
    """清理上传失败留下的半个文件

    清理失败不影响对外的拒绝结果：Windows 上杀软实时防护会短暂锁住刚写入的可疑文件，
    此时删除会失败。文件已不会进入向量库，也不参与去重（去重看的是向量库元数据），
    因此记一条日志即可，不必让请求变成500。
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"清理失败的上传文件未能删除（不影响本次拒绝结果）path={path}: {e}")


# 带记忆问答接口
# 刻意用 def 而非 async def：
#   本接口全程是同步阻塞调用（向量检索 + requests.post 调大模型 + sqlite 读写），
#   async def 会让这些操作跑在事件循环线程里，一个慢请求就冻结整个服务。
#   用 def 后 FastAPI 自动把它放进线程池，事件循环保持可调度。
#   注意 slowapi 对同步函数的包装器不 await，因此限流照常生效。
@app.post("/qa/chat")
@limiter.limit(RATE_LIMIT)
def chat(request: Request, req: ChatRequest):
    try:
        logger.info(f"问答请求，session_id={req.session_id}, query={req.query}")
        history = get_history(req.session_id)
        docs = search_vector_db(req.query, k=req.k)
        context_text = "\n".join([d.page_content for d in docs])

        messages = []
        # system提示词
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        # 加载历史对话
        for item in history:
            messages.append({"role": "user", "content": item["user"]})
            messages.append({"role": "assistant", "content": item["assistant"]})
        # 当前用户问题，附带检索上下文
        user_content = f"参考文档：{context_text}\n用户问题：{req.query}"
        messages.append({"role": "user", "content": user_content})

        answer, llm_err = call_llm(messages)
        if llm_err:
            logger.error(f"LLM调用失败: {llm_err}")
            return JSONResponse(status_code=500, content={"code": 500, "msg": llm_err, "data": None})
        append_history(req.session_id, req.query, answer)
        logger.info(f"问答完成，session_id={req.session_id}")
        return {
            "code": 0,
            "msg": "ok",
            "data": {
                "answer": answer,
                "retrieve_context": context_text,
                "recent_history": history
            }
        }
    except Exception as e:
        logger.error(f"问答接口异常 session={req.session_id}", exc_info=True)
        return JSONResponse(status_code=500, content={"code": 500, "msg": "问答失败", "error": str(e)})

@app.get("/doc/list", summary="获取已上传文档列表")
@limiter.limit(RATE_LIMIT)
def list_docs(request: Request):
    docs = get_uploaded_doc_list()
    logger.info(f"查询文档列表，共{len(docs)}条文档")
    return {"code": 0, "msg": "ok", "data": docs}

@app.delete("/doc/{doc_unique_id}", summary="删除指定文档以及对应向量")
@limiter.limit(RATE_LIMIT)
def del_doc(request: Request, doc_unique_id: str):
    cnt = delete_document_by_unique_id(doc_unique_id)
    logger.info(f"删除文档 doc_unique_id={doc_unique_id}, 删除向量切片数量={cnt}")
    return {"code": 0, "msg": f"已删除 {cnt} 条向量切片"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
