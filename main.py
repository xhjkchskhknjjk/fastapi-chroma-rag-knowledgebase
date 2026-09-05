from fastapi import FastAPI, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.openapi.models import APIKey
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import os
import requests
from dotenv import load_dotenv

from utils import process_pdf_to_vector, search_vector_db, get_history, append_history, is_pdf_already_exist, get_uploaded_doc_list, delete_document_by_unique_id
from logger import logger
# 这里补上 _rate_limit_exceeded_handler
from security import limiter, RATE_LIMIT, verify_token, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import time


# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
print(f"==== DEEPSEEK_API_KEY:[{DEEPSEEK_API_KEY}]")

# 【关键】先实例化app对象
app = FastAPI(title="RAG知识库服务")

# 注册限流
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==========OpenAPI文档增加token输入框（swagger右上角🔐按钮）==========
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="RAG知识库服务",
        version="1.0.0",
        description="Day13 增加token鉴权+限流",
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

# 白名单接口：docs/openapi.json跳过token校验
WHITE_LIST_PATH = {"/docs","/openapi.json"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path not in WHITE_LIST_PATH:
        try:
            verify_token(request)
        except HTTPException as e:
            logger.warning(f"token校验失败 path={request.url.path}")
            return JSONResponse(status_code=e.status_code, content={"code":e.status_code,"msg":e.detail})
    return await call_next(request)

# 再注册中间件
@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    start_time = time.time()
    logger.info(f"收到请求: method={request.method}, path={request.url.path}")
    response = await call_next(request)
    cost_ms = round((time.time() - start_time)*1000,2)
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

# 请求体校验模型
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="用户提问，1‑500字符")
    k: int = Field(default=3, ge=1, le=10, description="检索片段数量1~10")
    session_id: str = Field(default="default", description="会话ID区分不同用户")

# Deepseek配置
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
def call_llm(prompt: str):
    """调用大模型API，封装在main，属于接口业务逻辑"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
    res_json = resp.json()
    return res_json["choices"][0]["message"]["content"]

# 上传PDF接口
@app.post("/upload")
@limiter.limit(RATE_LIMIT)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    if is_pdf_already_exist(filename):
        logger.info(f"重复上传PDF：{filename}")
        return JSONResponse(status_code=200, content={"code":0,"msg":"该PDF已经上传过，无需重复导入"})
    try:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"开始处理PDF入库：{filename}")
        process_pdf_to_vector(save_path)
        logger.info(f"PDF入库完成：{filename}")
        return {"code":0, "msg":"上传并入库成功", "filename":file.filename}
    except Exception as e:
        logger.error(f"PDF处理失败：{filename}", exc_info=True)
        return JSONResponse(status_code=500, content={"code":500,"msg":"PDF处理失败","error":str(e)})

# 带记忆问答接口
@app.post("/qa/chat")
@limiter.limit(RATE_LIMIT)
async def chat(request: Request, req: ChatRequest):
    try:
        logger.info(f"问答请求，session_id={req.session_id}, query={req.query}")
        history = get_history(req.session_id)
        docs = search_vector_db(req.query, k=req.k)
        context_text = "\n".join([d.page_content for d in docs])
        # 拼接prompt：历史对话 + 参考文档 + 当前问题
        prompt = ""
        for item in history:
            prompt += f"用户：{item['user']}\n助手：{item['assistant']}\n"
        prompt += f"参考文档：{context_text}\n请严格基于参考文档回答用户问题：{req.query}"
        answer = call_llm(prompt)
        append_history(req.session_id, req.query, answer)
        logger.info(f"问答完成，session_id={req.session_id}")
        return {
            "code":0,
            "msg":"ok",
            "data":{
                "answer": answer,
                "retrieve_context": context_text,
                "recent_history": history
            }
        }
    except Exception as e:
        logger.error(f"问答接口异常 session={req.session_id}", exc_info=True)
        return JSONResponse(status_code=500, content={"code":500,"msg":"问答失败","error":str(e)})

@app.get("/doc/list", summary="获取已上传文档列表")
@limiter.limit(RATE_LIMIT)
def list_docs(request: Request):
    docs = get_uploaded_doc_list()
    logger.info(f"查询文档列表，共{len(docs)}条文档")
    return {"code":0, "msg":"ok", "data":docs}

@app.delete("/doc/{doc_unique_id}", summary="删除指定文档以及对应向量")
@limiter.limit(RATE_LIMIT)
def del_doc(request: Request, doc_unique_id: str):
    cnt = delete_document_by_unique_id(doc_unique_id)
    logger.info(f"删除文档 doc_unique_id={doc_unique_id}, 删除向量切片数量={cnt}")
    return {"code":0, "msg":f"已删除 {cnt} 条向量切片"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
