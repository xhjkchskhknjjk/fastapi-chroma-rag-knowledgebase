from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import requests
from dotenv import load_dotenv

# 导入自己写的工具
from utils import process_pdf_to_vector, search_vector_db, get_history, append_history

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
print(f"==== DEEPSEEK_API_KEY:[{DEEPSEEK_API_KEY}]")
app = FastAPI(title="RAG知识库服务")

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 请求体校验模型
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="用户提问，1‑500字符")
    k: int = Field(default=3, ge=1, le=10, description="检索片段数量1~10")
    session_id: str = Field(default="default", description="会话ID区分不同用户")

# Deepseek配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
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
async def upload_pdf(file: UploadFile = File(...)):
    try:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        # 调用工具函数处理入库
        process_pdf_to_vector(save_path)
        return {"code":0, "msg":"上传并入库成功", "filename":file.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"code":500,"msg":"PDF处理失败","error":str(e)})


# 带记忆问答接口
@app.post("/qa/chat")
async def chat(req: ChatRequest):
    try:
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
        return JSONResponse(status_code=500, content={"code":500,"msg":"问答失败","error":str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
