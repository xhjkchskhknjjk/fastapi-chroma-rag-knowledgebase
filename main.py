from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import requests
import json
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

UPLOAD_DIR = "./uploads"
CHROMA_PERSIST_DIR = "./chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

META_FILE = "./meta.json"

# 初始化元数据文件，兼容文件损坏、空文件
def init_meta():
    try:
        if os.path.exists(META_FILE):
            with open(META_FILE, "r", encoding="utf-8") as f:
                json.load(f)
        else:
            raise FileNotFoundError
    except Exception:
        # 文件不存在 / 空文件 / json格式错误，直接重建
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump({"ingested_files": []}, f, ensure_ascii=False, indent=2)

# 判断该pdf是否已经入库
def is_file_already_ingested(filename: str):
    init_meta()
    with open(META_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return filename in data["ingested_files"]

# 记录已经入库的文件名
def mark_file_ingested(filename: str):
    init_meta()
    with open(META_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if filename not in data["ingested_files"]:
        data["ingested_files"].append(filename)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 全局初始化embedding与向量库
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embedding_func
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]
)

def pdf_to_vectorstore(pdf_file_path: str):
    """传入pdf本地路径，完成加载、切分、写入向量库"""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    vector_db.add_documents(split_docs)
    return len(split_docs)

@app.get("/")
def hello():
    return {"msg": "RAG项目初始化完成，服务运行正常"}

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # 判断文件是否已经入库
        if is_file_already_ingested(file.filename):
            return {"code": 400, "msg": "该PDF文件已经导入向量库，请勿重复上传"}

        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        chunk_count = pdf_to_vectorstore(save_path)
        mark_file_ingested(file.filename)  #标记已入库
        return {
            "filename": file.filename,
            "save_path": save_path,
            "chunk_count": chunk_count,
            "msg": "pdf上传并成功写入向量库"
        }
    except Exception as e:
        return JSONResponse({"code": 500, "msg": "PDF处理失败", "error": str(e)}, status_code=500)

@app.get("/rag/search")
def rag_search(query: str, k: int = 3):
    """检索接口：输入query，返回向量召回片段"""
    docs = vector_db.similarity_search(query, k=k)
    result = [{"page_content": doc.page_content} for doc in docs]
    return {"query": query, "retrieve_result": result}

# ========== 新增 DeepSeek RAG问答接口 ==========
@app.get("/rag/chat")
async def rag_chat(query: str, k: int = 3):
    # 1、向量召回
    try:
        docs = vector_db.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return JSONResponse({"code": 500, "msg": "向量库检索失败", "error": str(e)}, status_code=500)

    # 2、RAG提示词
    prompt = f"""你是文档问答助手，只允许参考下面【参考文档】内容回答用户问题。
如果文档没有相关信息，直接回答“文档中未找到该信息”，不要编造内容。

【参考文档】
{context}

【用户问题】
{query}
"""
    # 读取DeepSeek配置
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL")

    # 校验环境变量
    if not api_key or not base_url or not model_name:
        return JSONResponse({"code": 500, "msg": "LLM环境变量未配置完整"}, status_code=500)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=30)
        res_json = resp.json()
    except requests.exceptions.RequestException as e:
        return JSONResponse({"code": 500, "msg": "调用大模型接口网络异常", "error": str(e)}, status_code=500)

    # 判断大模型是否正常返回choices
    if "choices" not in res_json:
        return JSONResponse({"code": 500, "msg": "大模型返回异常", "response": res_json}, status_code=500)

    answer = res_json["choices"][0]["message"]["content"]

    return JSONResponse({
        "query": query,
        "context": context,
        "answer": answer
    })

@app.delete("/vector/db/clear")
async def clear_vector_db():
    try:
        # 删除向量库文件夹
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        # 清空元数据记录
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump({"ingested_files": []}, f, ensure_ascii=False)
        return {"msg": "向量数据库以及文档元数据已全部清空"}
    except Exception as e:
        return JSONResponse({"code": 500, "msg": "清空向量库失败", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)