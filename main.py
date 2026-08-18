from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import requests

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

# 全局初始化embedding与向量库
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embedding_func
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n","\n","。","，"," "]
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
    return {"msg":"RAG项目初始化完成，服务运行正常"}


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        chunk_count = pdf_to_vectorstore(save_path)
        return {
            "filename": file.filename,
            "save_path": save_path,
            "chunk_count": chunk_count,
            "msg":"pdf上传并成功写入向量库"
        }
    except Exception as e:
        return JSONResponse({"code":500,"msg":"PDF处理失败","error":str(e)}, status_code=500)
        



@app.get("/rag/search")
def rag_search(query: str, k: int = 3):
    """检索接口：输入query，返回向量召回片段"""
    docs = vector_db.similarity_search(query, k=k)
    result = [{"page_content":doc.page_content} for doc in docs]
    return {"query":query,"retrieve_result":result}

# ========== 新增 DeepSeek RAG问答接口 ==========
@app.get("/rag/chat")
async def rag_chat(query: str, k: int = 3):
    # 1、向量召回
    try:
        docs = vector_db.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return JSONResponse({"code":500,"msg":"向量库检索失败","error":str(e)}, status_code=500)

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
        return JSONResponse({"code":500,"msg":"LLM环境变量未配置完整"}, status_code=500)

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
        return JSONResponse({"code":500,"msg":"调用大模型接口网络异常","error":str(e)}, status_code=500)

    # 判断大模型是否正常返回choices
    if "choices" not in res_json:
        return JSONResponse({"code":500,"msg":"大模型返回异常","response":res_json}, status_code=500)

    answer = res_json["choices"][0]["message"]["content"]

    return JSONResponse({
        "query": query,
        "context": context,
        "answer": answer
    })