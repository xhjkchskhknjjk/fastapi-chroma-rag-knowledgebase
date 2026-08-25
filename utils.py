from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# ---------------- 向量库配置 ----------------
CHROMA_PERSIST_DIR = "./chroma_db"
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

# 内存对话记忆
session_memory = {}

# ---------------- PDF处理工具 ----------------
def process_pdf_to_vector(pdf_file_path: str):
    """读取PDF → 分块 → 写入Chroma向量库"""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_func
    )
    vector_db.add_documents(split_docs)
    return True

def search_vector_db(query: str, k: int = 3):
    """向量检索，返回文档片段列表"""
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_func
    )
    return vector_db.similarity_search(query, k=k)

# ---------------- 对话记忆工具 ----------------
def get_history(session_id: str):
    if session_id not in session_memory:
        session_memory[session_id] = []
    return session_memory[session_id][-4:]

def append_history(session_id: str, user_q: str, bot_a: str):
    if session_id not in session_memory:
        session_memory[session_id] = []
    session_memory[session_id].append({"user": user_q, "assistant": bot_a})
    if len(session_memory[session_id]) > 4:
        session_memory[session_id] = session_memory[session_id][-4:]
