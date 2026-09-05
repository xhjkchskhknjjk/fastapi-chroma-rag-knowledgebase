import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 以当前utils脚本位置为基准，不受启动目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# 自动创建目录
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


# ---------------- 向量库配置 ----------------
CHROMA_PERSIST_DIR = "./chroma_db"
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
# 内存对话记忆
session_memory = {}

# 统一获取向量库实例，全局复用
def get_vector_db() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_func
    )

# ---------------- PDF处理工具 ----------------
def process_pdf_to_vector(pdf_file_path: str):
    """读取PDF → 分块 → 写入Chroma向量库，携带元数据doc_name、doc_unique_id"""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    filename = os.path.basename(pdf_file_path)
    doc_unique_id = str(uuid.uuid4())

    # 给每一个切片Document追加元数据
    for doc in split_docs:
        doc.metadata["doc_name"] = filename
        doc.metadata["doc_unique_id"] = doc_unique_id

    vector_db = get_vector_db()
    vector_db.add_documents(split_docs)
    return True


def search_vector_db(query: str, k: int = 3):
    """向量检索，返回文档片段列表"""
    vector_db = get_vector_db()
    return vector_db.similarity_search(query, k=k)

# ========== Day11 新增工具函数 ==========
def is_pdf_already_exist(file_name: str) -> bool:
    """判断该文件名PDF是否已经入库向量库，用于上传接口去重"""
    vector_db = get_vector_db()
    res = vector_db.get(where={"doc_name": file_name})
    return len(res["ids"]) > 0

def get_uploaded_doc_list():
    """获取全部已上传文档清单，去重返回 [{doc_unique_id, doc_name}]"""
    vector_db = get_vector_db()
    all_data = vector_db.get()
    doc_set = {}
    for meta in all_data["metadatas"]:
        if meta is None:
            continue
        uid = meta["doc_unique_id"]
        name = meta["doc_name"]
        doc_set[uid] = {"doc_unique_id": uid, "doc_name": name}
    return list(doc_set.values())

def delete_document_by_unique_id(doc_unique_id: str):
    """根据doc_unique_id删除该文档全部向量切片"""
    vector_db = get_vector_db()
    res = vector_db.get(where={"doc_unique_id": doc_unique_id})
    delete_ids = res.get("ids", [])
    # 处理ids为None的边界
    if delete_ids is None:
        delete_ids = []
    if len(delete_ids) > 0:
        vector_db.delete(ids=delete_ids)
    return len(delete_ids)


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
