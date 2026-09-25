import os
import uuid
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 以当前utils脚本位置为基准，不受启动目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "session.db")

# 自动创建目录
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# ---------------- 向量库配置 ----------------
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

# sqlite初始化会话表
def init_session_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS session_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        user_q TEXT,
        bot_a TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

init_session_db()

# 统一获取向量库实例，全局复用
def get_vector_db() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_func
    )

def filter_reference_text(text: str) -> str:
    """过滤参考文献片段，剔除 [数字]. 开头的参考文献行"""
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        stripped = line.strip()
        # 匹配 [1]. 这类参考文献标记
        if stripped.startswith("[") and stripped.find("].") > 0:
            continue
        out_lines.append(line)
    return "\n".join(out_lines)

# ---------------- PDF处理工具 ----------------
def process_pdf_to_vector(pdf_file_path: str):
    """读取PDF → 分块 → 过滤参考文献 → 写入Chroma向量库，携带元数据doc_name、doc_unique_id"""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    filename = os.path.basename(pdf_file_path)
    doc_unique_id = str(uuid.uuid4())
    # 给每一个切片Document追加元数据 + 文本过滤
    for doc in split_docs:
        doc.page_content = filter_reference_text(doc.page_content)
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

# ---------------- 对话记忆工具 sqlite持久化 ----------------
def get_history(session_id: str):
    """读取会话历史，只取最近4轮"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        SELECT user_q, bot_a FROM session_history
        WHERE session_id=?
        ORDER BY id DESC LIMIT 4
    ''', (session_id,))
    rows = cur.fetchall()
    conn.close()
    # 倒序恢复时间顺序
    rows.reverse()
    result = []
    for r in rows:
        result.append({"user": r[0], "assistant": r[1]})
    return result

def append_history(session_id: str, user_q: str, bot_a: str):
    """新增一轮对话，数据库只保留最近4轮，旧数据删除"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    # 插入新记录
    cur.execute('''
        INSERT INTO session_history(session_id, user_q, bot_a) VALUES (?,?,?)
    ''', (session_id, user_q, bot_a))
    # 查询该会话总条数
    cur.execute('''
        SELECT id FROM session_history WHERE session_id=? ORDER BY id DESC
    ''', (session_id,))
    all_ids = [row[0] for row in cur.fetchall()]
    # 如果大于4，删除更早的记录
    if len(all_ids) > 4:
        need_del = all_ids[4:]
        placeholders = ",".join(["?"]*len(need_del))
        cur.execute(f"DELETE FROM session_history WHERE id IN ({placeholders})", need_del)
    conn.commit()
    conn.close()
