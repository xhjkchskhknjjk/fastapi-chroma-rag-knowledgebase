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

# 会话库放在独立的数据目录，不再放在项目根目录。
# 这一点直接决定了容器能否只挂数据目录：旧版把 session.db 写死在 <项目>/session.db，
# 想持久化就只能 -v xxx:/app，而那会把镜像里构建好的代码整个遮住。
DATA_DIR = os.getenv("RAG_DATA_DIR") or os.path.join(BASE_DIR, "data")
# 会话库完整路径优先，其次由数据目录推导
SQLITE_DB_PATH = os.getenv("RAG_SESSION_DB") or os.path.join(DATA_DIR, "session.db")

# 旧版本把会话库放在项目根目录，第一次用新路径启动时自动搬过去，避免历史对话丢失。
# 注意：必须在下面 makedirs 之前执行——否则新位置会先出现一个空库，旧数据就搬不动了。
_LEGACY_SQLITE_PATH = os.path.join(BASE_DIR, "session.db")


def _migrate_legacy_session_db():
    """根目录存在旧会话库、且新位置还没有库时，把旧库移动过去（只做一次）"""
    if os.path.abspath(_LEGACY_SQLITE_PATH) == os.path.abspath(SQLITE_DB_PATH):
        return False
    if not os.path.exists(_LEGACY_SQLITE_PATH) or os.path.exists(SQLITE_DB_PATH):
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)
        os.replace(_LEGACY_SQLITE_PATH, SQLITE_DB_PATH)
        return True
    except OSError:
        # 跨盘或权限问题导致搬不动时不影响启动，只是历史对话不会带过来
        return False


MIGRATED_LEGACY_SESSION_DB = _migrate_legacy_session_db()

# 自动创建目录
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(SQLITE_DB_PATH)), exist_ok=True)

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

# 统一获取向量库实例
# 关键：必须在模块级构造一次并复用，不能每个请求都 new 一个。
#   chromadb 内部有一张按 persist path 索引的全局客户端表
#   (chromadb/api/shared_system_client.py)，并发创建同一路径的客户端会命中
#   "先查后建"竞态，抛 KeyError: '<chroma路径>'。改造前所有请求都跑在事件循环
#   线程里、天然串行，该竞态被掩盖；接口改由线程池执行后才暴露。
#   顺带也省掉了每个请求新建客户端的开销（实测 /doc/list 约 690ms 中的大部分）。
_vector_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embedding_func,
)


def get_vector_db() -> Chroma:
    """返回全局复用的向量库实例"""
    return _vector_db


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
    """读取PDF → 分块 → 过滤参考文献 → 写入Chroma向量库，携带元数据doc_name、doc_unique_id

    :return: (doc_unique_id, 切片数量)
    """
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
    # 没有任何切片时直接返回：把空列表交给 Chroma.add_documents 会抛
    # "Expected Embeddings to be non-empty list"，掩盖真实原因（PDF无可用文本）
    if not split_docs:
        return doc_unique_id, 0
    vector_db = get_vector_db()
    vector_db.add_documents(split_docs)
    # 返回唯一ID与切片数：调用方据此判断"PDF解析出来是空的"
    return doc_unique_id, len(split_docs)

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
