from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 以当前脚本所在位置为基准计算路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, "uploads", "info.pdf")
chroma_dir = os.path.join(BASE_DIR, "chroma_db")

# 校验PDF是否存在
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"找不到pdf文件：{pdf_path}\n请确认info.pdf放在项目uploads文件夹下")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 2.文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n","\n","。","，"," "]
)
split_docs = text_splitter.split_documents(docs)
print(f"切分后一共 {len(split_docs)} 个文本块")

# 3.本地embedding
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4.初始化Chroma持久化
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_func,
    persist_directory=chroma_dir
)

# 5.检索测试
query = "请查询院系联系电话"
retrieved_docs = vector_db.similarity_search(query,k=3)
print("\n====召回结果====")
for idx,doc in enumerate(retrieved_docs):
    print(f"\n【片段{idx+1}】")
    print(doc.page_content)
