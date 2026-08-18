from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

pdf_path = r"D:\fastapi_rag_project\uploads\info.pdf"
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

# 3.本地embedding，注意这里全部是英文短横线！
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4.初始化Chroma，持久化到磁盘 ./chroma_db
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_func,
    persist_directory="./chroma_db"
)

# 5.检索测试
query = "请查询院系联系电话"
retrieved_docs = vector_db.similarity_search(query,k=3)

print("\n====召回结果====")
for idx,doc in enumerate(retrieved_docs):
    print(f"\n【片段{idx+1}】")
    print(doc.page_content)