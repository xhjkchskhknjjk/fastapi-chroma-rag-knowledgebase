from langchain_community.document_loaders import PyPDFLoader

# 读取同目录test.pdf
loader = PyPDFLoader("test.pdf")
pages = loader.load_and_split()

# 打印全部页面文本
for idx, page in enumerate(pages):
    print(f"===== 第{idx+1}页 =====")
    print(page.page_content)