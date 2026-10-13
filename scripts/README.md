# scripts

早期学习/一次性验证脚本，不属于线上服务链路，保留供参考：

- `pdf_reader.py`：用 PyPDF2 提取 PDF 文本的最小示例
- `langchain_pdf.py`：用 PyPDFLoader 分页读取 PDF
- `chunk_vector_store.py`：手工版「切块 → 建向量库 → 相似度检索」验证

线上服务的等价能力已统一收敛到 `utils.py`。
