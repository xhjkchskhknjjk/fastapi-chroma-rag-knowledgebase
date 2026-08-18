import PyPDF2

def extract_pdf_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        # 遍历所有页面提取文字
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

if __name__ == "__main__":
    # 替换成你本地一份PDF文件路径测试
    pdf_content = extract_pdf_text("test.pdf")
    print("PDF提取内容：")
    print(pdf_content)