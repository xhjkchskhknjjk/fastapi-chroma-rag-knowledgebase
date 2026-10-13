"""上传接口安全校验用例

直接运行： python test_upload.py
覆盖：文件名消毒（路径穿越/保留名/超长/扩展名）、内容魔数校验、大小上限、
      空文件、失败清理、合法PDF可正常入库、被拒请求不污染向量库
"""
import contextlib
import io
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import _testconfig  # noqa: F401,E402  必须最先导入：在 security 之前注入测试配置

TEST_TOKEN = _testconfig.TEST_TOKEN


def _install_offline_stub():
    """顶掉真实 embedding 模型，避免单测联网拉模型"""
    import types

    if "sentence_transformers" in sys.modules:
        return
    from langchain_core.embeddings import Embeddings

    fake = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer(Embeddings):
        def __init__(self, model_name=None, *args, **kwargs):
            self._dim = 384

        def encode(self, sentences, **kwargs):
            """HuggingFaceEmbeddings 内部调 client.encode()，返回值需为numpy数组"""
            import numpy as np

            if isinstance(sentences, str):
                return np.zeros(self._dim, dtype="float32")
            return np.zeros((len(sentences), self._dim), dtype="float32")

        def embed_documents(self, texts):
            return [[0.0] * self._dim for _ in texts]

        def embed_query(self, text):
            return [0.0] * self._dim

    fake.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake


_install_offline_stub()

from file_validation import (  # noqa: E402
    UploadValidationError,
    resolve_upload_path,
    sanitize_filename,
    validate_content_length,
    validate_pdf_content,
)

MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
    b"%%EOF\n"
)


def build_pdf(text: str | None) -> bytes:
    """用 pypdf 底层对象构造真实可解析的PDF

    text=None 时不写内容流 -> 合法但抽不出文本（模拟扫描件/纯图片）；
    注意必须显式登记字体资源，否则内容流引用的 /F1 不存在，文本会抽成空串。
    """
    import io as _io

    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )

    if text is not None:
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 14 Tf 20 150 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)

    buf = _io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestValidateContentLength(unittest.TestCase):
    """Content-Length 预检"""

    def test_missing_header_allowed(self):
        validate_content_length(None, 1024)

    def test_within_limit_allowed(self):
        validate_content_length("500", 1024)

    def test_over_limit_rejected_with_413(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_content_length(str(10 * 1024 * 1024), 1024)
        self.assertEqual(ctx.exception.status_code, 413)

    def test_non_numeric_rejected(self):
        with self.assertRaises(UploadValidationError):
            validate_content_length("abc", 1024)

    def test_zero_and_negative_ignored(self):
        validate_content_length("0", 1024)
        validate_content_length("-5", 1024)


class TestSanitizeFilename(unittest.TestCase):
    """文件名消毒"""

    def test_normal_names_kept(self):
        for name in (
            "导师简介.pdf",
            "report-2026.pdf",
            "2025导师简历合并 .pdf",
            "a b(c)[d].pdf",
        ):
            with self.subTest(name=name):
                out = sanitize_filename(name)
                self.assertTrue(out.endswith(".pdf"))
                self.assertNotIn("/", out)
                self.assertNotIn("\\", out)

    def test_uppercase_extension_normalized(self):
        self.assertTrue(sanitize_filename("Report.PDF").endswith(".pdf"))

    def test_posix_path_traversal_stripped(self):
        """../../ 这类穿越必须被剥掉，只剩文件名"""
        out = sanitize_filename("../../../../windows/win.ini.pdf")
        self.assertEqual(out, "win.ini.pdf")
        self.assertNotIn("..", out)
        self.assertNotIn("/", out)

    def test_windows_path_traversal_stripped(self):
        out = sanitize_filename("..\\..\\evil.pdf")
        self.assertEqual(out, "evil.pdf")
        self.assertNotIn("\\", out)

    def test_absolute_windows_path_stripped(self):
        out = sanitize_filename("C:\\Windows\\System32\\evil.pdf")
        self.assertEqual(out, "evil.pdf")

    def test_null_byte_removed(self):
        out = sanitize_filename("bad\x00name.pdf")
        self.assertNotIn("\x00", out)
        self.assertTrue(out.endswith(".pdf"))

    def test_leading_dot_removed(self):
        """不能生成隐藏文件"""
        self.assertFalse(sanitize_filename("...hidden.pdf").startswith("."))

    def test_non_pdf_extension_rejected(self):
        for name in ("shell.php", "x.exe", "note.txt", "archive.pdf.exe", "noext"):
            with self.subTest(name=name):
                with self.assertRaises(UploadValidationError):
                    sanitize_filename(name)

    def test_double_extension_rejected(self):
        """a.pdf.exe 的扩展名是 .exe，必须拒绝"""
        with self.assertRaises(UploadValidationError):
            sanitize_filename("a.pdf.exe")

    def test_windows_reserved_names_rejected(self):
        for name in ("CON.pdf", "nul.pdf", "COM1.pdf", "LPT9.pdf"):
            with self.subTest(name=name):
                with self.assertRaises(UploadValidationError):
                    sanitize_filename(name)

    def test_empty_and_none_rejected(self):
        for name in (None, "", "   "):
            with self.subTest(name=name):
                with self.assertRaises(UploadValidationError):
                    sanitize_filename(name)

    def test_only_illegal_chars_rejected(self):
        with self.assertRaises(UploadValidationError):
            sanitize_filename("../..pdf")

    def test_long_name_truncated_and_unique(self):
        long_name = "很长的名字" * 60 + ".pdf"
        out1 = sanitize_filename(long_name)
        out2 = sanitize_filename(long_name)
        self.assertLessEqual(len(out1), 90)
        self.assertTrue(out1.endswith(".pdf"))
        # 截断后带随机后缀，避免同名不同文件互相覆盖
        self.assertNotEqual(out1, out2)


class TestValidatePdfContent(unittest.TestCase):
    """内容魔数校验"""

    def test_real_pdf_header_accepted(self):
        validate_pdf_content(MINIMAL_PDF)

    def test_pdf_with_leading_bom_accepted(self):
        validate_pdf_content(b"\xef\xbb\xbf%PDF-1.7\n")

    def test_pdf_with_leading_whitespace_accepted(self):
        validate_pdf_content(b"\r\n %PDF-1.4\n")

    def test_fake_pdf_rejected(self):
        """只把后缀改成 .pdf 的文件必须被挡下"""
        for payload in (b"<?php system($_GET['c']); ?>", b"MZ\x90\x00", b"PK\x03\x04", b"<html></html>"):
            with self.subTest(payload=payload[:8]):
                with self.assertRaises(UploadValidationError):
                    validate_pdf_content(payload)

    def test_empty_content_rejected(self):
        with self.assertRaises(UploadValidationError):
            validate_pdf_content(b"")


class TestResolveUploadPath(unittest.TestCase):
    """落盘路径兜底断言"""

    def test_normal_name_inside_upload_dir(self):
        path = resolve_upload_path("./uploads", "a.pdf")
        self.assertEqual(os.path.dirname(path), os.path.realpath("./uploads"))

    def test_traversal_attempt_raises(self):
        for bad in ("../evil.pdf", "..\\evil.pdf", "sub/evil.pdf"):
            with self.subTest(bad=bad):
                with self.assertRaises(UploadValidationError):
                    resolve_upload_path("./uploads", bad)


class TestUploadEndpoint(unittest.TestCase):
    """上传接口端到端行为"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import main
        import utils

        cls.main = main
        cls.utils = utils
        cls.client = TestClient(main.app, client=("127.0.0.1", 9000))
        cls.headers = {"X-Access-Token": TEST_TOKEN}

    def _upload(self, name, content, content_type="application/pdf"):
        return self.client.post(
            "/upload",
            headers=self.headers,
            files={"file": (name, io.BytesIO(content), content_type)},
        )

    def _doc_count(self):
        """直接从向量库统计切片数，用于判断是否被污染"""
        db = self.utils.get_vector_db()
        return len(db.get()["ids"])

    def test_fake_pdf_with_pdf_extension_rejected(self):
        before = self._doc_count()
        r = self._upload("malicious.pdf", b"<?php echo 1; ?>")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("PDF", r.json()["msg"])
        self.assertEqual(self._doc_count(), before, "被拒请求不应写入向量库")

    def test_non_pdf_extension_rejected(self):
        before = self._doc_count()
        r = self._upload("shell.php", b"<?php echo 1; ?>", content_type="application/x-php")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(self._doc_count(), before)

    def test_pdf_extension_with_non_pdf_content_rejected(self):
        """只改后缀的伪装文件必须返回400，而不是被兜成500"""
        before = self._doc_count()
        r = self._upload("fake.pdf", b"<?php system($_GET['c']); ?>")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("PDF", r.json()["msg"])
        self.assertEqual(self._doc_count(), before)

    def test_path_traversal_name_does_not_escape_upload_dir(self):
        before = self._doc_count()
        r = self._upload("../../evil.pdf", b"not a pdf at all")
        self.assertEqual(r.status_code, 400, r.text)
        # 关键断言：上传目录之外不能被写入任何文件
        self.assertFalse(os.path.exists(os.path.join(BASE_DIR, "..", "..", "evil.pdf")))
        self.assertEqual(self._doc_count(), before)

    def test_traversal_name_with_valid_pdf_is_sanitized_and_contained(self):
        """即使内容合法，穿越文件名也只能落在 uploads 目录内

        用结构完整的PDF，避免 MINIMAL_PDF 那种"魔数合法但无法解析"的样本
        触发 process_pdf_to_vector 的异常分支，把堆栈打到测试输出里。
        """
        r = self._upload("../../contained.pdf", build_pdf("traversal containment check"))
        self.assertEqual(r.status_code, 200, r.text)
        # 关键断言：uploads 目录之外不能出现该文件
        self.assertFalse(os.path.exists(os.path.join(BASE_DIR, "..", "..", "contained.pdf")))
        inside = os.path.join(BASE_DIR, "uploads", "contained.pdf")
        try:
            self.assertTrue(os.path.exists(inside))
            self.assertEqual(r.json()["filename"], "contained.pdf")
        finally:
            db = self.utils.get_vector_db()
            res = db.get(where={"doc_name": "contained.pdf"})
            if res["ids"]:
                db.delete(ids=res["ids"])
            if os.path.exists(inside):
                os.remove(inside)

    def test_empty_file_rejected(self):
        before = self._doc_count()
        r = self._upload("empty.pdf", b"")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("空", r.json()["msg"])
        self.assertEqual(self._doc_count(), before)

    def test_oversized_file_rejected_with_413(self):
        before = self._doc_count()
        old_limit = self.main.MAX_UPLOAD_BYTES
        self.main.MAX_UPLOAD_BYTES = 1024  # 临时把上限压到1KB
        try:
            r = self._upload("big.pdf", MINIMAL_PDF + b"x" * 4096)
            self.assertEqual(r.status_code, 413, r.text)
            self.assertIn("大小上限", r.json()["msg"])
        finally:
            self.main.MAX_UPLOAD_BYTES = old_limit
        self.assertEqual(self._doc_count(), before)

    def test_rejected_upload_leaves_no_file_on_disk(self):
        """校验失败必须清理掉已写入的半个文件"""
        for name, content in (("bad1.pdf", b"not pdf"), ("bad2.pdf", b"")):
            with self.subTest(name=name):
                self._upload(name, content)
                self.assertFalse(os.path.exists(os.path.join(BASE_DIR, "uploads", name)))

    def test_oversized_content_length_rejected_before_parsing(self):
        """Content-Length 超限应在进入解析前就被拦掉"""
        before = self._doc_count()
        old_limit = self.main.MAX_UPLOAD_BYTES
        self.main.MAX_UPLOAD_BYTES = 1024
        try:
            r = self._upload("huge.pdf", MINIMAL_PDF + b"x" * 4096)
            self.assertEqual(r.status_code, 413, r.text)
            self.assertEqual(r.json()["code"], 413)
        finally:
            self.main.MAX_UPLOAD_BYTES = old_limit
        self.assertEqual(self._doc_count(), before)

    def test_pdf_without_extractable_text_rejected(self):
        """合法但抽不出文本的PDF（扫描件/纯图片）必须被拒，且不留下文件"""
        before = self._doc_count()
        blank = build_pdf(None)
        self.assertTrue(blank.startswith(b"%PDF-"), "构造的PDF应以%PDF-开头")
        # pypdf 解析无文本PDF时会把异常直接写到 stderr，此处屏蔽以保持测试输出干净
        with contextlib.redirect_stderr(io.StringIO()):
            r = self._upload("scanned.pdf", blank)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("没有解析出任何文本", r.json()["msg"])
        self.assertFalse(os.path.exists(os.path.join(BASE_DIR, "uploads", "scanned.pdf")))
        self.assertEqual(self._doc_count(), before)

    def test_real_pdf_upload_succeeds_with_metadata(self):
        """带文本的真实PDF应入库成功，并返回可用的 doc_unique_id 与切片数"""
        before = self._doc_count()
        name = "unittest_real_doc.pdf"
        r = self._upload(name, build_pdf("hello rag knowledge base"))
        try:
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["code"], 0)
            self.assertEqual(body["filename"], name)
            self.assertTrue(body["doc_unique_id"])
            self.assertGreaterEqual(body["chunk_count"], 1)
            self.assertGreater(self._doc_count(), before)

            # 重复上传应被去重逻辑识别
            r2 = self._upload(name, build_pdf("hello rag knowledge base"))
            self.assertEqual(r2.status_code, 200)
            self.assertIn("已经上传过", r2.json()["msg"])
        finally:
            db = self.utils.get_vector_db()
            res = db.get(where={"doc_name": name})
            if res["ids"]:
                db.delete(ids=res["ids"])
            path = os.path.join(BASE_DIR, "uploads", name)
            if os.path.exists(path):
                os.remove(path)

    def test_upload_requires_token(self):
        r = self.client.post(
            "/upload",
            files={"file": ("a.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")},
        )
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
