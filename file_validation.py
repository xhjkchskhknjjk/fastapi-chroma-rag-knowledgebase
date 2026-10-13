"""上传文件安全校验与文件名消毒

单独成模块，不依赖 FastAPI，方便直接单元测试这几条纯函数。

设计要点：
- 用户提供的文件名同时是「落盘路径」「去重键」「向量库元数据 doc_name」，
  三者共用同一个值，所以必须消毒而不能简单丢弃或改成随机名。
- 扩展名只是最低门槛，真正的类型校验看文件头魔数，防止改了后缀的伪装文件。
- 错误统一抛 UploadValidationError，由接口层转成 400，而不是让全局处理器兜成 500。
"""
import os
import re
import unicodedata

PDF_EXTENSION = ".pdf"
PDF_MAGIC = b"%PDF-"

# 消毒后的文件名只保留这些字符：中日韩、字母、数字、下划线、连字符、点、括号、空格
_SAFE_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af._\-()\[\] ]")

# Windows 保留设备名，即使带扩展名也不允许作为文件名
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

MAX_STEM_LENGTH = 80


class UploadValidationError(Exception):
    """上传校验失败，接口层直接采用 status_code 与消息，不再靠解析异常文本判断"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def sanitize_filename(raw_name: str | None, max_stem_length: int = MAX_STEM_LENGTH) -> str:
    """把用户提供的文件名消毒成一个安全的纯文件名，非法直接抛 UploadValidationError

    处理内容：
    1. 去掉客户端可能带上的目录部分（含 Windows 反斜杠），只取最后一段；
    2. Unicode 规范化，避免同形不同码点绕过比较；
    3. 过滤控制字符与路径分隔符等危险字符；
    4. 去掉前导点，防止生成隐藏文件或 "." / ".."；
    5. 拦截 Windows 保留设备名；
    6. 限制长度并保留扩展名，防止超长文件名。
    """
    if raw_name is None or not str(raw_name).strip():
        raise UploadValidationError("文件名不能为空")

    name = str(raw_name).strip()

    # 客户端可能传 "C:\\a\\b.pdf" 或 "../../x.pdf"，统一按最后一段取名
    name = name.replace("\\", "/")
    name = name.rsplit("/", 1)[-1]

    name = unicodedata.normalize("NFC", name)
    name = _SAFE_CHARS.sub("_", name)
    name = name.strip(" .")

    if not name:
        raise UploadValidationError("文件名不合法，只包含非法字符")

    stem, extension = os.path.splitext(name)
    if extension.lower() != PDF_EXTENSION:
        raise UploadValidationError("只支持PDF文件，请上传扩展名为 .pdf 的文件")

    stem = stem.strip(" .")
    if not stem:
        raise UploadValidationError("文件名不合法，缺少有效的文件主名")
    if stem.upper() in _RESERVED_NAMES:
        raise UploadValidationError(f"文件名 {stem} 是系统保留名称，请重命名后再上传")

    # 超长主名截断，但保留扩展名，避免整体超限
    suffix = ""
    if len(stem) > max_stem_length:
        suffix = "_" + os.urandom(3).hex()
        stem = stem[: max_stem_length - len(suffix)]
    return f"{stem}{suffix}{PDF_EXTENSION}"


def validate_pdf_content(head_bytes: bytes) -> None:
    """按魔数校验内容确实是PDF，防止只改后缀的伪装文件"""
    if not head_bytes:
        raise UploadValidationError("上传的文件内容为空")
    # 少数生成器会在最前面写空白或BOM，跳过后再比对魔数
    stripped = head_bytes.lstrip(b"\xef\xbb\xbf \t\r\n")
    if not stripped.startswith(PDF_MAGIC):
        raise UploadValidationError("文件内容不是有效的PDF（缺少 %PDF- 文件头）")


def resolve_upload_path(upload_dir: str, safe_name: str) -> str:
    """拼接落盘路径，并断言最终路径确实位于上传目录内（路径穿越兜底）"""
    upload_root = os.path.realpath(upload_dir)
    target = os.path.realpath(os.path.join(upload_root, safe_name))
    if os.path.dirname(target) != upload_root:
        # 消毒后理论上不可能走到这里，保留这道断言作为最后防线
        raise UploadValidationError("文件名不合法，禁止写入上传目录以外的位置")
    return target


def validate_content_length(declared_length: str | None, max_bytes: int) -> None:
    """按 Content-Length 提前拒绝超大请求

    说明：这只是第一道闸。Starlette 在进入路由函数前就会把整个 multipart
    请求体解析并落到临时文件，所以这里的检查能防住"上传到 uploads 目录"，
    但防不住框架层的临时文件落盘——真要彻底限制需要更外层的网关/反向代理配置。
    """
    if not declared_length:
        return
    try:
        length = int(declared_length)
    except (TypeError, ValueError):
        raise UploadValidationError("Content-Length 不是合法整数")
    if length <= 0:
        return
    if length > max_bytes:
        raise UploadValidationError(
            f"请求体 {_human_size(length)} 超过大小上限 {_human_size(max_bytes)}，"
            "请压缩或拆分后再上传",
            status_code=413,
        )


def _human_size(num_bytes: int) -> str:
    """把字节数转成可读大小；上限很小时不能输出成"0MB" """
    if num_bytes < 1024:
        return f"{num_bytes}字节"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / 1024 / 1024:.1f}MB"
