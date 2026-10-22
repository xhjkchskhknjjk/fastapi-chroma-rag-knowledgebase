"""把 HuggingFace 离线模式设为默认，避免启动时阻塞在联网检查上

问题：utils.py 在模块级构造 HuggingFaceEmbeddings，transformers/sentence-transformers
      会向 huggingface.co 发 HEAD 请求检查 adapter 配置，每次超时后重试 5 次。
      本机/容器内连不上 huggingface.co 时，服务启动会卡数分钟才起来。
      模型其实已在本地缓存，这些网络检查纯属多余。

做法：在导入任何 langchain/transformers 之前设置离线环境变量（默认为离线），
      需要检查模型更新时用 RAG_HF_OFFLINE=0 显式打开。
"""
import os

from dotenv import load_dotenv

# 先加载 .env，保证下面的判断能读到用户配置
load_dotenv()

# 默认离线（"1"），显式设置 RAG_HF_OFFLINE=0 才联网检查更新
_offline_flag = os.getenv("RAG_HF_OFFLINE", "1").strip().lower()
HF_OFFLINE = _offline_flag not in ("0", "false", "no", "off")

if HF_OFFLINE:
    # 这两个变量必须早于 transformers / sentence_transformers 的导入才生效
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
