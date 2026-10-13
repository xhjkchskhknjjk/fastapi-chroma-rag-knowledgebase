"""测试共享配置：统一注入进程环境变量

为什么需要这个模块：
    security.py 在 import 时就把 APP_ENV / API_ACCESS_TOKEN 读成模块级常量，
    而 load_dotenv 默认 override=False（已存在的进程环境变量优先于 .env）。
    因此测试必须在导入 security 之前把配置设好，且所有测试文件必须用同一套值——
    否则谁先被 import，谁的配置生效，会出现「单跑通过、合跑失败」的顺序依赖。

用法（务必放在 import security / main 之前）：
    import _testconfig  # noqa: F401
"""
import logging
import os

TEST_APP_ENV = "dev"
# 测试专用弱口令，仅用于单测，绝不可用于任何真实环境
TEST_TOKEN = "test_token_do_not_use_in_prod_0123456789"
TEST_WRONG_TOKEN = "test_token_wrong_value_9876543210"

# 让单测与 .env / 部署环境彻底解耦
os.environ["APP_ENV"] = TEST_APP_ENV
os.environ["API_ACCESS_TOKEN"] = TEST_TOKEN
# 避免单测联网去 HuggingFace 拉模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 用例会故意喂损坏的PDF，pypdf 会把 PdfReadError 以 ERROR 打到日志里，
# 那是预期行为（被测代码已捕获），降噪以免污染测试输出
logging.getLogger("pypdf").setLevel(logging.CRITICAL)
