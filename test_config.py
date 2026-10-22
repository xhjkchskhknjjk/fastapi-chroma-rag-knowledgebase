"""配置项测试：验证 .env / 环境变量里的值真的会被使用

本次修复的本质问题是"配置写了但不生效"。因此测试必须证明两点：
  1. 改变配置值 -> 实际发出的请求/使用的路径随之改变（正向）
  2. 改错配置 -> 行为随之出错（反向，证明测试不是空转）
"""
import importlib
import io
import os
import subprocess
import sys
import types
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import _testconfig  # noqa: F401,E402  必须最先导入：在 security 之前注入测试配置


def _install_offline_stub():
    """顶掉真实 embedding 模型，避免单测联网拉模型"""
    if "sentence_transformers" in sys.modules:
        return
    from langchain_core.embeddings import Embeddings

    fake = types.ModuleType("sentence_transformers")

    class FakeST(Embeddings):
        def __init__(self, model_name=None, *a, **k):
            self._dim = 384

        def encode(self, sentences, **k):
            import numpy as np

            if isinstance(sentences, str):
                return np.zeros(self._dim, dtype="float32")
            return np.zeros((len(sentences), self._dim), dtype="float32")

        def embed_documents(self, texts):
            return [[0.0] * self._dim for _ in texts]

        def embed_query(self, text):
            return [0.0] * self._dim

    fake.SentenceTransformer = FakeST
    sys.modules["sentence_transformers"] = fake


_install_offline_stub()


class FakeResponse:
    """伪造 requests 响应，记录调用参数"""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class CapturingPost:
    """替换 requests.post，捕获 url / timeout / json 载荷"""

    def __init__(self, payload=None, status_code=200):
        self.calls = []
        self._payload = payload or {
            "choices": [{"message": {"content": "  桩回答  "}}],
            "model": "stub",
        }
        self._status_code = status_code

    def __call__(self, url, json=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout,
                           "headers": headers})
        return FakeResponse(self._payload, self._status_code)


class TestLLMConfigIsActuallyUsed(unittest.TestCase):
    """DEEPSEEK_URL / MODEL / TEMPERATURE / TIMEOUT 必须真正生效"""

    @classmethod
    def setUpClass(cls):
        import main

        cls.main = main

    def setUp(self):
        self._orig_post = self.main.requests.post
        self._orig_model = self.main.DEEPSEEK_MODEL
        self._orig_temp = self.main.DEEPSEEK_TEMPERATURE
        self._orig_timeout = self.main.DEEPSEEK_TIMEOUT
        self._orig_url = self.main.DEEPSEEK_URL
        self._orig_key = self.main.DEEPSEEK_API_KEY

    def tearDown(self):
        self.main.requests.post = self._orig_post
        self.main.DEEPSEEK_MODEL = self._orig_model
        self.main.DEEPSEEK_TEMPERATURE = self._orig_temp
        self.main.DEEPSEEK_TIMEOUT = self._orig_timeout
        self.main.DEEPSEEK_URL = self._orig_url
        self.main.DEEPSEEK_API_KEY = self._orig_key

    def test_model_from_config_reaches_payload(self):
        """改配置里的模型名，实际请求体里的 model 要跟着变"""
        self.main.DEEPSEEK_MODEL = "configured-model-abc"
        fake = CapturingPost()
        self.main.requests.post = fake

        content, err = self.main.call_llm([{"role": "user", "content": "hi"}])

        self.assertEqual(err, "")
        self.assertEqual(content, "桩回答", "应答内容应被 strip")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["json"]["model"], "configured-model-abc")

    def test_url_and_timeout_from_config(self):
        self.main.DEEPSEEK_URL = "https://example.invalid/v1/chat"
        self.main.DEEPSEEK_TIMEOUT = 123
        fake = CapturingPost()
        self.main.requests.post = fake

        self.main.call_llm([{"role": "user", "content": "hi"}])

        call = fake.calls[0]
        self.assertEqual(call["url"], "https://example.invalid/v1/chat")
        self.assertEqual(call["timeout"], 123)

    def test_temperature_from_config(self):
        self.main.DEEPSEEK_TEMPERATURE = 0.75
        fake = CapturingPost()
        self.main.requests.post = fake

        self.main.call_llm([{"role": "user", "content": "hi"}])

        self.assertEqual(fake.calls[0]["json"]["temperature"], 0.75)

    def test_missing_api_key_returns_error_without_http_call(self):
        """没有密钥时必须立刻报错，而不是发出一个注定 401 的请求"""
        self.main.DEEPSEEK_API_KEY = ""
        fake = CapturingPost()
        self.main.requests.post = fake

        content, err = self.main.call_llm([{"role": "user", "content": "hi"}])

        self.assertIsNone(content)
        self.assertIn("DEEPSEEK_API_KEY", err)
        self.assertEqual(fake.calls, [], "不应发出任何 HTTP 请求")

    def test_auth_header_uses_configured_key(self):
        self.main.DEEPSEEK_API_KEY = "sk-test-config-key"
        fake = CapturingPost()
        self.main.requests.post = fake

        self.main.call_llm([{"role": "user", "content": "hi"}])

        self.assertEqual(
            fake.calls[0]["headers"]["Authorization"], "Bearer sk-test-config-key"
        )


class TestSecurityConfigSurface(unittest.TestCase):
    """security 模块暴露的配置项必须来自环境变量"""

    def test_upload_dir_default(self):
        import security

        self.assertTrue(security.UPLOAD_DIR)
        self.assertIsInstance(security.UPLOAD_DIR, str)

    def test_effective_config_summary_hides_secrets(self):
        import security

        summary = security.EFFECTIVE_CONFIG_SUMMARY
        rendered = str(summary)
        # 只允许出现布尔标识，绝不允许出现密钥或token原文
        if security.DEEPSEEK_API_KEY:
            self.assertNotIn(security.DEEPSEEK_API_KEY, rendered)
        if security.EXPECTED_TOKEN:
            self.assertNotIn(security.EXPECTED_TOKEN, rendered)
        self.assertIn("API_KEY_CONFIGURED", summary)
        self.assertIn("API_ACCESS_TOKEN_CONFIGURED", summary)

    def test_app_version_default_non_empty(self):
        import security

        self.assertTrue(security.APP_VERSION)
        self.assertNotEqual(security.APP_VERSION.strip(), "")

    def test_main_source_has_no_config_assignment(self):
        """静态扫描：main.py 里不允许给配置项赋值（只能从 security 导入）

        为什么不能只靠运行时比对：如果硬编码的字符串与 .env 里的值恰好相同，
        模块级变量的最终值仍然"正确"，运行时比对发现不了遮蔽。
        必须直接看源码里有没有赋值语句。
        """
        import re

        src = open(os.path.join(BASE_DIR, "main.py"), encoding="utf-8").read()

        config_names = [
            "DEEPSEEK_URL",
            "DEEPSEEK_MODEL",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_TEMPERATURE",
            "DEEPSEEK_TIMEOUT",
            "SYSTEM_PROMPT",
            "UPLOAD_DIR",
            "APP_VERSION",
            "RAG_RATE_LIMIT",
            "MAX_UPLOAD_MB",
        ]
        violations = []
        for name in config_names:
            # 匹配 "NAME = ..." 或 "NAME: type = ..."，且行首不是 import/from
            pattern = re.compile(rf"^{name}\s*(?::[^=]+)?=\s*\S")
            for lineno, line in enumerate(src.splitlines(), 1):
                if pattern.match(line.strip()):
                    violations.append(f"  行{lineno}: {line.strip()}")

        self.assertEqual(
            violations, [],
            "main.py 中存在对配置项的硬编码赋值（应从 security 导入）：\n"
            + "\n".join(violations),
        )

    def test_main_module_does_not_shadow_config(self):
        """main 里使用到的配置值应与 security 解析出的值保持一致"""
        import security

        import main

        for name in (
            "DEEPSEEK_URL",
            "DEEPSEEK_MODEL",
            "DEEPSEEK_TEMPERATURE",
            "DEEPSEEK_TIMEOUT",
            "UPLOAD_DIR",
            "SYSTEM_PROMPT",
        ):
            with self.subTest(config=name):
                self.assertEqual(
                    getattr(main, name),
                    getattr(security, name),
                    f"main.{name} 与 security.{name} 不一致",
                )


class TestConfigFromEnvironment(unittest.TestCase):
    """跨进程验证：环境变量能覆盖配置（模块级常量只能在导入时读取）"""

    _CHILD = r'''
import os, sys, types
sys.path.insert(0, r"{base}")
os.environ["APP_ENV"] = "dev"
os.environ["API_ACCESS_TOKEN"] = "test_token_do_not_use_in_prod_0123456789"
os.environ["DEEPSEEK_API_KEY"] = "sk-child-test"
os.environ["RAG_UPLOAD_DIR"] = "/tmp/child_upload_dir"
# {version_key} 由父进程指定：正常用例用正确变量名，反向用例故意用错误变量名
os.environ["{version_key}"] = "9.9.9"

import security
print("UPLOAD_DIR=" + security.UPLOAD_DIR)
print("APP_VERSION=" + security.APP_VERSION)
print("MODEL=" + security.DEEPSEEK_MODEL)
print("TEMPERATURE=" + str(security.DEEPSEEK_TEMPERATURE))
print("TIMEOUT=" + str(security.DEEPSEEK_TIMEOUT))

from fastapi.testclient import TestClient
import main
schema = TestClient(main.app).get("/openapi.json").json()
print("OPENAPI_VERSION=" + schema["info"]["version"])
'''

    def _run(self, version_key="RAG_APP_VERSION"):
        code = self._CHILD.format(base=BASE_DIR, version_key=version_key)
        env = dict(os.environ)
        for k in ("RAG_UPLOAD_DIR", "RAG_APP_VERSION", "APP_VERSION"):
            env.pop(k, None)
        return subprocess.run(
            [sys.executable, "-c", code], cwd=BASE_DIR, env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=600,
        )

    def test_upload_dir_and_version_override(self):
        proc = self._run(version_key="RAG_APP_VERSION")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = proc.stdout
        self.assertIn("UPLOAD_DIR=/tmp/child_upload_dir", out)
        self.assertIn("APP_VERSION=9.9.9", out)
        # OpenAPI 文档里的版本号必须跟着配置走
        self.assertIn("OPENAPI_VERSION=9.9.9", out)

    def test_wrong_env_name_does_not_take_effect(self):
        """反向证明：变量名写错时配置不应生效（证明测试不是空转）"""
        proc = self._run(version_key="APP_VERSION")  # 故意用错误的变量名
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("APP_VERSION=1.0.0", proc.stdout, "错误的变量名不应生效")
        self.assertNotIn("APP_VERSION=9.9.9", proc.stdout)

    def test_prod_without_api_key_refuses_to_start(self):
        """prod 缺大模型密钥应拒绝启动：核心功能完全不可用"""
        code = r'''
import os, sys
sys.path.insert(0, r"{base}")
os.environ["APP_ENV"] = "prod"
os.environ["API_ACCESS_TOKEN"] = "a" * 32
os.environ.pop("DEEPSEEK_API_KEY", None)
os.environ["DEEPSEEK_API_KEY"] = ""
import security
print("SHOULD_NOT_REACH_HERE")
'''
        proc = subprocess.run(
            [sys.executable, "-c", code.format(base=BASE_DIR)],
            cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", timeout=300,
        )
        self.assertNotEqual(proc.returncode, 0, "prod 缺密钥竟启动成功")
        self.assertIn("DEEPSEEK_API_KEY", proc.stderr)
        self.assertNotIn("SHOULD_NOT_REACH_HERE", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
