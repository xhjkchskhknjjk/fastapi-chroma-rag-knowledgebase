"""鉴权改造验证用例

直接运行： python test_auth.py
覆盖：正确/错误/缺失token、空白容错、白名单、OPTIONS预检、dev非回环拒绝、
      dev回环免鉴权、prod缺token启动失败、启动告警文案、非ASCII输入不崩
"""
import os
import subprocess
import sys
import unittest

import _testconfig  # noqa: F401  必须最先导入：在 security 之前注入测试配置

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN = _testconfig.TEST_TOKEN
WRONG_TOKEN = _testconfig.TEST_WRONG_TOKEN


def _install_offline_stub():
    """在导入 utils 之前装好离线桩

    utils.py 在模块级就实例化 HuggingFaceEmbeddings；不装桩时单测会去HuggingFace拉模型，
    本机网络不可达会直接抛 RemoteProtocolError，导致用例与鉴权逻辑无关地失败。
    这里用假的 sentence_transformers 模块顶掉真实实现，保证单测只验证鉴权行为。
    """
    if "sentence_transformers" in sys.modules:
        return

    import types

    from langchain_core.embeddings import Embeddings

    fake_module = types.ModuleType("sentence_transformers")

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

    fake_module.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_module


class TestVerifyTokenUnit(unittest.TestCase):
    """verify_token 单元行为"""

    @classmethod
    def setUpClass(cls):
        import security

        cls.security = security

    def _request(self, token=None, host="127.0.0.1"):
        from starlette.requests import Request

        headers = []
        if token is not None:
            headers.append((b"x-access-token", token.encode("utf-8")))
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/qa/chat",
            "headers": headers,
            "client": (host, 12345),
            "query_string": b"",
        }
        return Request(scope)

    def test_correct_token_passes(self):
        self.assertTrue(self.security.verify_token(self._request(TOKEN)))

    def test_wrong_token_rejected(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            self.security.verify_token(self._request(WRONG_TOKEN))
        self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_token_rejected(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            self.security.verify_token(self._request(None))

    def test_surrounding_whitespace_tolerated(self):
        self.assertTrue(self.security.verify_token(self._request(f"  {TOKEN}  ")))

    def test_non_ascii_token_does_not_crash(self):
        """非ASCII取值必须走正常拒绝分支，而不是TypeError"""
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            self.security.verify_token(self._request("令牌-secret"))

    def test_token_content_not_in_error_detail(self):
        """错误响应里不得回显token内容"""
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            self.security.verify_token(self._request(WRONG_TOKEN))
        self.assertNotIn(WRONG_TOKEN, str(ctx.exception.detail))


class TestLoopbackDetection(unittest.TestCase):
    """回环地址判定"""

    @classmethod
    def setUpClass(cls):
        import security

        cls.security = security

    def _request(self, host):
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/doc/list",
            "headers": [],
            "client": (host, 1) if host else None,
            "query_string": b"",
        }
        return Request(scope)

    def test_ipv4_loopback(self):
        self.assertTrue(self.security.is_loopback_request(self._request("127.0.0.1")))

    def test_ipv6_loopback(self):
        self.assertTrue(self.security.is_loopback_request(self._request("::1")))

    def test_localhost_name(self):
        self.assertTrue(self.security.is_loopback_request(self._request("localhost")))

    def test_lan_address_is_not_loopback(self):
        self.assertFalse(self.security.is_loopback_request(self._request("192.168.1.20")))

    def test_missing_client_is_not_loopback(self):
        self.assertFalse(self.security.is_loopback_request(self._request(None)))


class TestAuthMiddleware(unittest.TestCase):
    """中间件端到端行为"""

    @classmethod
    def setUpClass(cls):
        _install_offline_stub()
        from fastapi.testclient import TestClient

        import main

        cls.client = TestClient(main.app, client=("127.0.0.1", 9000))
        cls.anon_client = TestClient(main.app, client=("192.168.1.20", 9000))

    def test_valid_token_returns_200(self):
        resp = self.client.get("/doc/list", headers={"X-Access-Token": TOKEN})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["code"], 0)

    def test_missing_token_returns_401_with_unified_body(self):
        resp = self.client.get("/doc/list")
        self.assertEqual(resp.status_code, 401)
        body = resp.json()
        self.assertEqual(set(body.keys()), {"code", "msg", "data"})
        self.assertEqual(body["code"], 401)
        self.assertEqual(body["data"], None)
        self.assertEqual(resp.headers.get("WWW-Authenticate"), "ApiKey")

    def test_wrong_token_returns_401(self):
        resp = self.client.get("/doc/list", headers={"X-Access-Token": WRONG_TOKEN})
        self.assertEqual(resp.status_code, 401)

    def test_whitelist_docs_without_token(self):
        for path in ("/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_openapi_json_declares_api_key_scheme(self):
        schema = self.client.get("/openapi.json").json()
        scheme = schema["components"]["securitySchemes"]["ApiKeyAuth"]
        self.assertEqual(scheme["name"], "X-Access-Token")
        self.assertEqual(scheme["in"], "header")

    def test_options_preflight_not_blocked_by_auth(self):
        """预检请求不能被鉴权拦成401，否则浏览器CORS会整体失败"""
        resp = self.client.options("/qa/chat")
        self.assertNotEqual(resp.status_code, 401)

    def test_token_required_even_from_lan(self):
        resp = self.anon_client.get("/doc/list")
        self.assertEqual(resp.status_code, 401)


class TestProductionAndDevPolicies(unittest.TestCase):
    """跨进程验证启动期策略"""

    _CHILD = r"""
import os, sys, types

BASE = r"{base}"
sys.path.insert(0, BASE)

_token = "{token}"
if _token == "__EMPTY__":
    # 用空串模拟"未配置token"：load_dotenv 默认 override=False，
    # 已存在的环境变量（哪怕为空）不会被 .env 覆盖，等价于容器里未注入该变量
    os.environ["API_ACCESS_TOKEN"] = ""
else:
    os.environ["API_ACCESS_TOKEN"] = _token
os.environ["APP_ENV"] = "{app_env}"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 注意：这里不能再调用 load_dotenv()，否则会把 .env 里的 API_ACCESS_TOKEN 补回来，
# 上面"未配置token"的模拟就失效了。security.py 导入时会自行加载 .env，
# 且默认 override=False，已存在的环境变量优先——这正是生产/容器的取值顺序。

# utils.py 模块级会实例化 HuggingFaceEmbeddings，先用假模块顶掉，避免联网拉模型
from langchain_core.embeddings import Embeddings
_fake = types.ModuleType("sentence_transformers")


class _FakeST(Embeddings):
    def __init__(self, model_name=None, *args, **kwargs):
        self._dim = 384

    def encode(self, sentences, **kwargs):
        import numpy as np

        if isinstance(sentences, str):
            return np.zeros(self._dim, dtype="float32")
        return np.zeros((len(sentences), self._dim), dtype="float32")

    def embed_documents(self, texts):
        return [[0.0] * self._dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self._dim


_fake.SentenceTransformer = _FakeST
sys.modules["sentence_transformers"] = _fake

import security
print("IMPORT_OK production_strict={{0}}".format(security.IS_PROD))
print("WARNING_EMITTED" if security.AUTH_STARTUP_WARNING else "NO_WARNING")

from fastapi.testclient import TestClient
import main

local = TestClient(main.app, client=("127.0.0.1", 5000))
lan = TestClient(main.app, client=("10.0.0.9", 5000))
print("LOCAL_STATUS={{0}}".format(local.get("/doc/list").status_code))
_lan_resp = lan.get("/doc/list")
print("LAN_STATUS={{0}}".format(_lan_resp.status_code))
print("LAN_MSG=" + str(_lan_resp.json().get("msg", "")))
"""

    def _run_child(self, app_env, token):
        code = self._CHILD.format(base=BASE_DIR, app_env=app_env, token=token)
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )

    def test_dev_without_token_allows_loopback_only(self):
        """dev且未配置token：本机放行，局域网来源401（原实现是全部放行）"""
        proc = self._run_child("dev", "__EMPTY__")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertIn("WARNING_EMITTED", out)
        self.assertIn("LOCAL_STATUS=200", out)
        self.assertIn("LAN_STATUS=401", out)

    def test_dev_with_no_token_rejection_message(self):
        proc = self._run_child("dev", "__EMPTY__")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LAN_STATUS=401", proc.stdout)
        self.assertIn("未启用访问令牌", proc.stdout)

    def test_prod_without_token_fails_fast(self):
        """prod缺token必须在import阶段就报错，绝不允许静默免鉴权启动"""
        proc = self._run_child("prod", "__EMPTY__")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("API_ACCESS_TOKEN", proc.stderr)

    def test_prod_with_short_token_fails_fast(self):
        proc = self._run_child("prod", "short")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("API_ACCESS_TOKEN", proc.stderr)

    def test_prod_with_strong_token_enforces_auth_everywhere(self):
        """prod配置强token：本机也必须带token"""
        proc = self._run_child("prod", "a" * 32)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertIn("production_strict=True", out)
        self.assertIn("NO_WARNING", out)
        self.assertIn("LOCAL_STATUS=401", out)
        self.assertIn("LAN_STATUS=401", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
