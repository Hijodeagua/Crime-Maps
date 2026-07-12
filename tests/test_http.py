"""
Tests for the shared HTTP layer: retry/backoff behavior and default timeouts.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

from crimemaps import http


class TestRetryConfig:
    def test_session_mounts_retrying_adapters(self):
        session = http.make_session()
        for scheme in ("http://example.com", "https://example.com"):
            retry = session.get_adapter(scheme).max_retries
            assert retry.total == http.RETRY_TOTAL
            assert retry.backoff_factor == http.RETRY_BACKOFF_FACTOR
            for status in (429, 500, 502, 503, 504):
                assert status in retry.status_forcelist

    def test_session_keeps_browser_headers(self):
        # The retry port must not drop the WAF-appeasing browser headers —
        # they are why municipal ArcGIS endpoints return 200 instead of 403.
        session = http.make_session()
        assert "Mozilla" in session.headers["User-Agent"]
        assert "Accept" in session.headers


class TestRetryBehavior:
    def _serve(self, handler_cls):
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def test_retries_5xx_then_succeeds(self):
        hits = []

        class FlakyHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                if len(hits) < 3:
                    self.send_response(503)
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok": true}')

            def log_message(self, *args):  # keep test output quiet
                pass

        server = self._serve(FlakyHandler)
        try:
            session = http.make_session(backoff_factor=0.01)
            session.trust_env = False  # ignore proxy env vars for localhost
            resp = session.get(
                f"http://127.0.0.1:{server.server_port}/", timeout=10
            )
            assert resp.status_code == 200
            assert len(hits) == 3, "expected two retried attempts before success"
        finally:
            server.shutdown()

    def test_gives_up_after_max_retries(self):
        hits = []

        class AlwaysFailHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                self.send_response(503)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = self._serve(AlwaysFailHandler)
        try:
            session = http.make_session(retries=2, backoff_factor=0.01)
            session.trust_env = False
            resp = session.get(
                f"http://127.0.0.1:{server.server_port}/", timeout=10
            )
            # raise_on_status=False: exhausted retries surface the last response
            assert resp.status_code == 503
            assert len(hits) == 3  # initial attempt + 2 retries
        finally:
            server.shutdown()


class TestDefaultTimeout:
    def test_module_get_applies_default_timeout(self, monkeypatch):
        captured = {}

        class FakeSession:
            def get(self, url, **kwargs):
                captured.update(kwargs)
                return "response"

        monkeypatch.setattr(http, "_shared", FakeSession())
        http.get("https://example.com/api")
        assert captured["timeout"] == http.DEFAULT_TIMEOUT

    def test_module_get_respects_explicit_timeout(self, monkeypatch):
        captured = {}

        class FakeSession:
            def get(self, url, **kwargs):
                captured.update(kwargs)
                return "response"

        monkeypatch.setattr(http, "_shared", FakeSession())
        http.get("https://example.com/api", timeout=7)
        assert captured["timeout"] == 7

    def test_session_applies_default_timeout(self, monkeypatch):
        captured = {}

        def fake_request(self, method, url, **kwargs):
            captured.update(kwargs)
            return "response"

        monkeypatch.setattr(requests.Session, "request", fake_request)
        session = http.make_session()
        session.get("https://example.com/api")
        assert captured["timeout"] == http.DEFAULT_TIMEOUT
