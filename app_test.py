"""Test app.py."""

import multiprocessing
import platform
import unittest
import requests
import os

from pathlib import Path
from werkzeug.exceptions import NotFound
from flask_testing import LiveServerTestCase

from app import acme, find_key, static_proxy, index_redirection, page_not_found
from app import redirect_legacy_versioned_paths
from app import redirect_legacy_flat_paths, redirect_canonical_host
from app import _resolve_legacy_flat_target

from app import ROOT
from app import app


if platform.system() == "Darwin":
    multiprocessing.set_start_method("fork")


class PysheeetTest(LiveServerTestCase):
    """Test app."""

    def create_app(self):
        """Create a app for test."""
        # remove env ACME_TOKEN*
        for k, v in os.environ.items():
            if not k.startswith("ACME_TOKEN"):
                continue
            del os.environ[k]

        self.token = "token"
        self.key = "key"
        os.environ["ACME_TOKEN"] = self.token
        os.environ["ACME_KEY"] = self.key
        os.environ["FLASK_ENV"] = "development"
        os.environ["FLASK_DEBUG"] = "1"
        app.config["TESTING"] = True
        app.config["LIVESERVER_PORT"] = 0
        return app

    def check_security_headers(self, resp):
        """Check security headers."""
        headers = resp.headers
        self.assertTrue("Content-Security-Policy" in headers)
        self.assertTrue("X-Content-Type-Options" in headers)
        self.assertTrue("Content-Security-Policy" in headers)
        self.assertTrue("Feature-Policy" in headers)
        self.assertEqual(headers["Feature-Policy"], "geolocation 'none'")
        self.assertEqual(headers["X-Frame-Options"], "SAMEORIGIN")

    def check_csrf_cookies(self, resp):
        """Check cookies for csrf."""
        cookies = resp.cookies
        self.assertTrue(cookies.get("__Secure-session"))
        self.assertTrue(cookies.get("__Secure-csrf-token"))

    def test_index_redirection_req(self):
        """Test that send a request for the index page."""
        url = self.get_server_url()
        resp = requests.get(url)
        self.check_security_headers(resp)
        self.check_csrf_cookies(resp)
        self.assertEqual(resp.status_code, 200)

    def test_static_proxy_req(self):
        """Test that send a request for notes."""
        url = self.get_server_url()
        notes = Path(ROOT) / "notes"
        for html in notes.rglob("*.html"):
            page = html.relative_to(ROOT)
            u = f"{url}/{page}"
            resp = requests.get(u)
            self.check_security_headers(resp)
            self.check_csrf_cookies(resp)
            self.assertEqual(resp.status_code, 200)

    def test_acme_req(self):
        """Test that send a request for a acme key."""
        url = self.get_server_url()
        u = url + "/.well-known/acme-challenge/token"
        resp = requests.get(u)
        self.check_security_headers(resp)
        self.assertEqual(resp.status_code, 200)

        u = url + "/.well-known/acme-challenge/foo"
        resp = requests.get(u)
        self.check_security_headers(resp)
        self.assertEqual(resp.status_code, 404)

    def test_find_key(self):
        """Test that find a acme key from the environment."""
        token = self.token
        key = self.key
        self.assertEqual(find_key(token), key)

        del os.environ["ACME_TOKEN"]
        del os.environ["ACME_KEY"]

        os.environ["ACME_TOKEN_ENV"] = token
        os.environ["ACME_KEY_ENV"] = key
        self.assertEqual(find_key(token), key)

        del os.environ["ACME_TOKEN_ENV"]
        del os.environ["ACME_KEY_ENV"]

    def test_acme(self):
        """Test that send a request for a acme key."""
        token = self.token
        key = self.key
        self.assertEqual(acme(token), key)

        token = token + "_env"
        key = key + "_env"
        os.environ["ACME_TOKEN_ENV"] = token
        os.environ["ACME_KEY_ENV"] = key
        self.assertEqual(find_key(token), key)

        del os.environ["ACME_TOKEN_ENV"]
        del os.environ["ACME_KEY_ENV"]

        self.assertRaises(NotFound, acme, token)

    def test_index_redirection(self):
        """Test index page redirection."""
        resp = index_redirection()
        self.assertEqual(resp.status_code, 200)
        resp.close()

    def test_static_proxy(self):
        """Test that request static pages."""
        notes = Path(ROOT) / "notes"
        for html in notes.rglob("*.html"):
            u = html.relative_to(ROOT)
            resp = static_proxy(u)
            self.assertEqual(resp.status_code, 200)
            resp.close()

        u = "notes/../conf.py"
        _, code = static_proxy(u)
        self.assertEqual(code, 404)

    def test_page_not_found(self):
        """Test page not found."""
        html, status_code = page_not_found(None)
        self.assertEqual(status_code, 404)

    def test_legacy_versioned_path_redirect(self):
        """Legacy /en/0.1.0/... and /0.1.0/... must 301 to flat paths."""
        url = self.get_server_url()
        cases = {
            "/en/0.1.0/notes/cpp/cpp_basic.html": "/notes/cpp/cpp_basic.html",
            "/0.1.0/notes/cpp/cpp_basic.html": "/notes/cpp/cpp_basic.html",
        }
        for legacy, flat in cases.items():
            resp = requests.get(url + legacy, allow_redirects=False)
            self.assertEqual(resp.status_code, 301)
            self.assertTrue(resp.headers["Location"].endswith(flat))

    def test_redirect_legacy_versioned_paths_passthrough(self):
        """Non-legacy paths must not be intercepted by the redirector."""
        with app.test_request_context("/notes/cpp/cpp_basic.html"):
            self.assertIsNone(redirect_legacy_versioned_paths())

    def test_redirect_legacy_versioned_paths_match(self):
        """Legacy versioned paths return a 301 to the flat canonical URL."""
        legacy = "/en/0.1.0/notes/cpp/cpp_basic.html"
        with app.test_request_context(legacy):
            resp = redirect_legacy_versioned_paths()
            self.assertEqual(resp.status_code, 301)
            self.assertTrue(
                resp.headers["Location"].endswith("/notes/cpp/cpp_basic.html")
            )

    def test_resolve_legacy_flat_target_explicit(self):
        """Explicit renames in the lookup table resolve correctly."""
        cases = {
            "notes/asm_basic.html": "notes/c/asm.html",
            "notes/cmake_basic.html": "notes/cpp/cpp_cmake.html",
            "notes/cpp_ranges.html": "notes/cpp/cpp_iterator.html",
            "notes/gdb_debug.html": "notes/debug/gdb.html",
            "notes/bash_find.html": "notes/tools/bash.html",
        }
        for old, new in cases.items():
            self.assertEqual(_resolve_legacy_flat_target(old), new)

    def test_resolve_legacy_flat_target_pattern(self):
        """Pattern fallback resolves flat URLs to existing nested files."""
        # cpp/cpp_basic.html exists, so cpp_basic.html should resolve to it.
        self.assertEqual(
            _resolve_legacy_flat_target("notes/cpp_basic.html"),
            "notes/cpp/cpp_basic.html",
        )

    def test_resolve_legacy_flat_target_unknown(self):
        """Unknown paths return None so the request is not redirected."""
        cases = (
            "notes/cpp/cpp_basic.html",
            "notes/totally_made_up.html",
            "about.html",
        )
        for path in cases:
            self.assertIsNone(_resolve_legacy_flat_target(path))

    def test_redirect_legacy_flat_paths_passthrough(self):
        """Current nested URLs are not intercepted by the legacy redirector."""
        with app.test_request_context("/notes/cpp/cpp_basic.html"):
            self.assertIsNone(redirect_legacy_flat_paths())

    def test_redirect_legacy_flat_paths_explicit(self):
        """A renamed flat URL returns 301 to its new nested location."""
        with app.test_request_context("/notes/asm_basic.html"):
            resp = redirect_legacy_flat_paths()
            self.assertEqual(resp.status_code, 301)
            self.assertTrue(
                resp.headers["Location"].endswith("/notes/c/asm.html")
            )

    def test_redirect_canonical_host_www(self):
        """Requests to www.cppcheatsheet.com 301 to the bare domain."""
        with app.test_request_context(
            "/notes/cpp/cpp_basic.html",
            base_url="https://www.cppcheatsheet.com",
        ):
            resp = redirect_canonical_host()
            self.assertEqual(resp.status_code, 301)
            self.assertEqual(
                resp.headers["Location"],
                "https://cppcheatsheet.com/notes/cpp/cpp_basic.html",
            )

    def test_redirect_canonical_host_preserves_query(self):
        """Non-www redirect preserves the query string."""
        with app.test_request_context(
            "/search?q=cmake",
            base_url="https://www.cppcheatsheet.com",
        ):
            resp = redirect_canonical_host()
            self.assertEqual(resp.status_code, 301)
            self.assertEqual(
                resp.headers["Location"],
                "https://cppcheatsheet.com/search?q=cmake",
            )

    def test_redirect_canonical_host_passthrough(self):
        """Requests already on the canonical host are not redirected."""
        with app.test_request_context(
            "/", base_url="https://cppcheatsheet.com",
        ):
            self.assertIsNone(redirect_canonical_host())


if __name__ == "__main__":
    unittest.main()
