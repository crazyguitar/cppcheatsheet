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


if __name__ == "__main__":
    unittest.main()
