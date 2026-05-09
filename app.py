# -*- coding: utf-8 -*-
"""This is a simple cheatsheet webapp."""

import os
import re

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from flask_sslify import SSLify
from flask_seasurf import SeaSurf
from flask_talisman import Talisman
from werkzeug.exceptions import NotFound
from werkzeug.utils import safe_join

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "docs", "_build", "html")

# Old sphinx-sitemap output (pre-Apr 2026) used the default
# "{lang}{version}{link}" scheme, publishing URLs like
# /en/0.1.0/notes/cpp/cpp_basic.html that the flat origin never served.
# Redirect those legacy paths to the canonical flat URL so Google can drop
# them from the index cleanly instead of 404-churning.
_LEGACY_VERSIONED_PATH = re.compile(r"^(?:en/)?\d+\.\d+\.\d+/(.+)$")


def find_key(token):
    """Find the key from the environment variable."""
    if token == os.environ.get("ACME_TOKEN"):
        return os.environ.get("ACME_KEY")
    for k, v in os.environ.items():
        if v == token and k.startswith("ACME_TOKEN_"):
            n = k.replace("ACME_TOKEN_", "")
            return os.environ.get("ACME_KEY_{}".format(n))


csp = {
    "default-src": "'none'",
    "style-src": [
        "'self'",
        "'unsafe-inline'",
        "fonts.googleapis.com",
    ],
    "script-src": [
        "'self'",
        "*.cloudflare.com",
        "*.cloudflareinsights.com",
        "*.googletagmanager.com",
        "*.google-analytics.com",
        "*.carbonads.com",
        "*.carbonads.net",
        "'unsafe-inline'",
        "'unsafe-eval'",
    ],
    "connect-src": [
        "'self'",
        "*.google-analytics.com",
        "*.analytics.google.com",
        "analytics.google.com",
        "*.googletagmanager.com",
        "*.doubleclick.net",
    ],
    "font-src": [
        "'self'",
        "fonts.gstatic.com",
    ],
    "form-action": "'self'",
    "base-uri": "'self'",
    "img-src": "*",
    "frame-src": "ghbtns.com",
    "frame-ancestors": "'none'",
    "object-src": "'none'",
}

feature_policy = {"geolocation": "'none'"}

app = Flask(__name__, template_folder=ROOT)
app.config["SECRET_KEY"] = os.urandom(16)
app.config["SESSION_COOKIE_NAME"] = "__Secure-session"
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
app.config["CSRF_COOKIE_NAME"] = "__Secure-csrf-token"
app.config["CSRF_COOKIE_HTTPONLY"] = True
app.config["CSRF_COOKIE_SECURE"] = True
csrf = SeaSurf(app)
talisman = Talisman(
    app,
    force_https=False,
    content_security_policy=csp,
    feature_policy=feature_policy,
)

if "DYNO" in os.environ:
    sslify = SSLify(app, permanent=True, skips=[".well-known"])


@app.errorhandler(404)
def page_not_found(e):
    """Redirect to 404.html."""
    return render_template("404.html"), 404


@app.before_request
def redirect_legacy_versioned_paths():
    """301 legacy sitemap URLs (/en/0.1.0/...) to flat canonical paths."""
    match = _LEGACY_VERSIONED_PATH.match(request.path.lstrip("/"))
    if match:
        return redirect("/" + match.group(1), code=301)


@app.route("/<path:path>")
def static_proxy(path):
    """Find static files safely."""
    try:
        return send_from_directory(ROOT, path)
    except NotFound:
        # Handle file not found or directory errors
        return render_template("404.html"), 404


@app.route("/")
def index_redirection():
    """Redirecting index file."""
    return send_from_directory(ROOT, "index.html")


@csrf.exempt
@app.route("/.well-known/acme-challenge/<token>")
def acme(token):
    """Find the acme-key from environment variable."""
    key = find_key(token)
    if key is None:
        abort(404)
    return key


if __name__ == "__main__":
    # Only run the app in debug mode during development
    app.run(debug=os.environ.get("FLASK_ENV") == "development")
