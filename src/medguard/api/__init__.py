"""HTTP API.

``main.create_app`` is an application *factory* rather than a module-level ``app``
object. That matters for testability: each test can build an app with its own
settings, instead of every test sharing one instance configured at import time.
"""

from __future__ import annotations

from medguard.api.main import create_app

__all__ = ["create_app"]
