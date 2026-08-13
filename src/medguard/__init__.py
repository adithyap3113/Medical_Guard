"""MedGuard -- medical knowledge RAG and evidence retrieval system.

This package is intentionally shallow at import time: importing ``medguard`` must not
read configuration, open network connections, or load models, so that ``--help`` and
the test suite stay fast and work on a machine with no services running.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
