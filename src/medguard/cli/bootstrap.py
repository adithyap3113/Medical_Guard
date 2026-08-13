"""Shared CLI startup and output objects.

Extracted so that ``cli/corpus.py`` and ``cli/db.py`` can call the same startup path as
``cli/app.py`` without importing it -- ``app.py`` registers those sub-apps, so importing
it back would be circular.

The consoles live here too, so the whole CLI shares one ``Console`` instance. Two
separate instances disagree about terminal width and colour support, which shows up as
tables that wrap differently depending on which command printed them.
"""

from __future__ import annotations

from rich.console import Console

__all__ = ["bootstrap", "console", "err_console"]

console = Console()
err_console = Console(stderr=True)


def bootstrap() -> None:
    """Load settings and configure logging with secret redaction active.

    Called at the start of every command. Registering the configured credentials with
    the log redactor here means the very first log line is already protected -- there
    is no window in which a secret could be printed.
    """
    from medguard.config import get_settings
    from medguard.logging import configure_logging

    settings = get_settings()
    configure_logging(
        level=settings.app.log_level,
        fmt=settings.app.log_format,
        secrets=[
            settings.gemini.api_key,
            settings.qdrant.api_key,
            settings.langfuse.public_key,
            settings.langfuse.secret_key,
        ],
    )
