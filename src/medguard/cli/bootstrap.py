"""Shared CLI startup and output objects.

Extracted so that ``cli/corpus.py`` and ``cli/db.py`` can call the same startup path as
``cli/app.py`` without importing it -- ``app.py`` registers those sub-apps, so importing
it back would be circular.

The consoles live here too, so the whole CLI shares one ``Console`` instance. Two
separate instances disagree about terminal width and colour support, which shows up as
tables that wrap differently depending on which command printed them.

Encoding is fixed here as well, before either console is constructed. See
:func:`force_utf8_streams`.
"""

from __future__ import annotations

import sys
from typing import TextIO

from rich.console import Console

__all__ = ["bootstrap", "console", "err_console", "force_utf8_streams"]

#: Spellings Python reports for UTF-8. Compared after lowercasing and normalising `-`
#: to `_`, so "UTF-8", "utf8" and "utf_8" all count as already-correct.
_UTF8_ALIASES = frozenset({"utf_8", "utf8"})


def force_utf8_streams() -> None:
    """Make ``stdout``/``stderr`` UTF-8 on Windows, before anything writes to them.

    The bug this fixes
    ------------------
    On a stock Windows console, Python's standard streams default to the system ANSI
    code page -- cp1252 here -- and any character outside it raises
    ``UnicodeEncodeError`` **at write time**. ``medguard corpus download`` printed
    ``Corpus: 20 entries → data/raw`` and died on the arrow, before a single byte had
    been downloaded. The traceback pointed at ``print``, which tells you nothing about
    the actual cause, and the workaround people find first (``set PYTHONIOENCODING``)
    has to be remembered every session on every machine.

    Rich makes this worse rather than better: it inspects ``file.encoding`` to decide
    whether it may emit box-drawing characters, so a cp1252 stream also silently
    degrades every table in the CLI to ASCII.

    Why here and only here
    ----------------------
    This module is imported by ``cli/app.py`` before any command runs, and the call
    below happens before ``Console()`` is constructed, so the consoles inherit the
    corrected encoding rather than caching the broken one. Doing it a second time in
    ``app.py`` would be redundant, and doing it *after* console construction would not
    work at all.

    Why not ``force_terminal=True``
    -------------------------------
    That was the other obvious fix and it is wrong: it makes Rich emit ANSI colour
    escapes even when output is redirected to a file or piped into another program,
    which corrupts exactly the non-interactive uses -- CI logs, ``| findstr`` -- that
    most need to be readable. The encoding is the bug; terminal detection is not.

    Why the condition is the encoding and not ``sys.platform == "win32"``
    ---------------------------------------------------------------------
    Windows is where this actually bites, but the *cause* is the stream encoding, and
    testing the cause has two advantages: a POSIX machine with a hostile ``PYTHONIOENCODING``
    gets the same protection, and the behaviour is reachable from the test suite on
    every platform instead of only on the one where it broke. A platform check would
    make the regression test Windows-only, which is how this bug survived Phase 2.

    Never raises. A CLI that refuses to start because it could not adjust an output
    stream is a worse outcome than one that prints a ``?`` in place of an arrow, so
    ``errors="replace"`` is set as a second line of defence and every failure mode
    below degrades to "leave the stream alone".
    """
    streams: tuple[TextIO, TextIO] = (sys.stdout, sys.stderr)
    for stream in streams:
        encoding = getattr(stream, "encoding", None)
        if isinstance(encoding, str) and encoding.lower().replace("-", "_") in _UTF8_ALIASES:
            continue
        # Not always a TextIOWrapper: pytest's capture, a redirect_stdout block and
        # some terminal wrappers substitute objects with no reconfigure() at all.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, AttributeError):
            # Detached, closed, or already replaced. Nothing to do and nothing worth
            # failing over.
            continue


force_utf8_streams()

console = Console()
err_console = Console(stderr=True)


def bootstrap() -> None:
    """Load settings and configure logging with secret redaction active.

    Called at the start of every command. Registering the configured credentials with
    the log redactor here means the very first log line is already protected -- there
    is no window in which a secret could be printed.
    """
    import urllib.parse

    from pydantic import SecretStr

    from medguard.config import get_settings
    from medguard.logging import configure_logging

    settings = get_settings()
    secrets: list[str | SecretStr | None] = [
        settings.gemini.api_key,
        settings.qdrant.api_key,
        settings.langfuse.public_key,
        settings.langfuse.secret_key,
    ]
    if settings.app.database_url:
        parsed_db = urllib.parse.urlparse(settings.app.database_url)
        if parsed_db.password:
            secrets.append(parsed_db.password)

    configure_logging(
        level=settings.app.log_level,
        fmt=settings.app.log_format,
        secrets=secrets,
    )
