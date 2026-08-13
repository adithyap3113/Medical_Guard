"""Structured logging with mandatory secret redaction.

Threat model
------------
"Do not log secrets" fails as a coding convention, because the leak is rarely a
deliberate ``log.info(api_key)``. It is:

* an exception whose message embeds the request URL, key in the query string;
* a third-party SDK logging its own config at DEBUG;
* ``log.info("config", **settings.model_dump())`` where one nested field is a key;
* a retry wrapper logging the full request payload.

So redaction is enforced at the *sink*, not at the call site. Every record --
including records emitted by uvicorn, httpx, and the Google SDK -- passes through
``redact_processor`` before it is rendered. There is no logging path that bypasses it.

Three independent defences, because each one alone has gaps:

1. **Key-name matching** -- any field whose *name* looks sensitive (``api_key``,
   ``authorization``, ``token``) is dropped regardless of its value. Catches secrets
   whose format we do not recognise.
2. **Value-pattern matching** -- regexes for known credential shapes (Google
   ``AIza...``, Langfuse ``pk-lf-``/``sk-lf-``, bearer tokens). Catches secrets that
   appear inside a free-text message under an innocent field name.
3. **Known-literal matching** -- the actual configured secret values are registered
   at startup and substring-replaced anywhere they appear. Catches secrets whose
   format we do *not* have a regex for, including in tracebacks.

Redaction deliberately runs *after* exception formatting, so stack traces are
scrubbed but remain complete and debuggable: only the credential substring is
replaced, never the surrounding message or frames.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable, MutableMapping
from typing import Any

import structlog
from pydantic import SecretStr

__all__ = [
    "REDACTED",
    "configure_logging",
    "get_logger",
    "redact_processor",
    "redact_value",
    "register_secrets",
]

REDACTED = "***REDACTED***"

#: Field names whose values are dropped wholesale, matched as substrings against the
#: lowercased key. Broad on purpose: a false positive costs one unhelpful log line,
#: a false negative costs a leaked credential.
_SENSITIVE_KEY_PARTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "credential",
    "private_key",
    "session_id",
    "cookie",
)

#: Credential shapes recognised by value, independent of the field they appear in.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Google / Gemini API keys.
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    # Langfuse public/secret keys.
    re.compile(r"\b[ps]k-lf-[0-9a-fA-F-]{8,}"),
    # Generic `sk-`-prefixed keys used by several vendors.
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    # Authorization headers.
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-=+/]{8,}"),
    # Credentials embedded in URLs, e.g. postgresql://user:pw@host or ?key=...
    re.compile(r"(?i)([?&](?:key|api_key|access_token)=)[^&\s\"']+"),
    re.compile(r"(?i)(://[^:/\s]+:)[^@/\s]+(@)"),
)

#: Literal secret values registered from settings at startup. Module-private and
#: never logged, iterated into output, or exposed via a public accessor.
_KNOWN_SECRETS: set[str] = set()

#: Below this length, substring replacement would corrupt unrelated text.
_MIN_LITERAL_SECRET_LEN = 8


def register_secrets(values: Iterable[str | SecretStr | None]) -> None:
    """Register literal secret values to be scrubbed from all future log output.

    Called by :func:`configure_logging` with the configured credentials. Safe to
    call repeatedly; values shorter than 8 characters are ignored because
    substring-replacing them would mangle ordinary text.
    """
    for value in values:
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if raw and len(raw.strip()) >= _MIN_LITERAL_SECRET_LEN:
            _KNOWN_SECRETS.add(raw.strip())


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _scrub_text(text: str) -> str:
    for secret in _KNOWN_SECRETS:
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            # Preserve the structural prefix/suffix groups, replace only the payload.
            text = pattern.sub(lambda m: "".join(g or "" for g in m.groups()) + REDACTED, text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def redact_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact a value, dropping it entirely if its key looks sensitive.

    Exposed for reuse outside logging -- for example when echoing configuration in
    ``medguard doctor`` or in an API error envelope.
    """
    if key is not None and _is_sensitive_key(key):
        return REDACTED if value is not None else None
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, MutableMapping):
        return {k: redact_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        scrubbed = [redact_value(v) for v in value]
        return type(value)(scrubbed) if isinstance(value, (list, tuple)) else set(scrubbed)
    return value


def redact_processor(
    _logger: object, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor that scrubs every field of every log record."""
    return {k: redact_value(v, key=str(k)) for k, v in event_dict.items()}


def _shared_processors() -> list[Any]:
    """Processors applied to structlog and stdlib records alike."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str = "console",
    secrets: Iterable[str | SecretStr | None] = (),
) -> None:
    """Configure structlog and route stdlib logging through the same pipeline.

    Routing stdlib logging through ``ProcessorFormatter`` is the point: it means
    uvicorn, httpx and the Google SDK cannot emit an unredacted line, because they
    share our sink. Configuring structlog alone would leave those paths open.
    """
    register_secrets(secrets)
    shared = _shared_processors()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # Render the traceback to text first, so redaction can scrub inside it.
            structlog.processors.format_exc_info,
            redact_processor,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # These are chatty at INFO and duplicate our own request logging.
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger. Prefer ``get_logger(__name__)``."""
    return structlog.stdlib.get_logger(name)
