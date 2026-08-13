"""Environment and dependency diagnostics.

This module is the engine behind ``medguard doctor`` and the API's ``/ready`` probe.
The checks live here rather than in the CLI so that both share one implementation --
otherwise the two inevitably drift and disagree about whether the system is healthy.

Two properties every check must have:

* **It never raises.** A doctor command that dies on the first broken dependency
  cannot tell you about the other six. Failures become ``FAIL`` rows.
* **It never prints a secret.** Checks report presence, reachability and redacted
  detail. See :mod:`medguard.logging` for why that is enforced rather than trusted.

The Gemini model check deserves particular attention: it asks the API which models
*this key* can use, instead of trusting documentation. Free-tier availability varies
by project and changes over time, so a live check is the only honest answer.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from enum import StrEnum

import psutil
from pydantic import BaseModel, ConfigDict, Field

from medguard.config import Settings, get_settings, project_root
from medguard.logging import redact_value

__all__ = [
    "Check",
    "CheckStatus",
    "DiagnosticReport",
    "run_diagnostics",
]

#: Rough floor for running Qdrant, the API and ONNX embedding models side by side.
_MIN_RECOMMENDED_RAM_GB = 6.0
_LOW_FREE_RAM_GB = 1.5


class CheckStatus(StrEnum):
    """Outcome of a single check.

    ``WARN`` exists so that "works, but you should know something" does not have to
    masquerade as either success or failure -- an unset Gemini key is a legitimate
    state during early phases, not a broken system.
    """

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class Check(BaseModel):
    """One diagnostic result."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: CheckStatus
    detail: str = ""
    latency_ms: float | None = Field(default=None, ge=0)
    #: True when this check must pass for the system to serve requests.
    critical: bool = False


class DiagnosticReport(BaseModel):
    """A full diagnostic run."""

    checks: list[Check] = Field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def critical_failures(self) -> list[Check]:
        return [c for c in self.failed if c.critical]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status is CheckStatus.WARN]

    @property
    def ok(self) -> bool:
        """True when nothing required for serving requests has failed.

        Deliberately tolerant of non-critical failures: during Phase 1 there is no
        index yet and no key is required, and a diagnostic that reports a red light
        for expected conditions trains people to ignore it.
        """
        return not self.critical_failures

    def exit_code(self) -> int:
        return 1 if self.critical_failures else 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python() -> Check:
    version = platform.python_version()
    ok = sys.version_info >= (3, 11)
    return Check(
        name="Python",
        status=CheckStatus.OK if ok else CheckStatus.FAIL,
        detail=f"{version} ({platform.machine()}) — requires >= 3.11",
        critical=True,
    )


def check_cpu() -> Check:
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    return Check(
        name="CPU",
        status=CheckStatus.OK,
        detail=f"{physical or '?'} physical / {logical or '?'} logical cores",
    )


def check_memory() -> Check:
    vm = psutil.virtual_memory()
    total_gb = vm.total / 1024**3
    available_gb = vm.available / 1024**3

    if total_gb < _MIN_RECOMMENDED_RAM_GB:
        status = CheckStatus.WARN
        note = f" — below the {_MIN_RECOMMENDED_RAM_GB:.0f} GB recommended for the full stack"
    elif available_gb < _LOW_FREE_RAM_GB:
        status = CheckStatus.WARN
        note = " — little free memory; close applications before ingesting"
    else:
        status = CheckStatus.OK
        note = ""

    return Check(
        name="RAM",
        status=status,
        detail=f"{total_gb:.1f} GB total, {available_gb:.1f} GB available{note}",
    )


def check_gpu() -> Check:
    """Detect an accelerator usable for local inference.

    Reports honestly rather than optimistically: an integrated GPU with no working
    CUDA or ROCm stack is CPU-only in practice, and pretending otherwise would lead to
    selecting models this machine cannot run.
    """
    for tool, vendor in (("nvidia-smi", "NVIDIA/CUDA"), ("rocm-smi", "AMD/ROCm")):
        if shutil.which(tool) is None:
            continue
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
                [tool, "--version"] if tool == "rocm-smi" else [tool, "-L"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            first = proc.stdout.strip().splitlines()[0][:80]
            return Check(name="GPU", status=CheckStatus.OK, detail=f"{vendor}: {first}")

    return Check(
        name="GPU",
        status=CheckStatus.WARN,
        detail="No CUDA/ROCm accelerator detected — CPU-only inference. "
        "This is expected here; models are chosen accordingly (ONNX, no PyTorch).",
    )


def check_disk() -> Check:
    usage = psutil.disk_usage(str(project_root()))
    free_gb = usage.free / 1024**3
    status = CheckStatus.OK if free_gb >= 5 else CheckStatus.WARN
    return Check(name="Disk", status=status, detail=f"{free_gb:.1f} GB free on the project volume")


def check_docker() -> Check:
    """Check that the Docker CLI exists *and* the daemon is actually running.

    The distinction matters: Docker Desktop being installed while its engine is
    stopped is the single most common reason Qdrant is unreachable, and reporting only
    "docker found" would send you looking in the wrong place.
    """
    if shutil.which("docker") is None:
        return Check(
            name="Docker",
            status=CheckStatus.WARN,
            detail="Docker CLI not found — needed to run Qdrant locally.",
        )
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check(name="Docker", status=CheckStatus.WARN, detail=f"docker info failed: {exc}")

    if proc.returncode != 0:
        return Check(
            name="Docker",
            status=CheckStatus.WARN,
            detail="Docker CLI present but the daemon is not running. "
            "Start Docker Desktop, then `docker compose -f docker/docker-compose.yml up -d`.",
        )
    return Check(
        name="Docker",
        status=CheckStatus.OK,
        detail=f"daemon running (server {proc.stdout.strip()})",
    )


def check_qdrant(settings: Settings, *, timeout_s: int | None = None) -> Check:
    """Check Qdrant reachability.

    ``timeout_s`` overrides the configured timeout. The readiness probe passes a short
    value: it is polled every few seconds, so it must not block for the full ingestion
    timeout when Qdrant is down. Waiting 30s to learn a service is unreachable would
    make the probe itself time out in most orchestrators.
    """
    from medguard.storage.vector_store import VectorStore

    if timeout_s is not None:
        settings = settings.model_copy(
            update={"qdrant": settings.qdrant.model_copy(update={"timeout_s": timeout_s})}
        )

    store = VectorStore(settings)
    try:
        health = store.health()
    finally:
        store.close()

    if not health.reachable:
        return Check(
            name="Qdrant",
            status=CheckStatus.WARN,
            detail=f"unreachable at {health.url} ({health.error}). "
            "Start it with `docker compose -f docker/docker-compose.yml up -d qdrant`.",
            latency_ms=health.latency_ms,
        )

    if health.collection_exists:
        detail = (
            f"reachable at {health.url}; collection {health.collection!r} "
            f"exists with {health.points_count if health.points_count is not None else '?'} points"
        )
    else:
        detail = (
            f"reachable at {health.url}; collection {health.collection!r} "
            "not created yet (expected before Phase 3 ingestion)"
        )
    return Check(name="Qdrant", status=CheckStatus.OK, detail=detail, latency_ms=health.latency_ms)


def check_database(settings: Settings) -> Check:
    from medguard.storage.database import Database

    db = Database(settings)
    try:
        health = db.health()
    finally:
        db.dispose()

    return Check(
        name="Database",
        status=CheckStatus.OK if health.reachable else CheckStatus.FAIL,
        detail=(
            f"{health.dialect}: {health.url}"
            if health.reachable
            else f"{health.dialect} unreachable — {health.error}"
        ),
        latency_ms=health.latency_ms,
    )


def check_gemini_key(settings: Settings) -> Check:
    """Report whether a key is *present*. Says nothing about validity -- that is next."""
    if settings.gemini.configured:
        return Check(
            name="Gemini API key",
            status=CheckStatus.OK,
            detail="configured (value hidden)",
        )
    return Check(
        name="Gemini API key",
        status=CheckStatus.WARN,
        detail="GEMINI_API_KEY not set — copy .env.example to .env and add a key from "
        "https://aistudio.google.com/apikey",
    )


def check_gemini_models(settings: Settings, *, smoke_test: bool = False) -> list[Check]:
    """Ask the live API which models this key can use, then optionally generate.

    This is the check that makes "do not assume an advertised free model is available
    to your project" enforceable instead of aspirational.
    """
    configured_model = settings.gemini.model
    if not settings.gemini.configured:
        return [
            Check(
                name="Gemini models",
                status=CheckStatus.SKIP,
                detail="skipped — no API key configured",
            ),
            Check(
                name=f"Gemini model ({configured_model})",
                status=CheckStatus.SKIP,
                detail="skipped — no API key configured",
            ),
        ]

    from medguard.providers.gemini import GeminiProvider

    health = GeminiProvider(settings.gemini).health(smoke_test=smoke_test)

    if not health.reachable:
        return [
            Check(
                name="Gemini models",
                status=CheckStatus.FAIL,
                detail=redact_value(health.error or "unreachable"),
                latency_ms=health.latency_ms,
            ),
            Check(
                name=f"Gemini model ({configured_model})",
                status=CheckStatus.SKIP,
                detail="skipped — API unreachable",
            ),
        ]

    preview = ", ".join(health.available_models[:6])
    more = (
        f" (+{len(health.available_models) - 6} more)" if len(health.available_models) > 6 else ""
    )
    checks = [
        Check(
            name="Gemini models",
            status=CheckStatus.OK,
            detail=f"{len(health.available_models)} generation models available: {preview}{more}",
            latency_ms=health.latency_ms,
        )
    ]

    if health.model_available:
        checks.append(
            Check(
                name=f"Gemini model ({configured_model})",
                status=CheckStatus.OK,
                detail="available to this API key",
            )
        )
    else:
        fallback = settings.gemini.fallback_model
        fallback_ok = fallback in health.available_models
        checks.append(
            Check(
                name=f"Gemini model ({configured_model})",
                status=CheckStatus.FAIL,
                detail=f"NOT available to this API key. Fallback {fallback!r} is "
                f"{'available — set GEMINI_MODEL to it' if fallback_ok else 'also unavailable'}.",
            )
        )

    if smoke_test:
        if health.model_available and not health.error:
            checks.append(
                Check(
                    name="Gemini generation",
                    status=CheckStatus.OK,
                    detail="smoke test succeeded — the key can generate text",
                )
            )
        else:
            checks.append(
                Check(
                    name="Gemini generation",
                    status=CheckStatus.FAIL if health.model_available else CheckStatus.SKIP,
                    detail=redact_value(health.error or "skipped — configured model unavailable"),
                )
            )

    return checks


def check_langfuse(settings: Settings) -> Check:
    if not settings.langfuse.enabled:
        return Check(
            name="Langfuse",
            status=CheckStatus.SKIP,
            detail="disabled via LANGFUSE_ENABLED=false",
        )
    if not settings.langfuse.configured:
        return Check(
            name="Langfuse",
            status=CheckStatus.WARN,
            detail="not configured — tracing runs as a no-op. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY to enable (free tier at https://cloud.langfuse.com).",
        )

    from medguard.observability.tracing import LangfuseTracer

    tracer = LangfuseTracer(settings.langfuse)
    try:
        health = tracer.health()
    finally:
        tracer.shutdown()

    if health.reachable:
        return Check(
            name="Langfuse", status=CheckStatus.OK, detail=f"authenticated at {health.base_url}"
        )
    return Check(
        name="Langfuse",
        status=CheckStatus.WARN,
        detail=f"configured but not reachable: {redact_value(health.error or 'unknown')} "
        "— tracing degrades to a no-op; the application is unaffected.",
    )


def check_embedding_config(settings: Settings) -> Check:
    """Report the embedding plan without loading anything.

    Phase 1 downloads no model weights, so this states intent and the collection name
    that will be used, letting you catch a misconfiguration before Phase 4 spends
    bandwidth on it.
    """
    return Check(
        name="Embeddings",
        status=CheckStatus.SKIP,
        detail=f"not loaded in Phase 1. Planned: {settings.embedding.model} "
        f"({settings.embedding.dimension}d) + {settings.embedding.sparse_model} "
        f"→ collection {settings.collection_name!r}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_diagnostics(
    settings: Settings | None = None,
    *,
    smoke_test: bool = False,
    include_hardware: bool = True,
    network_timeout_s: int | None = None,
) -> DiagnosticReport:
    """Run every check and return a report. Never raises.

    ``smoke_test`` spends one real Gemini call against the free-tier budget, so it is
    opt-in rather than the default. ``network_timeout_s`` caps how long connectivity
    probes may block; the readiness endpoint sets it low so the probe stays fast when
    a dependency is down.
    """
    s = settings if settings is not None else get_settings()

    checks: list[Check] = [check_python()]
    if include_hardware:
        checks += [check_cpu(), check_memory(), check_gpu(), check_disk(), check_docker()]
    checks += [
        check_database(s),
        check_qdrant(s, timeout_s=network_timeout_s),
        check_gemini_key(s),
        *check_gemini_models(s, smoke_test=smoke_test),
        check_langfuse(s),
        check_embedding_config(s),
    ]
    return DiagnosticReport(checks=checks)
