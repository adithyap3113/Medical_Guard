"""Tracing abstraction tests.

One property is under test above all others: **observability cannot break the
application.** Every degradation path -- unconfigured, disabled, client construction
failure, span failure, flush failure -- must leave callers working normally.
"""

from __future__ import annotations

import pytest

from medguard.config import LangfuseSettings
from medguard.observability.tracing import (
    LangfuseTracer,
    NoOpTracer,
    Span,
    Tracer,
    get_tracer,
)


class TestNoOpTracer:
    def test_spans_work_and_yield_a_handle(self) -> None:
        with NoOpTracer().span("retrieve", top_k=5) as span:
            assert isinstance(span, Span)
            assert span.trace_id is None

    def test_span_attribute_updates_are_silently_ignored(self) -> None:
        with NoOpTracer().span("retrieve") as span:
            span.update(candidates=10)
            span.score("faithfulness", 0.8)

    def test_flush_and_shutdown_are_safe(self) -> None:
        tracer = NoOpTracer()
        tracer.flush()
        tracer.shutdown()

    def test_exceptions_inside_a_span_still_propagate(self) -> None:
        """A tracer must never swallow the caller's errors."""
        with pytest.raises(ValueError, match="boom"), NoOpTracer().span("x"):
            raise ValueError("boom")


class TestTracerSelection:
    def test_unconfigured_environment_yields_a_noop(self) -> None:
        get_tracer.cache_clear()
        assert isinstance(get_tracer(), NoOpTracer)

    def test_disabled_flag_yields_a_noop_even_with_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The master switch must win, for offline work and for tests."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-00000000-0000-0000-0000-000000000000")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-00000000-0000-0000-0000-000000000000")
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")

        from medguard.config import get_settings

        get_settings.cache_clear()
        get_tracer.cache_clear()
        assert isinstance(get_tracer(), NoOpTracer)

    def test_tracer_is_cached(self) -> None:
        get_tracer.cache_clear()
        assert get_tracer() is get_tracer()


class TestLangfuseSettings:
    def test_partial_credentials_are_not_configured(self) -> None:
        """Half a credential pair is a misconfiguration, not a working tracer."""
        from pydantic import SecretStr

        assert (
            LangfuseSettings(public_key=SecretStr("pk-lf-x"), secret_key=None).configured is False
        )

    def test_legacy_host_variable_is_honoured(self) -> None:
        """Langfuse v3+ renamed LANGFUSE_HOST to LANGFUSE_BASE_URL."""
        settings = LangfuseSettings(host="https://self-hosted.example.com")
        assert settings.effective_base_url == "https://self-hosted.example.com"

    def test_base_url_defaults_to_cloud(self) -> None:
        assert LangfuseSettings().effective_base_url == "https://cloud.langfuse.com"


class TestDegradation:
    """Every failure mode must degrade to a no-op rather than raise."""

    def test_unconfigured_langfuse_tracer_is_inert(self) -> None:
        tracer = LangfuseTracer(LangfuseSettings())
        assert tracer.configured is False
        with tracer.span("retrieve") as span:
            span.update(k=1)
        tracer.flush()

    def test_client_construction_failure_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken backend must not stop the application from starting."""
        import langfuse
        from pydantic import SecretStr

        def explode(**_: object) -> None:
            raise RuntimeError("cannot reach langfuse")

        monkeypatch.setattr(langfuse, "Langfuse", explode)

        tracer = LangfuseTracer(
            LangfuseSettings(
                public_key=SecretStr("pk-lf-00000000-0000-0000-0000-000000000000"),
                secret_key=SecretStr("sk-lf-00000000-0000-0000-0000-000000000000"),
            )
        )
        with tracer.span("retrieve") as span:
            assert span.trace_id is None
        tracer.flush()

    def test_span_failure_degrades_and_does_not_raise(self) -> None:
        """A backend failure mid-request must not fail the user's request."""
        tracer = LangfuseTracer(LangfuseSettings())

        class BrokenClient:
            def start_as_current_observation(self, **_: object) -> object:
                raise RuntimeError("backend down")

            def flush(self) -> None:
                raise RuntimeError("backend down")

        tracer._client = BrokenClient()
        tracer._degraded = False

        with tracer.span("retrieve") as span:
            assert span.trace_id is None
        tracer.flush()

    def test_health_reports_unconfigured_without_network(self) -> None:
        health = LangfuseTracer(LangfuseSettings()).health()
        assert health.configured is False
        assert health.ok is True, "an unconfigured tracer is not an unhealthy system"


class TestInterchangeability:
    def test_both_implementations_satisfy_the_same_interface(self) -> None:
        """Application code must not branch on which tracer is active."""
        for tracer in (NoOpTracer(), LangfuseTracer(LangfuseSettings())):
            assert isinstance(tracer, Tracer)
            with tracer.span("s") as span, tracer.generation("g") as gen:
                span.update(a=1)
                gen.update(b=2)
            tracer.flush()
            tracer.shutdown()
