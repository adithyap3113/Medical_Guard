"""Health and readiness endpoint tests.

The behavioural contract these pin down is the liveness/readiness split. It is easy
to implement both as "return 200 if the process is up", which silently removes the
operational value of having two endpoints and can turn a dependency blip into a
restart storm.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medguard import __version__
from medguard.models import MEDICAL_DISCLAIMER


class TestLiveness:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_reports_the_version(self, client: TestClient) -> None:
        assert client.get("/health").json()["version"] == __version__

    def test_health_timestamp_is_timezone_aware(self, client: TestClient) -> None:
        assert client.get("/health").json()["timestamp"].endswith(("Z", "+00:00"))

    def test_health_does_not_touch_dependencies(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Liveness must stay green when every dependency is down.

        Otherwise an orchestrator restarts healthy processes during an outage of
        something they do not need in order to be alive.
        """

        def explode(*_: object, **__: object) -> None:
            raise AssertionError("liveness must not run diagnostics")

        monkeypatch.setattr("medguard.diagnostics.run_diagnostics", explode)
        assert client.get("/health").status_code == 200


class TestReadiness:
    def test_ready_returns_component_detail(self, client: TestClient) -> None:
        payload = client.get("/ready").json()
        assert "ready" in payload
        names = {c["name"] for c in payload["components"]}
        assert {"Python", "Database", "Qdrant"} <= names

    def test_ready_is_200_when_critical_dependencies_are_up(self, client: TestClient) -> None:
        """The isolated test env uses SQLite and in-memory Qdrant, so this is green."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_ready_returns_503_when_a_critical_check_fails(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A load balancer needs 503 to stop routing here."""
        from medguard.diagnostics import Check, CheckStatus, DiagnosticReport

        monkeypatch.setattr(
            "medguard.api.routes.health.run_diagnostics",
            lambda *_, **__: DiagnosticReport(
                checks=[
                    Check(
                        name="Database",
                        status=CheckStatus.FAIL,
                        detail="unreachable",
                        critical=True,
                    )
                ]
            ),
        )
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_ready_skips_hardware_and_smoke_test(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe runs constantly: it must be cheap and must never spend API quota."""
        captured: dict[str, object] = {}
        from medguard.diagnostics import DiagnosticReport, run_diagnostics

        def spy(*args: object, **kwargs: object) -> DiagnosticReport:
            captured.update(kwargs)
            return run_diagnostics(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("medguard.api.routes.health.run_diagnostics", spy)
        client.get("/ready")

        assert captured["smoke_test"] is False
        assert captured["include_hardware"] is False

    def test_ready_uses_a_short_network_timeout(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: the probe must not block on the full ingestion timeout.

        With Qdrant down and the default 30s timeout, /ready took ~5s, which exceeds
        a typical orchestrator probe deadline and would mark a healthy instance dead.
        """
        captured: dict[str, object] = {}
        from medguard.diagnostics import DiagnosticReport, run_diagnostics

        def spy(*args: object, **kwargs: object) -> DiagnosticReport:
            captured.update(kwargs)
            return run_diagnostics(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("medguard.api.routes.health.run_diagnostics", spy)
        client.get("/ready")

        timeout = captured["network_timeout_s"]
        assert isinstance(timeout, int)
        assert 0 < timeout <= 5


class TestApiContract:
    def test_disclaimer_is_part_of_the_openapi_contract(self, client: TestClient) -> None:
        """The disclaimer must not depend on a UI remembering to show it."""
        description = client.get("/openapi.json").json()["info"]["description"]
        assert MEDICAL_DISCLAIMER in description

    def test_openapi_makes_no_clinical_validation_claim(self, client: TestClient) -> None:
        """The API must disclaim clinical validation, never assert it.

        Checked with a negation-aware pattern rather than a bare substring search:
        "has not been clinically validated" contains the same words as the claim it
        denies, so a naive `not in` check would reject correct disclaimer wording.
        """
        import re

        description = client.get("/openapi.json").json()["info"]["description"].lower()

        affirmative_claims = [
            r"(?<!not )\bis clinically validated",
            r"(?<!not )\bis medically safe",
            r"\bfda[- ]approved",
            r"\bis a medical device",
            r"\bdiagnoses patients",
        ]
        for pattern in affirmative_claims:
            assert not re.search(pattern, description), f"OpenAPI asserts {pattern!r}"

        # And it must positively state the limitation, not merely omit the claim.
        assert "not been clinically validated" in description
        assert "not a medical device" in description

    def test_query_endpoint_is_exposed(self, client: TestClient) -> None:
        """The API contract exposes the health, ready, and query endpoints."""
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/health", "/ready", "/api/v1/query"} <= set(paths)


class TestErrorEnvelope:
    def test_domain_errors_return_the_safe_message_only(self, settings) -> None:  # type: ignore[no-untyped-def]
        """Vendor payloads must never reach a client."""
        from medguard.api.main import create_app
        from medguard.errors import ProviderAuthError

        app = create_app(settings)

        @app.get("/_boom")
        def boom() -> None:
            raise ProviderAuthError("key AIzaSECRET123456789 rejected by upstream")

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get("/_boom")

        assert response.status_code == 503
        body = response.json()
        assert "AIzaSECRET" not in str(body)
        assert body["error"] == "ProviderAuthError"
        assert body["message"] == ProviderAuthError.user_message
        assert body["trace_id"]
        assert body["disclaimer"] == MEDICAL_DISCLAIMER


class TestAppLifespanAndSharedResources:
    def test_lifespan_manages_shared_vector_store(self, settings) -> None:  # type: ignore[no-untyped-def]
        from medguard.api.main import create_app
        from medguard.storage.vector_store import VectorStore

        app = create_app(settings)
        with TestClient(app) as c:
            assert hasattr(app.state, "vector_store")
            vs = app.state.vector_store
            assert isinstance(vs, VectorStore)
            # Make sure health endpoint works during lifespan
            resp = c.get("/health")
            assert resp.status_code == 200

        # After lifespan exits, vector store is closed
        assert vs._client is None
