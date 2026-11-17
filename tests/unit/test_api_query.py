"""Tests for the POST /api/v1/query endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medguard.api.main import create_app
from medguard.config import Settings
from medguard.models import MEDICAL_DISCLAIMER
from medguard.storage.vector_store import VectorStore
from tests.unit.test_retrieval import _chunk


@pytest.fixture
def client(settings: Settings) -> TestClient:
    # Use in-memory Qdrant store populated with a few sample chunks
    store = VectorStore(settings)
    store.ensure_collection()
    store.upsert_chunks(
        [
            _chunk("cdc_htn", 0, "CDC", "guideline"),
            _chunk("who_htn", 1, "WHO", "guideline"),
        ]
    )
    app = create_app(settings)
    app.state.vector_store = store
    return TestClient(app)


class TestQueryEndpoint:
    def test_query_success_without_llm_key(self, client: TestClient) -> None:
        payload = {
            "question": "What is the recommended treatment for hypertension?",
            "top_k": 2,
            "strategy": "basic",
            "retriever": "dense",
        }
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert "answer" in data
        assert "citations" in data
        assert "retrieval" in data
        assert data["retrieval"]["strategy"] == "basic"
        assert data["retrieval"]["retriever"] == "dense"
        assert data["medical_disclaimer"] == MEDICAL_DISCLAIMER
        assert "trace_id" in data

    def test_query_empty_question_fails_validation(self, client: TestClient) -> None:
        payload = {"question": "", "top_k": 5}
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 422
        err = resp.json()
        assert err["error"] == "ValidationError"
        assert "trace_id" in err
        assert "Traceback" not in str(err)

    def test_query_whitespace_only_question_fails_validation(self, client: TestClient) -> None:
        payload = {"question": "    \t\n   ", "top_k": 5}
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 422
        err = resp.json()
        assert err["error"] == "ValidationError"
        assert "whitespace" in err["message"]
        assert "trace_id" in err

    def test_query_excessively_long_question_fails_validation(self, client: TestClient) -> None:
        long_q = "hypertension " * 300  # > 3000 chars
        payload = {"question": long_q, "top_k": 5}
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 422
        err = resp.json()
        assert err["error"] == "ValidationError"
        assert "trace_id" in err

    def test_query_invalid_top_k_fails_validation(self, client: TestClient) -> None:
        payload = {"question": "hypertension", "top_k": 0}
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 422

        payload_too_high = {"question": "hypertension", "top_k": 51}
        resp2 = client.post("/api/v1/query", json=payload_too_high)
        assert resp2.status_code == 422

    def test_query_invalid_retriever_fails_validation(self, client: TestClient) -> None:
        payload = {"question": "hypertension", "retriever": "nonexistent_retriever"}
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 422
        err = resp.json()
        assert err["error"] == "ValidationError"
        assert "invalid retriever" in err["message"]

    def test_query_unsupported_filter_fields_rejected(self, client: TestClient) -> None:
        payload = {
            "question": "hypertension",
            "top_k": 5,
            "unsupported_extra_field": "malicious_payload",
        }
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 422
        err = resp.json()
        assert err["error"] == "ValidationError"
        assert "extra" in err["message"].lower() or "unsupported" in err["message"].lower()

    def test_query_invalid_filter_enum_fails_validation(self, client: TestClient) -> None:
        payload = {
            "question": "hypertension",
            "source_organization": ["INVALID_PUBLISHER_123"],
        }
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 422

    def test_query_missing_body_fails_validation(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query",
            headers={"Content-Type": "application/json"},
            content=b"",
        )
        assert resp.status_code == 422
        err = resp.json()
        assert err["error"] == "ValidationError"
        assert "trace_id" in err

    def test_query_with_source_org_filter(self, client: TestClient) -> None:
        payload = {
            "question": "hypertension guidelines",
            "top_k": 2,
            "source_organization": ["CDC"],
            "retriever": "dense",
        }
        resp = client.post("/api/v1/query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval"]["retrieved_chunks"] >= 0
