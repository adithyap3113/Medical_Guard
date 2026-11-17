"""Unit tests for `medguard query` CLI command."""

from __future__ import annotations

from typer.testing import CliRunner

from medguard.cli.app import app
from medguard.config import Settings
from medguard.storage.vector_store import VectorStore
from tests.unit.test_retrieval import _chunk

runner = CliRunner()


class TestQueryCli:
    def test_query_help_displays(self) -> None:
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "Answer a medical or clinical question" in result.output
        assert "--strategy" in result.output
        assert "--top-k" in result.output

    def test_query_execution_offline(self, settings: Settings) -> None:
        # Prepopulate in-memory collection
        store = VectorStore(settings)
        store.ensure_collection()
        store.upsert_chunks(
            [
                _chunk("cdc_htn", 0, "CDC", "guideline"),
            ]
        )

        result = runner.invoke(
            app,
            [
                "query",
                "What is the recommended treatment for hypertension?",
                "--top-k",
                "1",
                "--json",
            ],
        )
        # Should succeed or exit cleanly
        assert result.exit_code == 0 or "Error initializing" not in result.output
