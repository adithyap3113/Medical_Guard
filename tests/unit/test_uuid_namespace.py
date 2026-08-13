"""The UUID namespace must never change.

This is the most consequential constant in the project and the one whose corruption is
hardest to notice. Qdrant point ids are derived from it by ``uuid5``; change it and
every id changes, the old points stay in the collection, the new ones land beside them,
and ingestion stops being idempotent for every document already indexed. Nothing raises.
No test elsewhere fails. The symptom is duplicate retrieval results, weeks later.

So the value is asserted against a literal duplicated here on purpose. Editing the
constant without editing this file -- which is what an accidental change looks like --
fails immediately with an explanation.
"""

from __future__ import annotations

import ast
import uuid
from pathlib import Path

from medguard.config import project_root
from medguard.constants import MEDGUARD_UUID_NAMESPACE

#: Deliberately duplicated rather than imported. Importing it would make this test
#: assert that a value equals itself, which is not a test.
EXPECTED = "759d085e-6dcc-4956-bb53-900708f531ef"


def called_functions(source: str) -> set[str]:
    """Names of functions actually *called* in ``source``.

    Parsed rather than grepped. A substring search cannot tell code from the prose
    explaining it, and these modules deliberately document the mechanisms they must not
    use -- a grep for "uuid4" hits the comment that says where the literal came from.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def reads_the_environment(source: str) -> bool:
    """True if ``source`` contains real code that reads an environment variable."""
    return any(
        (isinstance(node, ast.Attribute) and node.attr == "environ")
        or (isinstance(node, ast.Name) and node.id == "environ")
        for node in ast.walk(ast.parse(source))
    )


class TestTheNamespaceIsFrozen:
    def test_the_value_has_not_changed(self) -> None:
        assert str(MEDGUARD_UUID_NAMESPACE) == EXPECTED, (
            "MEDGUARD_UUID_NAMESPACE changed. Every Qdrant point id derives from it via "
            "uuid5, so this invalidates every id in every existing collection without "
            "raising anything. If the change is deliberate, drop and rebuild every "
            "collection, then update EXPECTED here. If it is not, revert it."
        )

    def test_it_is_a_real_uuid_object(self) -> None:
        """A str would work with uuid5 by accident and break the type contract."""
        assert isinstance(MEDGUARD_UUID_NAMESPACE, uuid.UUID)

    def test_it_is_committed_as_a_literal_not_computed(self) -> None:
        """A generated namespace differs per process, which is the whole failure mode."""
        source = (project_root() / "src" / "medguard" / "constants.py").read_text(encoding="utf-8")
        assert f'uuid.UUID("{EXPECTED}")' in source

        called = called_functions(source)
        for generator in ("uuid4", "uuid1", "uuid3", "getenv"):
            assert generator not in called, f"namespace must not be derived from {generator}()"
        assert not reads_the_environment(source), (
            "namespace must not be read from the environment -- that makes point ids "
            "differ between two machines pointed at the same collection"
        )


class TestDerivedIdsAreDeterministic:
    def test_the_same_chunk_id_always_maps_to_the_same_point_id(self) -> None:
        """The property that makes re-ingestion an upsert rather than a duplicate."""
        chunk_id = "cdc_anthrax_2023::0007"
        a = uuid.uuid5(MEDGUARD_UUID_NAMESPACE, chunk_id)
        b = uuid.uuid5(MEDGUARD_UUID_NAMESPACE, chunk_id)
        assert a == b

    def test_different_chunk_ids_map_to_different_point_ids(self) -> None:
        first = uuid.uuid5(MEDGUARD_UUID_NAMESPACE, "cdc_anthrax_2023::0007")
        second = uuid.uuid5(MEDGUARD_UUID_NAMESPACE, "cdc_anthrax_2023::0008")
        assert first != second

    def test_document_versions_do_not_collide(self) -> None:
        """Two editions of one guideline must not share a point id at the same position."""
        v1 = uuid.uuid5(MEDGUARD_UUID_NAMESPACE, "icmr_amr_2019::0000")
        v2 = uuid.uuid5(MEDGUARD_UUID_NAMESPACE, "icmr_amr_2022::0000")
        assert v1 != v2

    def test_a_different_namespace_produces_different_ids(self) -> None:
        """Demonstrates the damage the constant-unchanged test exists to prevent."""
        other = uuid.UUID("00000000-0000-4000-8000-000000000000")
        chunk_id = "cdc_anthrax_2023::0007"
        assert uuid.uuid5(MEDGUARD_UUID_NAMESPACE, chunk_id) != uuid.uuid5(other, chunk_id)

    def test_the_result_is_acceptable_to_qdrant_as_a_point_id(self) -> None:
        """Qdrant rejects arbitrary strings; a UUID is one of the two forms it takes."""
        point_id = uuid.uuid5(MEDGUARD_UUID_NAMESPACE, "cdc_anthrax_2023::0007")
        assert uuid.UUID(str(point_id)) == point_id
        assert point_id.version == 5


class TestPhase3ADoesNotUseItYet:
    def test_nothing_in_src_derives_point_ids_yet(self) -> None:
        """Phase 3A declares the namespace; Phase 3B is what may consume it.

        Parsed, not grepped: both `constants.py` and `schema.py` document the uuid5
        bridge in prose, and a substring search cannot tell an explanation from a call.
        """
        src = project_root() / "src"
        users = [
            path.relative_to(src).as_posix()
            for path in src.rglob("*.py")
            if "uuid5" in called_functions(path.read_text(encoding="utf-8"))
        ]
        assert users == [], f"uuid5 is a Phase 3B concern; found in {users}"

    def test_the_constant_lives_outside_config(self) -> None:
        """In config.py it would look tunable, which is exactly what it is not."""
        assert Path(project_root() / "src" / "medguard" / "constants.py").is_file()
        config = (project_root() / "src" / "medguard" / "config.py").read_text(encoding="utf-8")
        assert "MEDGUARD_UUID_NAMESPACE" not in config
