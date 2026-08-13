"""The corpus manifest: what the system was actually built on.

``data/manifest.json`` is committed; the PDFs are not. That asymmetry is the design.
A clone of this repository cannot reproduce 35 MB of government PDFs from git, but it
can state exactly which documents the corpus contained, what each one hashed to, where
it came from and under what terms -- and it can prove that the files on a given machine
are those same documents.

Two rules govern every field:

* **Never invent metadata.** A field the publisher does not state is ``null``, with the
  reason recorded in ``data/raw/SOURCES.md``. A plausible-looking guessed date is worse
  than an absent one, because nothing downstream can tell the difference.
* **Hashes are computed over raw file bytes**, so they are reproducible by anyone with
  ``sha256sum`` and independent of anything this codebase does.

``page_count`` is present in the schema but ``null`` until a PDF parser exists to count
pages. Declaring the field early keeps the schema stable across that boundary; filling
it with a guess would not.

Who may write this file
-----------------------
Exactly two callers, and this is a rule rather than an observation:

* :func:`write_manifest`, via :func:`build_manifest`, on ``medguard corpus download``
* ``medguard corpus sync`` -- in adopt mode, and in the ``--backfill-*`` modes

Nothing else, ever. In particular ``medguard ingest`` must not write here. The manifest
is the committed record of *what the corpus is*; ingestion is a consumer of that record,
and a consumer that rewrites its own input has no independent check left. If ingestion
learns something the manifest lacks -- a page count, say -- it writes that to the
database, and a human promotes it with an explicit ``corpus sync --backfill-*``. That
keeps every change to committed provenance a reviewable diff rather than a side effect
of a pipeline run.

Two rules govern backfill itself:

* **Null-only.** Backfill fills a field that is ``null``. It never overwrites a value,
  because a manifest value came from the publisher and a database value came from
  parsing a file, and the publisher wins.
* **``generated_at`` moves only on a real change.** A backfill that changes nothing
  leaves the file untouched, byte for byte, so ``git diff`` answers "did this run alter
  the corpus record?" without anyone having to read the contents.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medguard.config import Settings, get_settings
from medguard.corpus.download import DownloadOutcome, DownloadStatus
from medguard.corpus.sources import SourceDocument, SourceList
from medguard.errors import ManifestError
from medguard.models import DocumentType, SourceOrganization

__all__ = [
    "BACKFILL_FLAGS",
    "BackfillChange",
    "BackfillField",
    "IssueLevel",
    "Manifest",
    "ManifestDocument",
    "VerificationIssue",
    "VerificationReport",
    "apply_backfill",
    "build_manifest",
    "load_manifest",
    "plan_backfill",
    "sha256_file",
    "source_to_manifest_entry",
    "verify_corpus",
    "write_manifest",
]

MANIFEST_SCHEMA_VERSION = 1

_HASH_CHUNK_BYTES = 64 * 1024
_PDF_MAGIC = b"%PDF-"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _carry_forward(authoritative: date | None, carried: date | None) -> date | None:
    """Prefer a publisher-stated value, fall back to one carried from a prior manifest."""
    return authoritative if authoritative is not None else carried


def sha256_file(path: Path) -> str:
    """Lowercase hex SHA-256 of a file's raw bytes.

    Streamed rather than read whole: the corpus contains a 22 MB report, and a
    ``read()``-then-hash implementation would scale with the largest document rather
    than with a fixed buffer.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


class ManifestDocument(BaseModel):
    """The provenance record for one corpus document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    title: str
    source_url: str
    source_organization: SourceOrganization
    document_type: DocumentType
    publication_date: date | None = Field(
        default=None, description="None means the publisher states no full date."
    )
    license_or_usage_note: str
    local_filename: str
    sha256: str = Field(description="Lowercase hex, no algorithm prefix.")
    file_size_bytes: int = Field(ge=0)
    page_count: int | None = Field(
        default=None, ge=0, description="Populated by Phase 3 ingestion, null before it."
    )
    retrieved_at: datetime
    version_or_revision: str | None = None

    @field_validator("sha256")
    @classmethod
    def _check_hash(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            msg = f"sha256 must be 64 lowercase hex characters; got {v[:16]!r}…"
            raise ValueError(msg)
        return v


class Manifest(BaseModel):
    """The authoritative record of the corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=MANIFEST_SCHEMA_VERSION, ge=1)
    generated_at: datetime
    documents: list[ManifestDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicates(self) -> Manifest:
        for field in ("document_id", "local_filename", "sha256"):
            seen: dict[str, int] = {}
            for doc in self.documents:
                value = str(getattr(doc, field))
                seen[value] = seen.get(value, 0) + 1
            dupes = sorted(k for k, n in seen.items() if n > 1)
            if dupes:
                # Duplicate sha256 means the same bytes are indexed twice under two
                # ids, which double-counts a source in every retrieval metric.
                msg = f"duplicate {field} in manifest: {dupes}"
                raise ValueError(msg)
        return self

    def by_id(self, document_id: str) -> ManifestDocument | None:
        return next((d for d in self.documents if d.document_id == document_id), None)


class IssueLevel(StrEnum):
    """Severity of a verification finding.

    The distinction is actionable, and only ``ERROR`` affects the exit code:

    * ``ERROR`` -- the corpus on this machine is not the corpus the manifest describes.
      Nothing downstream should trust it. Phase 3A widened this to include a missing or
      unrecorded file: with the corpus spread across four publishers, "some documents
      are absent" stopped being a benign state and became the difference between an
      index over 20 documents and an index over 14, with no visible sign of which.
    * ``WARNING`` -- the manifest is internally honest but incomplete in an expected
      way. A ``null`` ``publication_date`` on a document whose publisher states no date
      is the canonical case: correct under "never invent metadata", and not a defect.
    * ``INFO`` -- nothing is wrong; there is work available. Reported only under
      ``--verbose``, because a clean run should stay quiet.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class VerificationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: IssueLevel
    document_id: str | None = None
    detail: str


class VerificationReport(BaseModel):
    """Outcome of checking the manifest against the files on disk."""

    checked: int = Field(default=0, ge=0)
    issues: list[VerificationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[VerificationIssue]:
        return [i for i in self.issues if i.level is IssueLevel.ERROR]

    @property
    def warnings(self) -> list[VerificationIssue]:
        return [i for i in self.issues if i.level is IssueLevel.WARNING]

    @property
    def notes(self) -> list[VerificationIssue]:
        return [i for i in self.issues if i.level is IssueLevel.INFO]

    @property
    def ok(self) -> bool:
        return not self.errors

    def exit_code(self) -> int:
        return 1 if self.errors else 0


def load_manifest(path: Path) -> Manifest:
    """Read and validate the manifest, normalising every failure to ``ManifestError``."""
    if not path.is_file():
        msg = f"manifest not found: {path}"
        raise ManifestError(msg)
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path} is not valid JSON: {exc}"
        raise ManifestError(msg) from exc
    except OSError as exc:
        msg = f"could not read {path}: {exc}"
        raise ManifestError(msg) from exc
    try:
        return Manifest.model_validate(raw)
    except ValueError as exc:
        msg = f"{path} failed validation: {exc}"
        raise ManifestError(msg) from exc


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Serialise the manifest deterministically.

    Documents are sorted by id and the file ends with a newline, so a regenerated
    manifest produces a diff that reflects real corpus changes rather than dictionary
    ordering. ``ensure_ascii=False`` keeps the em dashes in CDC titles readable instead
    of expanding them into escape sequences.
    """
    ordered = manifest.model_copy(
        update={"documents": sorted(manifest.documents, key=lambda d: d.document_id)}
    )
    payload = ordered.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _retrieved_at_for(
    outcome: DownloadOutcome,
    previous: ManifestDocument | None,
    file_path: Path,
) -> datetime:
    """Decide when this file was retrieved, without inventing a timestamp.

    Freshly downloaded files use now. A file that was already present keeps whatever
    the previous manifest recorded, provided the bytes have not changed. With no prior
    record, the filesystem modification time is used: for a downloaded file that *is*
    the moment the bytes were written, which is a measurement rather than a guess.
    """
    if outcome.status is DownloadStatus.DOWNLOADED:
        return _utcnow()
    if previous is not None and previous.sha256 == outcome.sha256:
        return previous.retrieved_at
    try:
        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
    except OSError:
        return _utcnow()


def build_manifest(
    sources: SourceList,
    outcomes: list[DownloadOutcome],
    settings: Settings | None = None,
    *,
    previous: Manifest | None = None,
) -> Manifest:
    """Assemble a manifest from the source definitions and a download run.

    Only documents that are actually on disk get an entry: a manifest listing a file
    that was never fetched would assert provenance for bytes nobody has.
    """
    s = settings if settings is not None else get_settings()
    raw_dir = s.corpus.raw_dir
    by_id = {o.document_id: o for o in outcomes}

    documents: list[ManifestDocument] = []
    for source in sources.documents:
        outcome = by_id.get(source.document_id)
        file_path = raw_dir / source.local_filename

        if outcome is not None and outcome.ok and outcome.sha256 is not None:
            digest = outcome.sha256
            size = outcome.file_size_bytes or 0
        elif file_path.is_file():
            # Not part of this run (retired, or filtered out by --only) but present.
            digest = sha256_file(file_path)
            size = file_path.stat().st_size
        else:
            continue

        prior = previous.by_id(source.document_id) if previous is not None else None
        documents.append(
            ManifestDocument(
                document_id=source.document_id,
                title=source.title,
                source_url=source.source_url,
                source_organization=source.source_organization,
                document_type=source.document_type,
                # The publisher wins when it states a date; a prior manifest value
                # (which can only have come from a backfill) fills the gap when it does
                # not. Taking `source.publication_date` unconditionally would discard
                # every backfilled date on the next download run.
                publication_date=_carry_forward(
                    source.publication_date, prior.publication_date if prior else None
                ),
                license_or_usage_note=source.license_or_usage_note,
                local_filename=source.local_filename,
                sha256=digest,
                file_size_bytes=size,
                page_count=prior.page_count if prior is not None else None,
                retrieved_at=_retrieved_at_for(
                    outcome
                    or DownloadOutcome(
                        document_id=source.document_id,
                        local_filename=source.local_filename,
                        status=DownloadStatus.PRESENT,
                        sha256=digest,
                        file_size_bytes=size,
                    ),
                    prior,
                    file_path,
                ),
                version_or_revision=source.version_or_revision,
            )
        )

    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        generated_at=_utcnow(),
        documents=documents,
    )


def verify_corpus(
    manifest: Manifest,
    settings: Settings | None = None,
    *,
    database_values: dict[str, dict[BackfillField, int | date | None]] | None = None,
) -> VerificationReport:
    """Check the manifest against the files on disk.

    Never raises: a verification command that dies on the first bad file cannot tell
    you about the other eleven.

    ``database_values`` enables the ``--verbose`` freshness comparison. Pass what the
    ``documents`` table holds and any field the database knows and the manifest does not
    is reported as an ``INFO`` -- work available, not a defect. Omit it and that section
    is skipped entirely, which keeps ``verify`` runnable with no database at all.
    """
    s = settings if settings is not None else get_settings()
    raw_dir = s.corpus.raw_dir
    issues: list[VerificationIssue] = []
    recorded_filenames: set[str] = set()
    digests_seen: dict[str, str] = {}

    for doc in manifest.documents:
        recorded_filenames.add(doc.local_filename)
        path = raw_dir / doc.local_filename

        if doc.publication_date is None:
            issues.append(
                VerificationIssue(
                    level=IssueLevel.WARNING,
                    document_id=doc.document_id,
                    detail="publication_date is null. Expected where the publisher "
                    "states no full date; see data/raw/SOURCES.md for the reason "
                    "recorded against this document.",
                )
            )

        if not path.is_file():
            issues.append(
                VerificationIssue(
                    level=IssueLevel.ERROR,
                    document_id=doc.document_id,
                    detail=f"recorded in the manifest but absent from {raw_dir}. "
                    "Run `medguard corpus download`.",
                )
            )
            continue

        try:
            with path.open("rb") as handle:
                magic = handle.read(len(_PDF_MAGIC))
            size = path.stat().st_size
            digest = sha256_file(path)
        except OSError as exc:
            issues.append(
                VerificationIssue(
                    level=IssueLevel.ERROR,
                    document_id=doc.document_id,
                    detail=f"unreadable: {type(exc).__name__}: {exc}",
                )
            )
            continue

        if magic != _PDF_MAGIC:
            issues.append(
                VerificationIssue(
                    level=IssueLevel.ERROR,
                    document_id=doc.document_id,
                    detail=f"not a PDF: file starts with {magic!r}, expected {_PDF_MAGIC!r}",
                )
            )
        if digest != doc.sha256:
            issues.append(
                VerificationIssue(
                    level=IssueLevel.ERROR,
                    document_id=doc.document_id,
                    detail=f"sha256 mismatch: manifest records {doc.sha256[:16]}…, "
                    f"file hashes to {digest[:16]}…. The file changed or is corrupt.",
                )
            )
        if size != doc.file_size_bytes:
            issues.append(
                VerificationIssue(
                    level=IssueLevel.ERROR,
                    document_id=doc.document_id,
                    detail=f"size mismatch: manifest records {doc.file_size_bytes} bytes, "
                    f"file is {size} bytes",
                )
            )

        # Two ids over identical bytes double-counts one source in every retrieval
        # metric. The Manifest model already rejects duplicate recorded hashes; this
        # catches the case where two *files* have converged on the same content.
        if digest in digests_seen:
            issues.append(
                VerificationIssue(
                    level=IssueLevel.ERROR,
                    document_id=doc.document_id,
                    detail=f"duplicate content: hashes identically to "
                    f"{digests_seen[digest]}. The same document is recorded twice.",
                )
            )
        else:
            digests_seen[digest] = doc.document_id

    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.pdf")):
            if path.name not in recorded_filenames:
                issues.append(
                    VerificationIssue(
                        level=IssueLevel.ERROR,
                        detail=f"{path.name} is present on disk but has no manifest entry. "
                        "An unrecorded file has no provenance, so nothing can cite it. "
                        "Add it to data/sources.yaml, then run `medguard corpus sync`.",
                    )
                )
        for path in sorted(raw_dir.glob("*.part")):
            issues.append(
                VerificationIssue(
                    level=IssueLevel.WARNING,
                    detail=f"{path.name} is a leftover partial download; it is safe to delete.",
                )
            )

    if database_values is not None:
        issues.extend(_freshness_notes(manifest, database_values))

    return VerificationReport(checked=len(manifest.documents), issues=issues)


def _freshness_notes(
    manifest: Manifest,
    database_values: dict[str, dict[BackfillField, int | date | None]],
) -> list[VerificationIssue]:
    """Report fields the database knows and the manifest does not.

    Deliberately one-directional. The reverse gap -- manifest populated, database null
    -- means the database has not been rebuilt from the manifest yet, which is a
    different problem with a different fix and does not belong in this message.
    """
    return [
        VerificationIssue(
            level=IssueLevel.INFO,
            document_id=change.document_id,
            detail=f"database has newer operational values; {change.field.value} is null "
            f"in the manifest and {change.value} in the database. Run "
            f"`medguard corpus sync {BACKFILL_FLAGS[change.field]}` to promote.",
        )
        for change in plan_backfill(manifest, database_values, set(BackfillField))
    ]


class BackfillField(StrEnum):
    """A manifest field that operational work can discover after the fact.

    A closed set, not free text. These are precisely the fields that ``sources.yaml``
    cannot state because they are properties of the *file* rather than of the
    publisher's page, and that ingestion therefore learns first. Everything else in a
    manifest entry is transcribed from a publisher and is not backfillable at all.
    """

    PAGE_COUNT = "page_count"
    PUBLICATION_DATE = "publication_date"


#: The CLI flag that promotes each field, so `verify --verbose` can name the exact
#: command to run rather than leaving the reader to guess the pluralisation.
BACKFILL_FLAGS: dict[BackfillField, str] = {
    BackfillField.PAGE_COUNT: "--backfill-page-counts",
    BackfillField.PUBLICATION_DATE: "--backfill-publication-dates",
}


class BackfillChange(BaseModel):
    """One field on one document that a backfill would fill in."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    field: BackfillField
    value: int | date

    def describe(self) -> str:
        return f"{self.document_id}: {self.field.value} null → {self.value}"


def plan_backfill(
    manifest: Manifest,
    values: dict[str, dict[BackfillField, int | date | None]],
    fields: set[BackfillField],
) -> list[BackfillChange]:
    """Work out what a backfill would change, without changing anything.

    Separated from :func:`apply_backfill` so the CLI can show a user exactly what is
    about to happen to a committed file and let them decline it, and so the null-only
    rule is testable without going near a database.

    The rule, enforced here and nowhere else: a change is proposed only when the
    manifest field is ``null`` **and** the incoming value is not. A populated manifest
    field is never a candidate, whatever the database says. Provenance transcribed from
    a publisher outranks a value derived from parsing a PDF, and if the two genuinely
    disagree that is a finding for a human, not something to resolve by overwriting.
    """
    changes: list[BackfillChange] = []
    for doc in manifest.documents:
        available = values.get(doc.document_id)
        if not available:
            continue
        for field in sorted(fields, key=lambda f: f.value):
            if getattr(doc, field.value) is not None:
                continue
            incoming = available.get(field)
            if incoming is None:
                continue
            changes.append(BackfillChange(document_id=doc.document_id, field=field, value=incoming))
    return changes


def apply_backfill(manifest: Manifest, changes: list[BackfillChange]) -> Manifest:
    """Return a manifest with ``changes`` applied, and ``generated_at`` restamped.

    An empty change list returns the manifest **unchanged**, including its original
    ``generated_at``. That is the whole point: ``generated_at`` is what a reviewer reads
    to answer "did this run touch the corpus record?", and a timestamp that moves on
    every no-op run cannot answer it. Callers must not write the file when nothing
    changed -- see ``medguard corpus sync``.
    """
    if not changes:
        return manifest

    by_document: dict[str, dict[str, int | date]] = {}
    for change in changes:
        by_document.setdefault(change.document_id, {})[change.field.value] = change.value

    updated = [
        doc.model_copy(update=by_document[doc.document_id])
        if doc.document_id in by_document
        else doc
        for doc in manifest.documents
    ]
    return manifest.model_copy(update={"documents": updated, "generated_at": _utcnow()})


def source_to_manifest_entry(
    source: SourceDocument,
    path: Path,
    *,
    retrieved_at: datetime | None = None,
    page_count: int | None = None,
    publication_date: date | None = None,
) -> ManifestDocument:
    """Build a single manifest entry from a source definition and a file on disk.

    Used by ``medguard corpus sync`` to adopt a file that exists but was never
    recorded, without rebuilding the whole manifest.

    ``page_count`` and ``publication_date`` are carry-forward slots for values that
    ``sources.yaml`` cannot supply. Both used to be hard-coded ``None`` here, which was
    a real bug rather than a simplification: a backfilled page count survived
    ``corpus download`` (:func:`build_manifest` preserved it) but was silently erased by
    the next ``corpus sync`` that rebuilt the entry from its source. The two paths now
    agree, and ``tests/unit/test_backfill.py`` holds them to it.

    The publisher still wins: an explicit ``source.publication_date`` overrides the
    carried value, because ``sources.yaml`` is transcribed from a publisher's page and a
    carried value was derived from a file.
    """
    return ManifestDocument(
        document_id=source.document_id,
        title=source.title,
        source_url=source.source_url,
        source_organization=source.source_organization,
        document_type=source.document_type,
        publication_date=_carry_forward(source.publication_date, publication_date),
        license_or_usage_note=source.license_or_usage_note,
        local_filename=source.local_filename,
        sha256=sha256_file(path),
        file_size_bytes=path.stat().st_size,
        page_count=page_count,
        retrieved_at=retrieved_at or datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        version_or_revision=source.version_or_revision,
    )
