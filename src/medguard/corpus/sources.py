"""The editable corpus definition: parsing and validating ``data/sources.yaml``.

Why YAML and not Python
-----------------------
The corpus is data, not logic. Hard-coding a URL list inside a script means changing
the corpus requires editing code, which drags every corpus change through import
paths, linting and type checking for no benefit. Externalising it means a reviewer can
see the whole corpus in one file and diff it meaningfully.

Validation is strict on purpose. A malformed entry discovered here produces one clear
error before any network traffic; the same entry discovered later produces a
half-downloaded corpus whose manifest disagrees with the files on disk.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medguard.errors import CorpusError
from medguard.models import DocumentType, SourceOrganization

__all__ = ["SourceDocument", "SourceList", "load_sources"]

#: Document ids become filenames, database primary keys, chunk id prefixes and Qdrant
#: payload values. Restricting the character set now avoids quoting problems in four
#: places later.
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{2,63}$")


class SourceDocument(BaseModel):
    """One document the corpus is defined to contain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(description="Stable slug; also the local filename stem.")
    title: str = Field(min_length=3)
    source_url: str = Field(description="Direct URL to the PDF as published.")
    source_organization: SourceOrganization
    document_type: DocumentType
    local_filename: str = Field(description="Filename under the corpus data directory.")
    license_or_usage_note: str = Field(
        min_length=3,
        description="Verbatim usage terms as stated by the publisher, with the page "
        "they were read from. Never a guess -- see the 'never invent metadata' rule.",
    )
    publication_date: date | None = Field(
        default=None,
        description="Only where the publisher states a full date. A catalogue "
        "placeholder such as 'YYYY/01/01' is not a publication date; record None.",
    )
    version_or_revision: str | None = Field(default=None)
    retired: bool = Field(
        default=False,
        description="True when the source URL no longer serves the document. The entry "
        "is kept so an already-downloaded copy still has provenance, but no download "
        "is attempted.",
    )
    retired_reason: str | None = Field(default=None)
    notes: str | None = Field(default=None, description="Free-text provenance commentary.")

    @field_validator("document_id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            msg = (
                f"document_id {v!r} must be 3-64 chars of lowercase letters, digits and "
                "underscores, starting with a letter or digit"
            )
            raise ValueError(msg)
        return v

    @field_validator("source_url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            msg = f"source_url must be http(s); got {v[:40]!r}"
            raise ValueError(msg)
        return v

    @field_validator("local_filename")
    @classmethod
    def _check_filename(cls, v: str) -> str:
        if not v.endswith(".pdf"):
            msg = f"local_filename must end in .pdf; got {v!r}"
            raise ValueError(msg)
        # A filename carrying a path separator would escape the corpus directory.
        if "/" in v or "\\" in v or v.startswith("."):
            msg = f"local_filename must be a bare filename; got {v!r}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _retired_needs_a_reason(self) -> SourceDocument:
        if self.retired and not (self.retired_reason or "").strip():
            msg = f"{self.document_id}: retired entries must state retired_reason"
            raise ValueError(msg)
        return self


class SourceList(BaseModel):
    """The whole corpus definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    documents: list[SourceDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_duplicates(self) -> SourceList:
        for field in ("document_id", "local_filename", "source_url"):
            seen: dict[str, int] = {}
            for doc in self.documents:
                value = str(getattr(doc, field))
                seen[value] = seen.get(value, 0) + 1
            dupes = sorted(k for k, n in seen.items() if n > 1)
            if dupes:
                msg = f"duplicate {field} in sources: {dupes}"
                raise ValueError(msg)
        return self

    @property
    def active(self) -> list[SourceDocument]:
        """Entries that should actually be downloaded."""
        return [d for d in self.documents if not d.retired]

    def by_id(self, document_id: str) -> SourceDocument | None:
        return next((d for d in self.documents if d.document_id == document_id), None)


def load_sources(path: Path) -> SourceList:
    """Read and validate the corpus definition.

    Raises :class:`~medguard.errors.CorpusError` for every failure mode -- missing
    file, unparseable YAML, wrong shape -- so callers handle one exception type rather
    than three unrelated ones from three libraries.
    """
    if not path.is_file():
        msg = f"corpus source list not found: {path}"
        raise CorpusError(msg)

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"{path} is not valid YAML: {exc}"
        raise CorpusError(msg) from exc
    except OSError as exc:
        msg = f"could not read {path}: {exc}"
        raise CorpusError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"{path} must contain a mapping at the top level, got {type(raw).__name__}"
        raise CorpusError(msg)

    try:
        return SourceList.model_validate(raw)
    except ValueError as exc:
        msg = f"{path} failed validation: {exc}"
        raise CorpusError(msg) from exc
