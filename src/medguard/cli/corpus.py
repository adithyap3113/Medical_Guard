"""``medguard corpus`` -- acquire, record and verify the source documents.

Three verbs, deliberately separate:

* ``download`` fetches what ``data/sources.yaml`` defines and rewrites the manifest.
* ``verify`` checks the manifest against the files on disk and exits non-zero on any
  integrity failure, so it is usable as a CI gate.
* ``sync`` edits the manifest, in one of two modes.

The split matters because the three have very different blast radii. ``verify`` is
read-only and safe to run constantly. ``download`` writes files. ``sync`` edits the
committed provenance record, which is why every mode asks before writing.

``sync``, and why it now does two things
----------------------------------------
Phase 2 gave ``sync`` exactly one job -- *adopt* a file that exists on disk but was
never recorded -- and one absolute rule: never modify an entry that already exists.

Phase 3A widens that rule rather than breaking it. ``--backfill-page-counts`` and
``--backfill-publication-dates`` do modify existing entries, but only where the field
is ``null``, and only with a value the database learned by processing the file. The
invariant that actually mattered is intact: **``sync`` never overwrites a value.** An
entry that *disagrees* with its file is still an integrity failure for ``verify`` to
report, not something to quietly paper over.

The direction is deliberate. Ingestion writes what it learns to the database; a human
promotes it into the committed manifest with an explicit command. That is what keeps a
change to provenance a reviewable diff instead of a side effect of a pipeline run --
see the "who may write this file" section of :mod:`medguard.corpus.manifest`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table
from rich.text import Text

from medguard.cli.bootstrap import bootstrap, console, err_console

if TYPE_CHECKING:
    # Type-only: the runtime imports stay inside the command bodies so that
    # `medguard --help` does not pay for SQLAlchemy and pydantic-settings.
    from datetime import date

    from medguard.config import Settings
    from medguard.corpus.manifest import BackfillField, Manifest

    OperationalValues = dict[str, dict[BackfillField, int | date | None]]

__all__ = ["corpus_app"]

corpus_app = typer.Typer(
    name="corpus",
    help="Acquire, record and verify the source-document corpus.",
    no_args_is_help=True,
)


def _status_text(label: str, style: str) -> Text:
    return Text(label, style=style)


def _read_operational_values(settings: Settings) -> OperationalValues | None:
    """Read the backfillable columns out of the ``documents`` table.

    Returns ``None`` rather than raising when the database is missing, unmigrated or
    unreachable. Both callers treat that as "no comparison available": neither
    ``verify`` nor ``sync`` has any business failing because an *optional* enrichment
    source is absent, and on a fresh clone it always will be.
    """
    from sqlalchemy import select

    from medguard.corpus.manifest import BackfillField
    from medguard.storage.database import Database
    from medguard.storage.schema import DocumentRow

    database = Database(settings)
    try:
        with database.session() as session:
            rows = session.execute(
                select(
                    DocumentRow.document_id, DocumentRow.page_count, DocumentRow.publication_date
                )
            ).all()
    except Exception:  # noqa: BLE001 - any failure means "no values available"
        return None
    finally:
        database.dispose()

    return {
        row.document_id: {
            BackfillField.PAGE_COUNT: row.page_count,
            BackfillField.PUBLICATION_DATE: row.publication_date,
        }
        for row in rows
    }


@corpus_app.command("download")
def download(
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download documents that are already present."),
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Restrict the run to these document ids. Repeatable."),
    ] = None,
    write_manifest_file: Annotated[
        bool,
        typer.Option(
            "--write-manifest/--no-write-manifest",
            help="Rewrite data/manifest.json from the result of this run.",
        ),
    ] = True,
) -> None:
    """Download every document defined in data/sources.yaml."""
    bootstrap()
    from medguard.config import get_settings
    from medguard.corpus.download import DownloadStatus, download_corpus
    from medguard.corpus.manifest import build_manifest, load_manifest, write_manifest
    from medguard.corpus.sources import load_sources
    from medguard.errors import CorpusError

    settings = get_settings()
    try:
        sources = load_sources(settings.corpus.sources_path)
    except CorpusError as exc:
        err_console.print(f"[red]{exc.detail}[/]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Corpus: [bold]{len(sources.documents)}[/] entries "
        f"([bold]{len(sources.active)}[/] active) → {settings.corpus.raw_dir}"
    )

    outcomes = download_corpus(sources, settings, force=force, only=only or None)

    styles = {
        DownloadStatus.DOWNLOADED: ("DOWNLOADED", "bold green"),
        DownloadStatus.PRESENT: ("PRESENT", "green"),
        DownloadStatus.RETIRED: ("RETIRED", "dim"),
        DownloadStatus.FAILED: ("FAILED", "bold red"),
    }

    table = Table(title="Corpus download", header_style="bold", expand=True)
    table.add_column("Status", width=11, no_wrap=True)
    table.add_column("Document", width=32, no_wrap=True)
    table.add_column("Detail", overflow="fold")
    table.add_column("MB", justify="right", width=7, no_wrap=True)

    for outcome in outcomes:
        label, style = styles[outcome.status]
        size = (
            f"{outcome.file_size_bytes / 1024 / 1024:.2f}"
            if outcome.file_size_bytes is not None
            else ""
        )
        table.add_row(
            _status_text(label, style),
            outcome.document_id,
            outcome.error or outcome.local_filename,
            size,
        )
    console.print(table)

    failed = [o for o in outcomes if o.status is DownloadStatus.FAILED]
    usable = [o for o in outcomes if o.ok]
    console.print(
        f"[green]{len(usable)} available[/] · [red]{len(failed)} failed[/] · "
        f"[dim]{len(outcomes) - len(usable) - len(failed)} skipped[/]"
    )

    if write_manifest_file:
        previous = None
        if settings.corpus.manifest_path.is_file():
            try:
                previous = load_manifest(settings.corpus.manifest_path)
            except CorpusError as exc:
                err_console.print(
                    f"[yellow]Existing manifest unusable, rebuilding: {exc.detail}[/]"
                )
        manifest = build_manifest(sources, outcomes, settings, previous=previous)
        write_manifest(manifest, settings.corpus.manifest_path)
        console.print(
            f"Manifest: [bold]{len(manifest.documents)}[/] documents → "
            f"{settings.corpus.manifest_path}"
        )

    # Non-zero on failure so this is usable in CI. The Phase 1 script always exited 0,
    # which made "did the corpus download?" unanswerable by anything but a human.
    raise typer.Exit(code=1 if failed else 0)


@corpus_app.command("verify")
def verify(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Also compare the manifest against the database and report values "
            "the database knows that the manifest does not.",
        ),
    ] = False,
) -> None:
    """Check every manifest entry against the file on disk. Exits 1 on any error."""
    bootstrap()
    from medguard.config import get_settings
    from medguard.corpus.manifest import IssueLevel, load_manifest, verify_corpus
    from medguard.errors import CorpusError

    settings = get_settings()
    try:
        manifest = load_manifest(settings.corpus.manifest_path)
    except CorpusError as exc:
        err_console.print(f"[red]{exc.detail}[/]")
        raise typer.Exit(code=1) from exc

    database_values = None
    if verbose:
        database_values = _read_operational_values(settings)
        if database_values is None:
            err_console.print(
                "[yellow]Database unreadable; skipping the freshness comparison. "
                "Integrity checks below are unaffected.[/]"
            )

    report = verify_corpus(manifest, settings, database_values=database_values)

    table = Table(title="Corpus verification", header_style="bold", expand=True)
    table.add_column("Level", width=8, no_wrap=True)
    table.add_column("Document", width=32, no_wrap=True)
    table.add_column("Detail", overflow="fold")

    styles = {
        IssueLevel.ERROR: ("ERROR", "bold red"),
        IssueLevel.WARNING: ("WARN", "bold yellow"),
        IssueLevel.INFO: ("INFO", "cyan"),
    }

    if not report.issues:
        table.add_row(
            _status_text("PASS", "bold green"),
            f"{report.checked} documents",
            "every recorded document is present, is a PDF, and matches its hash",
        )
    for issue in report.issues:
        label, style = styles[issue.level]
        table.add_row(_status_text(label, style), issue.document_id or "—", issue.detail)
    console.print(table)

    summary = (
        f"[bold]{report.checked}[/] checked · [red]{len(report.errors)} error(s)[/] · "
        f"[yellow]{len(report.warnings)} warning(s)[/]"
    )
    if verbose:
        summary += f" · [cyan]{len(report.notes)} note(s)[/]"
    console.print(summary)
    raise typer.Exit(code=report.exit_code())


@corpus_app.command("sync")
def sync(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the per-document confirmation prompt."),
    ] = False,
    backfill_page_counts: Annotated[
        bool,
        typer.Option(
            "--backfill-page-counts",
            help="Promote page counts the database has learned into manifest entries "
            "where the field is null.",
        ),
    ] = False,
    backfill_publication_dates: Annotated[
        bool,
        typer.Option(
            "--backfill-publication-dates",
            help="Promote publication dates the database has learned into manifest "
            "entries where the field is null.",
        ),
    ] = False,
) -> None:
    """Update the manifest: adopt unrecorded files, or backfill null operational fields.

    With no flags this adopts corpus files that exist on disk but have no manifest
    entry. With a `--backfill-*` flag it instead promotes values from the database into
    entries where that field is null.

    Never overwrites a value that is already recorded, in either mode. An entry that
    disagrees with its file is an integrity failure for `verify` to report, not
    something to quietly paper over.
    """
    bootstrap()
    from medguard.config import get_settings
    from medguard.corpus.manifest import BackfillField, Manifest, load_manifest
    from medguard.errors import CorpusError

    settings = get_settings()

    fields: set[BackfillField] = set()
    if backfill_page_counts:
        fields.add(BackfillField.PAGE_COUNT)
    if backfill_publication_dates:
        fields.add(BackfillField.PUBLICATION_DATE)

    manifest: Manifest
    if settings.corpus.manifest_path.is_file():
        try:
            manifest = load_manifest(settings.corpus.manifest_path)
        except CorpusError as exc:
            err_console.print(f"[red]{exc.detail}[/]")
            raise typer.Exit(code=1) from exc
    elif fields:
        err_console.print(
            f"[red]No manifest at {settings.corpus.manifest_path}. There is nothing to "
            "backfill into; run `medguard corpus download` first.[/]"
        )
        raise typer.Exit(code=1)
    else:
        from datetime import UTC, datetime

        manifest = Manifest(generated_at=datetime.now(UTC))

    if fields:
        _sync_backfill(settings, manifest, fields, yes=yes)
    else:
        _sync_adopt(settings, manifest, yes=yes)


def _sync_backfill(
    settings: Settings, manifest: Manifest, fields: set[BackfillField], *, yes: bool
) -> None:
    """Promote database-derived values into null manifest fields."""
    from medguard.corpus.manifest import apply_backfill, plan_backfill, write_manifest

    values = _read_operational_values(settings)
    if values is None:
        err_console.print(
            "[red]Could not read the database. Backfill takes its values from the "
            "`documents` table; run `medguard db upgrade` and ingest first.[/]"
        )
        raise typer.Exit(code=1)

    changes = plan_backfill(manifest, values, fields)
    if not changes:
        # Deliberately does not write. `generated_at` is what tells a reviewer whether
        # a run touched the corpus record, so a no-op must leave the file untouched.
        console.print("Nothing to backfill. The manifest already has every value.")
        raise typer.Exit(code=0)

    table = Table(title="Proposed backfill", header_style="bold", expand=True)
    table.add_column("Document", width=32, no_wrap=True)
    table.add_column("Field", width=18, no_wrap=True)
    table.add_column("Manifest", width=10, no_wrap=True)
    table.add_column("Database", overflow="fold")
    for change in changes:
        table.add_row(change.document_id, change.field.value, "null", str(change.value))
    console.print(table)

    accepted = changes
    if not yes:
        accepted = [c for c in changes if typer.confirm(f"  Apply {c.describe()}?", default=True)]
        if not accepted:
            console.print("\nNothing applied.")
            raise typer.Exit(code=0)

    write_manifest(apply_backfill(manifest, accepted), settings.corpus.manifest_path)
    console.print(
        f"\n[green]Backfilled {len(accepted)} field(s)[/] → {settings.corpus.manifest_path}"
    )


def _sync_adopt(settings: Settings, manifest: Manifest, *, yes: bool) -> None:
    """Record manifest entries for corpus files that exist on disk but are unrecorded."""
    from medguard.corpus.manifest import source_to_manifest_entry, write_manifest
    from medguard.corpus.sources import load_sources
    from medguard.errors import CorpusError

    try:
        sources = load_sources(settings.corpus.sources_path)
    except CorpusError as exc:
        err_console.print(f"[red]{exc.detail}[/]")
        raise typer.Exit(code=1) from exc

    recorded = {d.document_id for d in manifest.documents}
    additions = []
    for source in sources.documents:
        if source.document_id in recorded:
            continue
        path = settings.corpus.raw_dir / source.local_filename
        if not path.is_file():
            continue
        console.print(
            f"\n[bold]{source.document_id}[/] — {source.title}\n"
            f"  file: {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)\n"
            f"  from: {source.source_url}"
        )
        if not yes and not typer.confirm("  Record this document in the manifest?", default=True):
            console.print("  [dim]skipped[/]")
            continue
        additions.append(source_to_manifest_entry(source, path))

    orphans = sorted(
        p.name
        for p in settings.corpus.raw_dir.glob("*.pdf")
        if p.name not in {d.local_filename for d in sources.documents}
    )
    for name in orphans:
        err_console.print(
            f"[yellow]{name} is on disk but has no entry in sources.yaml — "
            "add one there first; sync will not invent provenance.[/]"
        )

    if not additions:
        console.print("\nNothing to add.")
        raise typer.Exit(code=0)

    from datetime import UTC, datetime

    updated = manifest.model_copy(
        update={
            "documents": [*manifest.documents, *additions],
            # Only on a real change; the early return above covers the no-op case.
            "generated_at": datetime.now(UTC),
        }
    )
    write_manifest(updated, settings.corpus.manifest_path)
    console.print(
        f"\n[green]Added {len(additions)} document(s)[/] → {settings.corpus.manifest_path}"
    )
