"""Rendering for ``medguard doctor``.

Kept separate from :mod:`medguard.diagnostics`: that module decides *what is true*,
this one decides *how it looks*. The split means ``/ready`` can reuse the checks
without dragging Rich into the API, and the checks stay testable without parsing
terminal output.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from medguard.config import Settings
from medguard.diagnostics import Check, CheckStatus, DiagnosticReport

__all__ = ["render_report", "render_settings"]

_STATUS_STYLE: dict[CheckStatus, tuple[str, str]] = {
    CheckStatus.OK: ("PASS", "bold green"),
    CheckStatus.WARN: ("WARN", "bold yellow"),
    CheckStatus.FAIL: ("FAIL", "bold red"),
    CheckStatus.SKIP: ("SKIP", "dim"),
}


def _status_cell(check: Check) -> Text:
    label, style = _STATUS_STYLE[check.status]
    return Text(label, style=style)


def render_report(console: Console, report: DiagnosticReport) -> None:
    """Print the diagnostic table plus an actionable summary."""
    table = Table(
        title="MedGuard environment diagnostics",
        title_style="bold",
        header_style="bold",
        show_lines=False,
        expand=True,
    )
    table.add_column("Status", width=6, no_wrap=True)
    table.add_column("Check", width=28, no_wrap=True)
    table.add_column("Detail", overflow="fold")
    table.add_column("ms", justify="right", width=6, no_wrap=True)

    for check in report.checks:
        table.add_row(
            _status_cell(check),
            check.name,
            check.detail,
            f"{check.latency_ms:.0f}" if check.latency_ms is not None else "",
        )

    console.print(table)

    counts = {status: 0 for status in CheckStatus}
    for check in report.checks:
        counts[check.status] += 1
    console.print(
        f"[green]{counts[CheckStatus.OK]} passed[/] · "
        f"[yellow]{counts[CheckStatus.WARN]} warning(s)[/] · "
        f"[red]{counts[CheckStatus.FAIL]} failed[/] · "
        f"[dim]{counts[CheckStatus.SKIP]} skipped[/]"
    )

    if report.critical_failures:
        names = ", ".join(c.name for c in report.critical_failures)
        console.print(
            Panel(
                f"Critical checks failed: {names}\nThe application cannot serve requests.",
                border_style="red",
                title="Not ready",
            )
        )
    elif report.failed:
        names = ", ".join(c.name for c in report.failed)
        console.print(
            Panel(
                f"Non-critical checks failed: {names}\n"
                "The application will start, but the affected features will not work.",
                border_style="yellow",
                title="Degraded",
            )
        )
    else:
        console.print(
            Panel(
                "Core environment is healthy. Warnings above are expected while the "
                "corpus has not been ingested yet.",
                border_style="green",
                title="Ready",
            )
        )


def render_settings(console: Console, settings: Settings) -> None:
    """Print the effective configuration with every secret redacted.

    Uses :meth:`Settings.redacted`, which converts each ``SecretStr`` to a presence
    indicator. No key prefix, suffix, or length is shown, since those leak usable
    information about a credential.
    """
    table = Table(title="Effective configuration (secrets redacted)", header_style="bold")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")

    for group, values in settings.redacted().items():
        if not isinstance(values, dict):
            table.add_row(group, str(values))
            continue
        for key, value in values.items():
            table.add_row(f"{group}.{key}", "—" if value is None else str(value))

    table.add_row("derived.collection_name", settings.collection_name)
    console.print(table)
