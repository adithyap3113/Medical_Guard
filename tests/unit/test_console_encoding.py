"""The CLI must not die on its own output.

`medguard corpus download` crashed on a stock Windows console with

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

before a single byte had been downloaded. The offending character was the arrow in
``Corpus: 20 entries → data/raw``: Python's standard streams default to the system ANSI
code page on Windows, and cp1252 has no U+2192.

That is a bad class of bug. It is invisible to a developer whose terminal is UTF-8, it
is invisible to CI on Linux, it has nothing to do with the feature being exercised, and
the failure names ``print`` rather than the encoding. So the fix gets a test that runs
everywhere, which is why :func:`force_utf8_streams` keys off the stream encoding rather
than off ``sys.platform``.

The subprocess tests are the ones that matter. In-process assertions cannot reproduce
this: pytest replaces ``sys.stdout`` with a capture object that accepts any string, so
the encoding that broke the CLI is exactly the thing the test harness hides.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from medguard.cli.bootstrap import force_utf8_streams

#: The two characters the CLI actually prints that cp1252 cannot encode.
ARROW = "→"
MIDDOT = "·"


def _run_under(encoding: str, code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a child process whose stdio starts at `encoding`.

    The pipes are decoded as UTF-8 regardless of what the child *started* at, because
    the point of the fix is that the child ends up writing UTF-8. Letting
    ``subprocess`` fall back to the parent's locale encoding decodes a correct arrow
    into ``â†'`` and fails a test that is actually passing -- which happened, and cost
    a debugging round.
    """
    return subprocess.run(  # noqa: S603 - fixed interpreter, literal source, no shell
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env={**_clean_env(), "PYTHONIOENCODING": encoding},
    )


def _clean_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    return env


class TestTheBugIsReal:
    """Establish that the failure mode exists, so the fix is not testing nothing."""

    def test_an_unprotected_stream_still_dies_on_the_arrow(self) -> None:
        result = _run_under("cp1252", f"print('entries {ARROW} data/raw')")

        assert result.returncode != 0
        assert "UnicodeEncodeError" in result.stderr


class TestTheFixHolds:
    def test_the_shared_console_prints_the_arrow_under_cp1252(self) -> None:
        """Importing bootstrap is what repairs the stream, before Console is built."""
        result = _run_under(
            "cp1252",
            "from medguard.cli.bootstrap import console\n"
            f"console.print('entries {ARROW} data/raw {MIDDOT} done')\n",
        )

        assert result.returncode == 0, result.stderr
        assert ARROW in result.stdout
        assert MIDDOT in result.stdout

    def test_stderr_is_repaired_too(self) -> None:
        """Errors are printed to `err_console`; a crash there hides the real failure."""
        result = _run_under(
            "cp1252",
            "from medguard.cli.bootstrap import err_console\n"
            f"err_console.print('failed {ARROW} see log')\n",
        )

        assert result.returncode == 0, result.stderr
        assert ARROW in result.stderr

    def test_the_download_summary_line_survives(self) -> None:
        """The exact string that crashed, reconstructed from `cli/corpus.py`."""
        result = _run_under(
            "cp1252",
            "from medguard.cli.bootstrap import console\n"
            "console.print(f'Corpus: [bold]20[/] entries ([bold]19[/] active) "
            f"{ARROW} data/raw')\n"
            f"console.print('[green]19 available[/] {MIDDOT} [red]0 failed[/]')\n",
        )

        assert result.returncode == 0, result.stderr

    def test_a_utf8_stream_is_left_alone(self) -> None:
        result = _run_under(
            "utf-8",
            "import sys\n"
            "from medguard.cli.bootstrap import console\n"
            f"console.print('{ARROW}')\n"
            "sys.stderr.write(sys.stdout.encoding.lower().replace('-', '_'))\n",
        )

        assert result.returncode == 0, result.stderr
        assert result.stderr.strip().endswith("utf_8")


class TestItNeverRaises:
    """Every degraded case leaves the stream alone rather than failing the CLI."""

    def test_a_stream_without_reconfigure_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`io.StringIO` has an encoding of None and no reconfigure -- pytest's case."""
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())

        force_utf8_streams()  # must simply return

    def test_a_stream_whose_reconfigure_raises_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Detached(io.StringIO):
            encoding = "cp1252"

            def reconfigure(self, **_kwargs: object) -> None:
                msg = "underlying buffer has been detached"
                raise ValueError(msg)

        monkeypatch.setattr(sys, "stdout", Detached())
        monkeypatch.setattr(sys, "stderr", Detached())

        force_utf8_streams()  # must swallow it

    def test_it_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Called at import and callable again; the second call must be a no-op."""
        calls: list[dict[str, object]] = []

        class Recording(io.StringIO):
            encoding = "cp1252"

            def reconfigure(self, **kwargs: object) -> None:
                calls.append(kwargs)
                type(self).encoding = "utf-8"

        monkeypatch.setattr(sys, "stdout", Recording())
        monkeypatch.setattr(sys, "stderr", Recording())

        force_utf8_streams()
        first = len(calls)
        force_utf8_streams()

        assert first > 0
        assert len(calls) == first
        assert calls[0] == {"encoding": "utf-8", "errors": "replace"}
