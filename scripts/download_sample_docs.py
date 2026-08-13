"""Download WHO sample documents for MedGuard ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DOWNLOADS: list[tuple[str, str]] = [
    (
        "who_hypertension_2021.pdf",
        "https://iris.who.int/bitstream/handle/10665/344424/9789240033986-eng.pdf?sequence=1&isAllowed=y",
    ),
    (
        "who_malaria_2023.pdf",
        "https://iris.who.int/bitstream/handle/10665/373339/WHO-UCN-GMP-2023.01-Rev.1-eng.pdf?sequence=1&isAllowed=y",
    ),
    (
        "who_essential_medicines_2023.pdf",
        "https://iris.who.int/bitstream/handle/10665/371090/WHO-MHP-HPS-EML-2023.02-eng.pdf?sequence=1&isAllowed=y",
    ),
]


def download_one(filename: str, url: str, out_dir: Path) -> None:
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  already present: {filename} ({dest.stat().st_size / 1024:.0f} KB)")
        return

    print(f"  downloading {filename} ...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,*/*;q=0.8",
        }
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0, headers=headers) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=64 * 1024):
                    f.write(chunk)
    except httpx.HTTPError as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return

    # Sanity: valid PDF starts with %PDF-
    with dest.open("rb") as f:
        magic = f.read(5)
    if magic != b"%PDF-":
        print(f"    FAILED: not a valid PDF (got {magic!r})", file=sys.stderr)
        dest.unlink()
        return

    print(f"    saved: {dest.stat().st_size / 1024:.0f} KB")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing to: {DATA_DIR}\n")
    for filename, url in DOWNLOADS:
        download_one(filename, url, DATA_DIR)
    present = sum(1 for f, _ in DOWNLOADS if (DATA_DIR / f).exists())
    print(f"\nStatus: {present}/{len(DOWNLOADS)} documents ready.")


if __name__ == "__main__":
    main()
