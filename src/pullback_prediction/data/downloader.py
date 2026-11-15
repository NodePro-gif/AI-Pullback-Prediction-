"""Download raw XAUUSD bid/ask candles using the pinned dukascopy-node CLI."""

from __future__ import annotations

import shutil
import subprocess
from datetime import date
from pathlib import Path

DUKASCOPY_NODE_VERSION = "1.51.1"


def download_dukascopy_bid_ask(
    start_date: str,
    end_date_exclusive: str,
    *,
    timeframe: str = "M1",
    output_dir: str | Path = "data/raw",
    batch_size: int = 10,
    batch_pause_ms: int = 1000,
    retries: int = 3,
) -> tuple[Path, Path]:
    """Download separate bid and ask CSV files without overwriting raw data.

    Requires Node.js/npm (npx). The end date is exclusive. Completed downloads
    are moved into output_dir only after both sides are present; cache files are
    retained to resume an interrupted download.
    """
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date_exclusive)
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format.") from exc
    if start >= end:
        raise ValueError("start_date must be earlier than end_date_exclusive.")
    if timeframe not in {"M1", "M5"}:
        raise ValueError("timeframe must be M1 or M5.")
    if batch_size < 1 or batch_pause_ms < 0 or retries < 0:
        raise ValueError("Batch size must be positive; pause and retries cannot be negative.")

    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("Node.js/npm is required; install Node.js so the npx command exists.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    work_dir = destination / ".dukascopy-download-tmp"
    cache_dir = destination / ".dukascopy-cache"
    work_dir.mkdir(parents=True, exist_ok=True)

    final_paths = {
        side: destination
        / f"xauusd-{timeframe.lower()}-{side}-{start_date}-{end_date_exclusive}.csv"
        for side in ("bid", "ask")
    }
    existing = [str(path) for path in final_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite raw market data: " + ", ".join(existing)
        )

    downloaded: dict[str, Path] = {}
    for side in ("bid", "ask"):
        filename = (
            f"xauusd-{timeframe.lower()}-{side}-{start_date}-{end_date_exclusive}"
        )
        command = [
            npx,
            "--yes",
            f"dukascopy-node@{DUKASCOPY_NODE_VERSION}",
            "-i",
            "xauusd",
            "-from",
            start_date,
            "-to",
            end_date_exclusive,
            "-t",
            timeframe.lower(),
            "--price-type",
            side,
            "-f",
            "csv",
            "--directory",
            str(work_dir),
            "--file-name",
            filename,
            "--batch-size",
            str(batch_size),
            "--batch-pause",
            str(batch_pause_ms),
            "--retries",
            str(retries),
            "--rate-limit-retries",
            "3",
            "--cache",
            "--cache-path",
            str(cache_dir),
        ]
        subprocess.run(command, check=True)
        source_path = work_dir / f"{filename}.csv"
        if not source_path.is_file() or source_path.stat().st_size == 0:
            raise RuntimeError(f"Dukascopy downloader did not produce {source_path}.")
        downloaded[side] = source_path

    for side, source_path in downloaded.items():
        shutil.move(str(source_path), str(final_paths[side]))
    return final_paths["bid"], final_paths["ask"]
