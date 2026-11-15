from pathlib import Path

import pytest

from pullback_prediction.data.downloader import download_dukascopy_bid_ask


def test_downloader_fetches_bid_and_ask_without_overwriting_raw_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("pullback_prediction.data.downloader.shutil.which", lambda _: "/usr/bin/npx")

    def fake_run(command, check):
        calls.append(command)
        directory = Path(command[command.index("--directory") + 1])
        filename = command[command.index("--file-name") + 1]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{filename}.csv").write_text(
            "timestamp,open,high,low,close\n1704067200000,1,2,0,1\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("pullback_prediction.data.downloader.subprocess.run", fake_run)
    bid_path, ask_path = download_dukascopy_bid_ask(
        "2015-01-01",
        "2016-01-01",
        timeframe="M1",
        output_dir=tmp_path,
    )

    assert bid_path.exists()
    assert ask_path.exists()
    assert len(calls) == 2
    assert calls[0][calls[0].index("--price-type") + 1] == "bid"
    assert calls[1][calls[1].index("--price-type") + 1] == "ask"
    assert all("--cache" in command for command in calls)
    assert all("--retries" in command for command in calls)


def test_downloader_rejects_existing_raw_file(tmp_path):
    (tmp_path / "xauusd-m1-bid-2015-01-01-2016-01-01.csv").write_text(
        "existing", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        download_dukascopy_bid_ask(
            "2015-01-01", "2016-01-01", timeframe="M1", output_dir=tmp_path
        )


def test_downloader_rejects_unsupported_timeframe(tmp_path):
    with pytest.raises(ValueError, match="M1 or M5"):
        download_dukascopy_bid_ask(
            "2015-01-01", "2016-01-01", timeframe="H1", output_dir=tmp_path
        )
