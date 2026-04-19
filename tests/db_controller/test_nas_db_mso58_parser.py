from __future__ import annotations

from pathlib import Path

import ifi.db_controller.nas_db_mixin_parse_csv as parse_mod
from ifi.db_controller.nas_db_base import NasDBBase
from ifi.db_controller.nas_db_mixin_parse_csv import NasDBMixinParseCsv


class _DummyNasCsvParser(NasDBMixinParseCsv, NasDBBase):
    def __init__(self):
        self.access_mode = "local"
        self.sftp_client = None
        self._ensure_logger(component=__name__)


def test_parse_mso58_local_file_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "sample_ALL.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Model,MSO58",
                "Record Length,3",
                "Sample Interval,1e-6",
                "Some Header,Value",
                "TIME,CH1,CH2",
                "0.0,1.0,2.0",
                "1.0e-6,3.0,4.0",
                "2.0e-6,5.0,6.0",
            ]
        ),
        encoding="utf-8",
    )

    parser = _DummyNasCsvParser()
    header_content = csv_path.read_text(encoding="utf-8").splitlines()[:10]

    df = parser._parse_mso58(str(csv_path), header_content)

    assert df is not None
    assert list(df.columns) == ["TIME", "CH0", "CH1"]
    assert len(df) == 3
    assert df.attrs["metadata"]["record_length"] == 3


def test_parse_mso58_local_file_uses_temp_staging(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "sample_ALL.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Model,MSO58",
                "Record Length,1",
                "Sample Interval,1e-6",
                "TIME,CH1",
                "0.0,1.0",
            ]
        ),
        encoding="utf-8",
    )

    copied = {}
    original_copyfile = parse_mod.shutil.copyfile

    def recording_copyfile(src, dst, *args, **kwargs):
        copied["src"] = Path(src)
        copied["dst"] = Path(dst)
        return original_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(parse_mod.shutil, "copyfile", recording_copyfile)

    parser = _DummyNasCsvParser()
    header_content = csv_path.read_text(encoding="utf-8").splitlines()[:10]

    df = parser._parse_mso58(str(csv_path), header_content)

    assert df is not None
    assert copied["src"] == csv_path
    assert copied["dst"].name == csv_path.name
    assert copied["dst"] != csv_path


def test_parse_mso58_crcrlf_preserves_first_data_row(tmp_path: Path):
    csv_path = tmp_path / "sample_ALL_crcrlf.csv"
    csv_path.write_bytes(
        b"Model,MSO58\r\r\n"
        b"Record Length,3\r\r\n"
        b"Sample Interval,1e-6\r\r\n"
        b"\r\r\n"
        b"TIME,CH1,CH2\r\r\n"
        b"0.0,1.0,2.0\r\r\n"
        b"1.0e-6,3.0,4.0\r\r\n"
        b"2.0e-6,5.0,6.0\r\r\n"
    )

    parser = _DummyNasCsvParser()
    header_content = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]

    df = parser._parse_mso58(str(csv_path), header_content)

    assert df is not None
    assert len(df) == 3
    assert list(df.columns) == ["TIME", "CH0", "CH1"]
    assert df.iloc[0]["TIME"] == 0.0
