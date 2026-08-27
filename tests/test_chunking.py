from pathlib import Path

import pytest

from setisignals.ray_utils import chunk_file


def _write(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


@pytest.mark.parametrize("n_chunks", [1, 2, 5, 100])
def test_chunk_file_covers_and_reassembles(tmp_path: Path, n_chunks: int) -> None:
    lines = [f"line{i}|value{i}|\n".encode() for i in range(37)]
    content = b"".join(lines)
    path = _write(tmp_path, "data.spike", content)

    ranges = chunk_file(path, n_chunks)

    assert ranges, "expected at least one chunk for a non-empty file"
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(content)
    for (_, end), (next_start, _) in zip(ranges, ranges[1:]):
        assert end == next_start

    reassembled = b""
    total_lines = 0
    with open(path, "rb") as f:
        for start, end in ranges:
            f.seek(start)
            chunk = f.read(end - start)
            reassembled += chunk
            total_lines += len(chunk.splitlines())

    assert reassembled == content
    assert total_lines == len(lines)


def test_chunk_file_no_trailing_newline(tmp_path: Path) -> None:
    content = b"a|1|\nb|2|\nc|3|"  # no trailing newline
    path = _write(tmp_path, "data.spike", content)

    ranges = chunk_file(path, 4)

    reassembled = b""
    with open(path, "rb") as f:
        for start, end in ranges:
            f.seek(start)
            reassembled += f.read(end - start)
    assert reassembled == content


def test_chunk_file_empty(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.spike", b"")
    assert chunk_file(path, 8) == []


def test_chunk_file_single_line(tmp_path: Path) -> None:
    content = b"only|one|line|\n"
    path = _write(tmp_path, "single.spike", content)
    ranges = chunk_file(path, 8)
    assert ranges == [(0, len(content))]


def test_chunk_file_more_chunks_than_lines(tmp_path: Path) -> None:
    content = b"a|\nb|\n"
    path = _write(tmp_path, "two_lines.spike", content)
    ranges = chunk_file(path, 100)
    assert len(ranges) <= 2
    reassembled = b""
    with open(path, "rb") as f:
        for start, end in ranges:
            f.seek(start)
            reassembled += f.read(end - start)
    assert reassembled == content
