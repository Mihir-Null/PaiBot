import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.memory_index.search import SearchResult


def _make_result(text="chunk text", source="MEMORY.md", start=1, end=5, score=0.9):
    return SearchResult(text=text, source=source, start_line=start, end_line=end, score=score)


# --- is_available ---

def test_is_available_returns_false_when_binary_absent():
    from nanobot.memory_index.qmd import QMDSearcher

    with patch("shutil.which", return_value=None):
        searcher = QMDSearcher("qmd")
        assert searcher.is_available() is False


def test_is_available_returns_true_when_binary_present():
    from nanobot.memory_index.qmd import QMDSearcher

    with patch("shutil.which", return_value="/usr/local/bin/qmd"):
        searcher = QMDSearcher("qmd")
        assert searcher.is_available() is True


# --- rerank: fallback paths ---

async def test_rerank_returns_empty_list_for_empty_candidates():
    from nanobot.memory_index.qmd import QMDSearcher

    searcher = QMDSearcher("qmd")
    result = await searcher.rerank("query", [], top_k=3)
    assert result == []


async def test_rerank_returns_candidates_sliced_when_binary_absent():
    from nanobot.memory_index.qmd import QMDSearcher

    candidates = [_make_result(text=f"chunk {i}") for i in range(5)]
    with patch("shutil.which", return_value=None):
        searcher = QMDSearcher("qmd")
        result = await searcher.rerank("query", candidates, top_k=2)
    assert result == candidates[:2]


# --- rerank: happy path ---

async def test_rerank_happy_path_respects_qmd_order():
    from nanobot.memory_index.qmd import QMDSearcher

    c0 = _make_result(text="chunk 0", score=0.9)
    c1 = _make_result(text="chunk 1", score=0.8)
    c2 = _make_result(text="chunk 2", score=0.7)
    candidates = [c0, c1, c2]

    # QMD says: best is c2, then c0 (c1 omitted → appended at end)
    qmd_output = json.dumps({"ranked": [2, 0]}).encode()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(qmd_output, b""))

    with patch("shutil.which", return_value="/usr/local/bin/qmd"), \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        searcher = QMDSearcher("qmd")
        result = await searcher.rerank("query", candidates, top_k=3)

    assert result[0] is c2
    assert result[1] is c0
    assert result[2] is c1  # appended as unmentioned candidate


async def test_rerank_top_k_limits_output():
    from nanobot.memory_index.qmd import QMDSearcher

    candidates = [_make_result(text=f"chunk {i}") for i in range(5)]
    qmd_output = json.dumps({"ranked": [4, 3, 2, 1, 0]}).encode()

    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(qmd_output, b""))

    with patch("shutil.which", return_value="/usr/local/bin/qmd"), \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        searcher = QMDSearcher("qmd")
        result = await searcher.rerank("query", candidates, top_k=2)

    assert len(result) == 2
    assert result[0] is candidates[4]
    assert result[1] is candidates[3]


# --- rerank: error handling ---

async def test_rerank_falls_back_on_subprocess_exception():
    from nanobot.memory_index.qmd import QMDSearcher

    candidates = [_make_result(text=f"chunk {i}") for i in range(3)]

    with patch("shutil.which", return_value="/usr/local/bin/qmd"), \
         patch("asyncio.create_subprocess_exec", side_effect=OSError("no such file")):
        searcher = QMDSearcher("qmd")
        result = await searcher.rerank("query", candidates, top_k=2)

    assert result == candidates[:2]


async def test_rerank_falls_back_on_invalid_json_response():
    from nanobot.memory_index.qmd import QMDSearcher

    candidates = [_make_result(text=f"chunk {i}") for i in range(3)]
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"not-valid-json", b""))

    with patch("shutil.which", return_value="/usr/local/bin/qmd"), \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        searcher = QMDSearcher("qmd")
        result = await searcher.rerank("query", candidates, top_k=2)

    assert result == candidates[:2]


async def test_rerank_falls_back_on_timeout():
    from nanobot.memory_index.qmd import QMDSearcher

    candidates = [_make_result(text=f"chunk {i}") for i in range(3)]
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch("shutil.which", return_value="/usr/local/bin/qmd"), \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        searcher = QMDSearcher("qmd")
        result = await searcher.rerank("query", candidates, top_k=2)

    assert result == candidates[:2]
    mock_proc.kill.assert_called_once()
    mock_proc.wait.assert_called_once()
