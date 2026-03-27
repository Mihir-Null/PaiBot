# Semantic Memory PR 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `nanobot/memory_index/` package providing hybrid BM25+vector search over MEMORY.md and HISTORY.md, exposed to the agent as a `memory_search` tool.

**Architecture:** Fully standalone module with a single integration point (`MemorySearchTool`). `MemoryIndex` manages a SQLite database at `workspace/memory/.index.db`. The agent calls `memory_search` explicitly; the index is rebuilt asynchronously at gateway startup. Full MEMORY.md context injection is replaced with a short pointer note when enabled.

**Tech Stack:** Python 3.11+, SQLite FTS5 (BM25), sqlite-vec (cosine ANN), httpx (Ollama embeddings), pytest-asyncio, struct (float32 serialization)

---

## File Map

**Create:**
- `nanobot/memory_index/__init__.py` — exports `MemoryIndex`
- `nanobot/memory_index/index.py` — `MemoryIndex` class: DB init, model-mismatch rebuild, `startup_index()`, `search()`
- `nanobot/memory_index/indexer.py` — `Chunk` dataclass, `chunk_memory_file()`, `chunk_history_file()`, `write_chunks()`, `file_hash()`
- `nanobot/memory_index/search.py` — `SearchResult` dataclass, `bm25_search()`, `vector_search()`, `rrf_merge()`, `apply_temporal_decay()`, `mmr_rerank()`, `sync_search()`
- `nanobot/memory_index/embeddings.py` — `EmbeddingProvider` ABC, `OllamaEmbeddingProvider`, `EmbeddingUnavailableError`
- `nanobot/memory_index/schema.sql` — static table DDL (all except `chunks_vec`, which is created in code with configurable dimension)
- `nanobot/agent/tools/memory_search.py` — `MemorySearchTool`
- `tests/test_memory_index/__init__.py`
- `tests/test_memory_index/test_chunking.py`
- `tests/test_memory_index/test_search.py`
- `tests/test_memory_index/test_indexer.py`
- `tests/test_memory_index/test_embeddings.py`
- `tests/test_memory_index/test_tool.py`

**Modify:**
- `nanobot/config/schema.py` — add `MemoryIndexEmbeddingConfig`, `MemoryIndexQueryConfig`, `MemoryIndexConfig`; add `memory_index` field to `Config`
- `nanobot/agent/loop.py` — add `memory_index_config` param to `__init__`; register tool in `_register_default_tools()`
- `nanobot/agent/context.py` — add `memory_index_enabled` param to `ContextBuilder.__init__()`; conditional memory section in `build_system_prompt()`
- `pyproject.toml` — add `[memory]` optional dependency group

---

## Task 1: Config Schema

**Files:**
- Modify: `nanobot/config/schema.py`
- Test: `tests/config/test_schema.py` (existing file — add to it)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/config/test_schema.py

def test_memory_index_defaults():
    from nanobot.config.schema import MemoryIndexConfig
    cfg = MemoryIndexConfig()
    assert cfg.enabled is False
    assert cfg.embedding.provider == "ollama"
    assert cfg.embedding.model == "nomic-embed-text"
    assert cfg.embedding.base_url == "http://localhost:11434"
    assert cfg.embedding.batch_size == 16
    assert cfg.embedding.dim == 768
    assert cfg.query.top_k == 5
    assert cfg.query.mmr_lambda == 0.5
    assert cfg.query.temporal_decay_half_life_days == 90.0


def test_memory_index_on_root_config():
    from nanobot.config.schema import Config
    cfg = Config()
    assert cfg.memory_index.enabled is False


def test_memory_index_camel_case():
    from nanobot.config.schema import MemoryIndexConfig
    cfg = MemoryIndexConfig.model_validate({
        "enabled": True,
        "embedding": {"model": "mxbai-embed-large", "dim": 1024},
    })
    assert cfg.enabled is True
    assert cfg.embedding.model == "mxbai-embed-large"
    assert cfg.embedding.dim == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/config/test_schema.py -k "memory_index" -v
```

Expected: `ImportError` or `AttributeError` — `MemoryIndexConfig` doesn't exist yet.

- [ ] **Step 3: Add config models to `nanobot/config/schema.py`**

Add after the `ExecToolConfig` class (before `MCPServerConfig`):

```python
class MemoryIndexEmbeddingConfig(Base):
    """Embedding provider config for the memory index."""

    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    batch_size: int = 16
    dim: int = 768  # must match model output dimension


class MemoryIndexQueryConfig(Base):
    """Search and ranking config for the memory index."""

    top_k: int = 5
    mmr_lambda: float = 0.5
    temporal_decay_half_life_days: float = 90.0


class MemoryIndexConfig(Base):
    """Semantic memory index configuration. Off by default."""

    enabled: bool = False
    embedding: MemoryIndexEmbeddingConfig = Field(default_factory=MemoryIndexEmbeddingConfig)
    query: MemoryIndexQueryConfig = Field(default_factory=MemoryIndexQueryConfig)
```

Add to `Config` root model (after the `tools` field):

```python
    memory_index: MemoryIndexConfig = Field(default_factory=MemoryIndexConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/config/test_schema.py -k "memory_index" -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add nanobot/config/schema.py tests/config/test_schema.py
git commit -m "feat(memory_index): add MemoryIndexConfig to schema"
```

---

## Task 2: SQLite Schema + MemoryIndex Skeleton

**Files:**
- Create: `nanobot/memory_index/schema.sql`
- Create: `nanobot/memory_index/__init__.py`
- Create: `nanobot/memory_index/index.py`
- Create: `tests/test_memory_index/__init__.py`
- Create: `tests/test_memory_index/test_indexer.py` (partial — DB init tests only)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_index/__init__.py
# (empty)
```

```python
# tests/test_memory_index/test_indexer.py

import sqlite3
from pathlib import Path
import pytest
from nanobot.config.schema import MemoryIndexConfig
from nanobot.memory_index.index import MemoryIndex


def test_index_creates_tables(tmp_path):
    idx = MemoryIndex(tmp_path, MemoryIndexConfig())
    conn = sqlite3.connect(idx.db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','shadow') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()}
    conn.close()
    assert "files" in tables
    assert "chunks" in tables
    assert "index_meta" in tables
    # chunks_fts is a virtual table
    vtables = {r[0] for r in sqlite3.connect(idx.db_path).execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "chunks_fts" in vtables


def test_index_stores_model_metadata(tmp_path):
    cfg = MemoryIndexConfig()
    idx = MemoryIndex(tmp_path, cfg)
    conn = sqlite3.connect(idx.db_path)
    row = conn.execute("SELECT value FROM index_meta WHERE key='embedding_model'").fetchone()
    conn.close()
    assert row is not None
    assert row[0] == cfg.embedding.model


def test_index_rebuilds_on_model_mismatch(tmp_path):
    # Create index with model A
    cfg_a = MemoryIndexConfig()
    idx_a = MemoryIndex(tmp_path, cfg_a)
    # Write a dummy file record to verify it gets wiped
    conn = sqlite3.connect(idx_a.db_path)
    conn.execute("INSERT INTO files(path, mtime, size, hash) VALUES ('x', 0, 0, 'abc')")
    conn.commit()
    conn.close()

    # Create index with different model — should wipe and rebuild
    cfg_b = MemoryIndexConfig()
    cfg_b.embedding.model = "mxbai-embed-large"
    cfg_b.embedding.dim = 1024
    idx_b = MemoryIndex(tmp_path, cfg_b)
    conn = sqlite3.connect(idx_b.db_path)
    file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    stored_model = conn.execute("SELECT value FROM index_meta WHERE key='embedding_model'").fetchone()[0]
    conn.close()
    assert file_count == 0
    assert stored_model == "mxbai-embed-large"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_index/test_indexer.py -k "creates_tables or stores_model or rebuilds" -v
```

Expected: `ModuleNotFoundError` — `nanobot.memory_index.index` doesn't exist.

- [ ] **Step 3: Create `nanobot/memory_index/schema.sql`**

```sql
-- Static table DDL for nanobot memory index.
-- chunks_vec is created separately in code (dimension is configurable).

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id    INTEGER PRIMARY KEY,
    path  TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size  INTEGER NOT NULL,
    hash  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER REFERENCES files(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    created_at REAL NOT NULL,
    embedding  BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    tokenize='porter unicode61'
);
```

- [ ] **Step 4: Create `nanobot/memory_index/index.py`**

```python
"""MemoryIndex — lifecycle and public search interface."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import MemoryIndexConfig
    from nanobot.memory_index.search import SearchResult


class MemoryIndex:
    """Manages the SQLite memory index database."""

    def __init__(self, workspace: Path, config: MemoryIndexConfig) -> None:
        self.workspace = workspace
        self.config = config
        self.memory_dir = workspace / "memory"
        self.db_path = self.memory_dir / ".index.db"
        self._vec_available = False
        self._provider = None
        self._memory_dir_setup()
        self._check_vec_available()
        self._setup_db()
        self._setup_provider()

    def _memory_dir_setup(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _check_vec_available(self) -> None:
        """Check once at init whether sqlite-vec can be loaded."""
        try:
            import sqlite_vec  # noqa: F401
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            import sqlite_vec as sv
            sv.load(conn)
            conn.close()
            self._vec_available = True
        except Exception as e:
            logger.warning("sqlite-vec unavailable, BM25-only search active: {}", e)

    def _open_conn(self) -> sqlite3.Connection:
        """Open a connection with sqlite-vec loaded if available."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if self._vec_available:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        return conn

    def _setup_db(self) -> None:
        schema_sql = (Path(__file__).parent / "schema.sql").read_text()
        dim = self.config.embedding.dim

        if self.db_path.exists() and self._has_model_mismatch():
            logger.info("Embedding model changed, rebuilding memory index")
            self.db_path.unlink()

        conn = self._open_conn()
        try:
            conn.executescript(schema_sql)
            if self._vec_available:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec "
                    f"USING vec0(embedding FLOAT[{dim}])"
                )
            row = conn.execute(
                "SELECT 1 FROM index_meta WHERE key='embedding_model'"
            ).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO index_meta(key, value) VALUES (?, ?)",
                    ["embedding_model", self.config.embedding.model],
                )
                conn.execute(
                    "INSERT INTO index_meta(key, value) VALUES (?, ?)",
                    ["embedding_dim", str(dim)],
                )
            conn.commit()
        finally:
            conn.close()

    def _has_model_mismatch(self) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT value FROM index_meta WHERE key='embedding_model'"
                ).fetchone()
                if not row:
                    return False
                if row[0] != self.config.embedding.model:
                    return True
                row2 = conn.execute(
                    "SELECT value FROM index_meta WHERE key='embedding_dim'"
                ).fetchone()
                if row2 and int(row2[0]) != self.config.embedding.dim:
                    return True
                return False
            finally:
                conn.close()
        except Exception:
            return False

    def _setup_provider(self) -> None:
        if self.config.embedding.provider == "ollama":
            from nanobot.memory_index.embeddings import OllamaEmbeddingProvider
            self._provider = OllamaEmbeddingProvider(
                base_url=self.config.embedding.base_url,
                model=self.config.embedding.model,
                batch_size=self.config.embedding.batch_size,
            )

    async def startup_index(self) -> None:
        """Re-index memory files at startup (skips unchanged files)."""
        memory_file = self.memory_dir / "MEMORY.md"
        history_file = self.memory_dir / "HISTORY.md"
        for path in [memory_file, history_file]:
            if path.exists():
                try:
                    await self._index_file(path)
                except Exception:
                    logger.exception("Failed to index {}", path.name)

    async def _index_file(self, path: Path) -> None:
        """Index a single file if its content has changed."""
        from nanobot.memory_index.indexer import (
            chunk_history_file, chunk_memory_file, file_hash, write_chunks,
        )

        current_hash = await asyncio.to_thread(file_hash, path)
        stat = path.stat()

        conn = self._open_conn()
        try:
            row = conn.execute(
                "SELECT hash FROM files WHERE path = ?", [str(path)]
            ).fetchone()
        finally:
            conn.close()

        if row and row["hash"] == current_hash:
            return  # unchanged

        if path.name == "MEMORY.md":
            chunks = await asyncio.to_thread(chunk_memory_file, path)
        else:
            chunks = await asyncio.to_thread(chunk_history_file, path)

        embeddings = None
        if self._vec_available and self._provider and chunks:
            from nanobot.memory_index.embeddings import EmbeddingUnavailableError
            try:
                embeddings = await self._provider.embed_batch([c.text for c in chunks])
            except EmbeddingUnavailableError:
                logger.warning(
                    "Ollama unavailable while indexing {}, storing without embeddings",
                    path.name,
                )

        await asyncio.to_thread(
            write_chunks,
            self.db_path,
            path,
            stat.st_mtime,
            stat.st_size,
            current_hash,
            chunks,
            embeddings,
            self._vec_available,
        )
        logger.info("Indexed {} ({} chunks)", path.name, len(chunks))

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Hybrid BM25+vector search with temporal decay and MMR."""
        from nanobot.memory_index.embeddings import EmbeddingUnavailableError
        from nanobot.memory_index.search import sync_search

        k = top_k or self.config.query.top_k
        query_vec = None

        if self._vec_available and self._provider:
            try:
                vecs = await self._provider.embed_batch([query])
                query_vec = vecs[0]
            except EmbeddingUnavailableError:
                logger.debug("Ollama unavailable during search, using BM25-only")

        return await asyncio.to_thread(
            sync_search,
            self.db_path,
            query,
            query_vec,
            k,
            self._vec_available,
            self.config.query.mmr_lambda,
            self.config.query.temporal_decay_half_life_days,
        )
```

- [ ] **Step 5: Create `nanobot/memory_index/__init__.py`**

```python
"""Semantic memory index for nanobot."""

from nanobot.memory_index.index import MemoryIndex

__all__ = ["MemoryIndex"]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_memory_index/test_indexer.py -k "creates_tables or stores_model or rebuilds" -v
```

Expected: 3 PASSED.

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory_index/ tests/test_memory_index/
git commit -m "feat(memory_index): add SQLite schema and MemoryIndex skeleton"
```

---

## Task 3: File Chunking

**Files:**
- Create: `nanobot/memory_index/indexer.py` (chunking functions only — write pipeline added in Task 5)
- Create: `tests/test_memory_index/test_chunking.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_index/test_chunking.py

import time
from pathlib import Path
import pytest
from nanobot.memory_index.indexer import (
    MIN_TOKENS, TARGET_TOKENS, chunk_history_file, chunk_memory_file,
)


def test_chunk_memory_splits_on_headers(tmp_path):
    f = tmp_path / "MEMORY.md"
    f.write_text(
        "## Section A\n\nContent about section A.\n\n"
        "## Section B\n\nContent about section B.\n"
    )
    chunks = chunk_memory_file(f)
    assert len(chunks) == 2
    assert "Content about section A" in chunks[0].text
    assert "Content about section B" in chunks[1].text


def test_chunk_memory_skips_short_sections(tmp_path):
    f = tmp_path / "MEMORY.md"
    # 'hi' is way below MIN_TOKENS
    long_content = "word " * 60  # ~75 tokens
    f.write_text(f"## Tiny\n\nhi\n\n## Normal\n\n{long_content}\n")
    chunks = chunk_memory_file(f)
    # Tiny section (< MIN_TOKENS) should not produce a standalone chunk
    for c in chunks:
        assert len(c.text) // 4 >= MIN_TOKENS


def test_chunk_memory_created_at_from_mtime(tmp_path):
    f = tmp_path / "MEMORY.md"
    f.write_text("## Facts\n\nSome fact.\n")
    before = time.time()
    chunks = chunk_memory_file(f)
    after = time.time()
    assert chunks
    assert before - 1 <= chunks[0].created_at <= after + 1


def test_chunk_memory_large_section_splits_on_paragraphs(tmp_path):
    f = tmp_path / "MEMORY.md"
    # Each paragraph is ~50 tokens; with TARGET_TOKENS=400 we need 9+ paras to split
    para = "word " * 55 + "\n"  # ~55 tokens
    content = "\n\n".join([para] * 10)
    f.write_text(f"## Big Section\n\n{content}\n")
    chunks = chunk_memory_file(f)
    assert len(chunks) >= 2


def test_chunk_history_splits_by_timestamp(tmp_path):
    f = tmp_path / "HISTORY.md"
    f.write_text(
        "[2025-01-01 10:00] USER: hello world\n\n"
        "[2025-01-02 11:00] USER: another entry\n\n"
    )
    chunks = chunk_history_file(f)
    assert len(chunks) >= 1
    # created_at must be parsed from the timestamp, not file mtime
    assert chunks[0].created_at > 0
    # Jan 1 2025 is around unix 1735689600
    assert chunks[0].created_at > 1_700_000_000


def test_chunk_history_groups_small_entries(tmp_path):
    f = tmp_path / "HISTORY.md"
    entries = "".join(
        f"[2025-01-{i + 1:02d} 10:00] USER: short entry {i}\n\n" for i in range(5)
    )
    f.write_text(entries)
    chunks = chunk_history_file(f)
    # All 5 tiny entries should fit in a single window
    assert len(chunks) == 1


def test_chunk_history_splits_large_window(tmp_path):
    f = tmp_path / "HISTORY.md"
    # Each entry ~60 tokens; 8 entries = ~480 tokens > TARGET_TOKENS
    big_entry = "word " * 60
    entries = "".join(
        f"[2025-01-{i + 1:02d} 10:00] USER: {big_entry}\n\n" for i in range(8)
    )
    f.write_text(entries)
    chunks = chunk_history_file(f)
    assert len(chunks) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_index/test_chunking.py -v
```

Expected: `ImportError` — `nanobot.memory_index.indexer` doesn't exist.

- [ ] **Step 3: Create `nanobot/memory_index/indexer.py`**

```python
"""File chunking and SQLite write pipeline for the memory index."""

from __future__ import annotations

import hashlib
import re
import struct
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TOKEN_ESTIMATE_RATIO = 4   # characters per token (heuristic)
TARGET_TOKENS = 400
OVERLAP_TOKENS = 50
MIN_TOKENS = 50

_HISTORY_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")


@dataclass
class Chunk:
    text: str
    start_line: int
    end_line: int
    created_at: float  # unix timestamp


def file_hash(path: Path) -> str:
    """Return sha256 hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_memory_file(path: Path) -> list[Chunk]:
    """Split MEMORY.md into chunks by ## section headers."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    mtime = path.stat().st_mtime

    # Group lines into sections delimited by ## headers
    sections: list[tuple[int, str]] = []
    current_start = 0
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        if line.startswith("## ") and current_lines:
            sections.append((current_start, "".join(current_lines)))
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_start, "".join(current_lines)))

    chunks: list[Chunk] = []
    for start_line, section_text in sections:
        chunks.extend(_split_section(section_text, start_line, mtime))
    return chunks


def chunk_history_file(path: Path) -> list[Chunk]:
    """Split HISTORY.md into chunks by [YYYY-MM-DD HH:MM] entry boundaries."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Parse entries
    entries: list[tuple[int, float, list[str]]] = []
    current_start = 0
    current_ts: float | None = None
    current_lines: list[str] = []

    for i, line in enumerate(lines):
        m = _HISTORY_TS_RE.match(line)
        if m:
            if current_lines and current_ts is not None:
                entries.append((current_start, current_ts, current_lines[:]))
            current_start = i
            current_ts = _parse_ts(m.group(1))
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines and current_ts is not None:
        entries.append((current_start, current_ts, current_lines[:]))

    # Group entries into ~TARGET_TOKENS windows
    chunks: list[Chunk] = []
    buf_lines: list[str] = []
    buf_start = 0
    buf_ts: float | None = None
    buf_tokens = 0

    for start_line, ts, entry_lines in entries:
        entry_tokens = len("".join(entry_lines)) // TOKEN_ESTIMATE_RATIO
        if buf_lines and buf_tokens + entry_tokens > TARGET_TOKENS:
            chunk_text = "".join(buf_lines).strip()
            if chunk_text:
                chunks.append(Chunk(chunk_text, buf_start, start_line - 1, buf_ts or ts))
            buf_lines = entry_lines[:]
            buf_start = start_line
            buf_ts = ts
            buf_tokens = entry_tokens
        else:
            if not buf_lines:
                buf_start = start_line
                buf_ts = ts
            buf_lines.extend(entry_lines)
            buf_tokens += entry_tokens

    if buf_lines:
        chunk_text = "".join(buf_lines).strip()
        if chunk_text:
            end_line = buf_start + len(buf_lines) - 1
            chunks.append(Chunk(chunk_text, buf_start, end_line, buf_ts or 0.0))

    return chunks


def _split_section(text: str, base_line: int, created_at: float) -> list[Chunk]:
    """Split one section into <=TARGET_TOKENS chunks with OVERLAP_TOKENS overlap."""
    stripped = text.strip()
    if not stripped:
        return []

    token_count = len(stripped) // TOKEN_ESTIMATE_RATIO
    if token_count < MIN_TOKENS:
        return []  # too short to be useful

    if token_count <= TARGET_TOKENS:
        n_lines = stripped.count("\n")
        return [Chunk(stripped, base_line, base_line + n_lines, created_at)]

    paragraphs = re.split(r"\n\n+", stripped)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    line_offset = base_line

    for para in paragraphs:
        para_tokens = len(para) // TOKEN_ESTIMATE_RATIO
        if buf and buf_tokens + para_tokens > TARGET_TOKENS:
            chunk_text = "\n\n".join(buf).strip()
            n_lines = chunk_text.count("\n")
            if len(chunk_text) // TOKEN_ESTIMATE_RATIO >= MIN_TOKENS:
                chunks.append(Chunk(chunk_text, line_offset, line_offset + n_lines, created_at))
            # overlap: keep last paragraph for context continuity
            overlap = buf[-1]
            line_offset += n_lines + 2
            buf = [overlap, para]
            buf_tokens = len(overlap) // TOKEN_ESTIMATE_RATIO + para_tokens
        else:
            buf.append(para)
            buf_tokens += para_tokens

    if buf:
        chunk_text = "\n\n".join(buf).strip()
        if chunk_text:
            n_lines = chunk_text.count("\n")
            tail_tokens = len(chunk_text) // TOKEN_ESTIMATE_RATIO
            if chunks and tail_tokens < MIN_TOKENS:
                # Merge short tail into last chunk
                last = chunks[-1]
                merged = last.text + "\n\n" + chunk_text
                chunks[-1] = Chunk(merged, last.start_line, last.end_line + n_lines + 2, created_at)
            else:
                chunks.append(Chunk(chunk_text, line_offset, line_offset + n_lines, created_at))

    return chunks


def _parse_ts(ts_str: str) -> float:
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M").timestamp()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory_index/test_chunking.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add nanobot/memory_index/indexer.py tests/test_memory_index/test_chunking.py
git commit -m "feat(memory_index): add file chunking for MEMORY.md and HISTORY.md"
```

---

## Task 4: Embedding Provider

**Files:**
- Create: `nanobot/memory_index/embeddings.py`
- Create: `tests/test_memory_index/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_index/test_embeddings.py

import httpx
import pytest
from nanobot.memory_index.embeddings import (
    EmbeddingUnavailableError, OllamaEmbeddingProvider,
)


def _mock_response(embedding: list[float]) -> httpx.Response:
    r = httpx.Response(200, json={"embedding": embedding})
    r._request = httpx.Request("POST", "http://localhost:11434/api/embeddings")
    return r


async def test_embed_batch_returns_vectors(monkeypatch):
    async def mock_post(self, url, **kw):
        return _mock_response([0.1, 0.2, 0.3])

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text", batch_size=16
    )
    result = await provider.embed_batch(["hello world", "foo bar"])
    assert len(result) == 2
    assert result[0] == pytest.approx([0.1, 0.2, 0.3])
    assert result[1] == pytest.approx([0.1, 0.2, 0.3])


async def test_embed_batch_raises_on_connection_error(monkeypatch):
    async def mock_post(self, url, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text", batch_size=16
    )
    with pytest.raises(EmbeddingUnavailableError):
        await provider.embed_batch(["hello"])


async def test_embed_batch_caches_results(monkeypatch):
    call_count = 0

    async def mock_post(self, url, **kw):
        nonlocal call_count
        call_count += 1
        return _mock_response([0.5, 0.5])

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text", batch_size=16
    )
    await provider.embed_batch(["same text"])
    await provider.embed_batch(["same text"])
    assert call_count == 1  # second call hits cache


async def test_embed_batch_respects_batch_size(monkeypatch):
    calls = []

    async def mock_post(self, url, **kw):
        calls.append(kw.get("json", {}).get("prompt", ""))
        return _mock_response([0.1])

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text", batch_size=2
    )
    # 3 texts with batch_size=2 should make 3 individual calls (Ollama API is per-text)
    await provider.embed_batch(["a", "b", "c"])
    assert len(calls) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_index/test_embeddings.py -v
```

Expected: `ImportError` — `nanobot.memory_index.embeddings` doesn't exist.

- [ ] **Step 3: Create `nanobot/memory_index/embeddings.py`**

```python
"""Embedding providers for the memory index."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import httpx
from loguru import logger


class EmbeddingUnavailableError(Exception):
    """Raised when the embedding provider cannot be reached."""


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeds text via the Ollama /api/embeddings endpoint."""

    def __init__(self, base_url: str, model: str, batch_size: int = 16) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text. Uses in-memory cache to avoid re-embedding."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached.append((i, text))

        if uncached:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for i, text in uncached:
                        r = await client.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": self.model, "prompt": text},
                        )
                        r.raise_for_status()
                        vec = r.json()["embedding"]
                        self._cache[self._cache_key(text)] = vec
                        results[i] = vec
            except httpx.ConnectError as e:
                raise EmbeddingUnavailableError(f"Ollama unreachable: {e}") from e
            except httpx.TimeoutException as e:
                raise EmbeddingUnavailableError(f"Ollama timed out: {e}") from e
            except Exception as e:
                raise EmbeddingUnavailableError(f"Embedding failed: {e}") from e

        return [r for r in results if r is not None]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory_index/test_embeddings.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add nanobot/memory_index/embeddings.py tests/test_memory_index/test_embeddings.py
git commit -m "feat(memory_index): add OllamaEmbeddingProvider with caching"
```

---

## Task 5: Indexer Write Pipeline

**Files:**
- Modify: `nanobot/memory_index/indexer.py` (add `write_chunks()`)
- Modify: `tests/test_memory_index/test_indexer.py` (add write pipeline tests)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_memory_index/test_indexer.py`:

```python
import sqlite3
from pathlib import Path
import pytest
from nanobot.config.schema import MemoryIndexConfig
from nanobot.memory_index.index import MemoryIndex
from nanobot.memory_index.indexer import Chunk, write_chunks, file_hash


def _open(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_write_chunks_inserts_rows(tmp_path):
    idx = MemoryIndex(tmp_path, MemoryIndexConfig())
    f = tmp_path / "memory" / "MEMORY.md"
    f.write_text("## Facts\n\nSome fact.\n")
    chunks = [Chunk("Some fact.", 2, 2, 1_700_000_000.0)]
    write_chunks(idx.db_path, f, f.stat().st_mtime, f.stat().st_size, file_hash(f), chunks, None, False)
    conn = _open(idx.db_path)
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
    conn.close()


def test_write_chunks_replaces_on_reindex(tmp_path):
    idx = MemoryIndex(tmp_path, MemoryIndexConfig())
    f = tmp_path / "memory" / "MEMORY.md"
    f.write_text("## A\n\nOld content.\n")
    old_chunk = [Chunk("Old content.", 2, 2, 1_700_000_000.0)]
    write_chunks(idx.db_path, f, f.stat().st_mtime, f.stat().st_size, "hash1", old_chunk, None, False)

    new_chunk = [Chunk("New content.", 2, 2, 1_700_000_000.0)]
    write_chunks(idx.db_path, f, f.stat().st_mtime, f.stat().st_size, "hash2", new_chunk, None, False)

    conn = _open(idx.db_path)
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks").fetchall()]
    conn.close()
    assert texts == ["New content."]  # old chunk gone


def test_write_chunks_populates_fts(tmp_path):
    idx = MemoryIndex(tmp_path, MemoryIndexConfig())
    f = tmp_path / "memory" / "MEMORY.md"
    f.write_text("## Facts\n\nThe sky is blue.\n")
    chunks = [Chunk("The sky is blue.", 2, 2, 1_700_000_000.0)]
    write_chunks(idx.db_path, f, f.stat().st_mtime, f.stat().st_size, file_hash(f), chunks, None, False)
    conn = _open(idx.db_path)
    result = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'sky'"
    ).fetchall()
    conn.close()
    assert len(result) == 1


async def test_startup_index_skips_unchanged_file(tmp_path):
    cfg = MemoryIndexConfig()
    idx = MemoryIndex(tmp_path, cfg)
    f = tmp_path / "memory" / "MEMORY.md"
    f.write_text("## Facts\n\nA fact.\n")

    await idx.startup_index()
    conn = _open(idx.db_path)
    count_after_first = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    # Run again — should not duplicate
    await idx.startup_index()
    conn = _open(idx.db_path)
    count_after_second = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert count_after_first == count_after_second
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_index/test_indexer.py -k "write_chunks or startup_index" -v
```

Expected: `ImportError` — `write_chunks` not yet defined.

- [ ] **Step 3: Add `write_chunks()` to `nanobot/memory_index/indexer.py`**

Add after the `_parse_ts` function:

```python
def write_chunks(
    db_path: Path,
    file_path: Path,
    mtime: float,
    size: int,
    hash_: str,
    chunks: list[Chunk],
    embeddings: list[list[float]] | None,
    vec_available: bool,
) -> None:
    """Write (or replace) chunks for one file in the SQLite index."""
    conn = sqlite3.connect(db_path)
    if vec_available:
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            vec_available = False

    try:
        # Remove existing records for this file (CASCADE removes chunks)
        old_row = conn.execute(
            "SELECT id FROM files WHERE path = ?", [str(file_path)]
        ).fetchone()
        if old_row:
            old_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM chunks WHERE file_id = ?", [old_row[0]]
                ).fetchall()
            ]
            for cid in old_ids:
                conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", [cid])
                if vec_available:
                    try:
                        conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", [cid])
                    except Exception:
                        pass
            conn.execute("DELETE FROM files WHERE id = ?", [old_row[0]])

        # Insert new file record
        cur = conn.execute(
            "INSERT INTO files(path, mtime, size, hash) VALUES (?, ?, ?, ?)",
            [str(file_path), mtime, size, hash_],
        )
        file_id = cur.lastrowid

        for i, chunk in enumerate(chunks):
            emb_bytes = None
            if embeddings and i < len(embeddings):
                vec = embeddings[i]
                emb_bytes = struct.pack(f"{len(vec)}f", *vec)

            cur = conn.execute(
                "INSERT INTO chunks(file_id, text, start_line, end_line, created_at, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [file_id, chunk.text, chunk.start_line, chunk.end_line, chunk.created_at, emb_bytes],
            )
            chunk_id = cur.lastrowid

            conn.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", [chunk_id, chunk.text]
            )

            if vec_available and emb_bytes:
                try:
                    conn.execute(
                        "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                        [chunk_id, emb_bytes],
                    )
                except Exception:
                    pass

        conn.commit()
    finally:
        conn.close()
```

Also add `import struct` and `import sqlite3` at the top of `indexer.py`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory_index/test_indexer.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add nanobot/memory_index/indexer.py tests/test_memory_index/test_indexer.py
git commit -m "feat(memory_index): add write pipeline and startup_index"
```

---

## Task 6: Search Functions

**Files:**
- Create: `nanobot/memory_index/search.py`
- Create: `tests/test_memory_index/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_index/test_search.py

import math
import sqlite3
import struct
import time
from pathlib import Path
import pytest
from nanobot.memory_index.indexer import Chunk, write_chunks
from nanobot.memory_index.search import (
    SearchResult,
    apply_temporal_decay,
    bm25_search,
    mmr_rerank,
    rrf_merge,
    sync_search,
)
from nanobot.config.schema import MemoryIndexConfig
from nanobot.memory_index.index import MemoryIndex


def _make_db(tmp_path: Path, chunks_by_file: dict[str, list[Chunk]]) -> Path:
    """Create a test index DB with given chunks (no embeddings)."""
    cfg = MemoryIndexConfig()
    idx = MemoryIndex(tmp_path, cfg)
    for filename, chunks in chunks_by_file.items():
        f = tmp_path / "memory" / filename
        f.write_text("placeholder")
        write_chunks(idx.db_path, f, f.stat().st_mtime, f.stat().st_size, filename, chunks, None, False)
    return idx.db_path


def _open(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_bm25_search_returns_matching_chunk(tmp_path):
    db = _make_db(tmp_path, {"MEMORY.md": [
        Chunk("The capital of France is Paris.", 1, 1, time.time()),
        Chunk("Python is a programming language.", 2, 2, time.time()),
    ]})
    conn = _open(db)
    ids = bm25_search(conn, "France Paris", 5)
    conn.close()
    assert len(ids) >= 1  # at least the France/Paris chunk matches


def test_bm25_search_returns_empty_on_no_match(tmp_path):
    db = _make_db(tmp_path, {"MEMORY.md": [
        Chunk("The sky is blue.", 1, 1, time.time()),
    ]})
    conn = _open(db)
    ids = bm25_search(conn, "xyzzy_nonexistent_token_zzz", 5)
    conn.close()
    assert ids == []


def test_rrf_merge_boosts_shared_doc(tmp_path):
    # doc 1 appears in both lists; doc 2 only in bm25; doc 3 only in vec
    scores = rrf_merge([1, 2], [1, 3])
    assert scores[1] > scores[2]   # doc 1 boosted by both
    assert scores[1] > scores[3]


def test_rrf_merge_single_list(tmp_path):
    scores = rrf_merge([5, 6, 7], [])
    assert 5 in scores
    assert scores[5] > scores[6] > scores[7]


def test_temporal_decay_penalizes_old_chunk(tmp_path):
    now = time.time()
    old_ts = now - 180 * 86400  # 180 days ago
    db = _make_db(tmp_path, {"MEMORY.md": [
        Chunk("Recent fact.", 1, 1, now),
        Chunk("Old fact.", 2, 2, old_ts),
    ]})
    conn = _open(db)
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()]
    conn.close()
    recent_id, old_id = ids[0], ids[1]
    base_scores = {recent_id: 1.0, old_id: 1.0}
    conn = _open(db)
    decayed = apply_temporal_decay(base_scores, conn, half_life_days=90.0)
    conn.close()
    # 180-day-old chunk should score roughly exp(-2) ≈ 0.135 of original
    assert decayed[old_id] < decayed[recent_id]
    assert decayed[old_id] == pytest.approx(math.exp(-2.0), rel=0.05)


def test_mmr_rerank_increases_diversity(tmp_path):
    # Two near-identical vectors and one diverse vector
    # MMR should prefer the diverse one in the top-2
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.99, 0.14, 0.0]  # very similar to a
    vec_c = [0.0, 1.0, 0.0]   # diverse

    scores = {1: 0.9, 2: 0.8, 3: 0.5}  # doc 1 and 2 score higher, but similar
    chunk_vecs = {1: vec_a, 2: vec_b, 3: vec_c}

    result = mmr_rerank(scores, chunk_vecs, query_vec=[1.0, 0.0, 0.0], top_k=2, lambda_=0.5)
    assert 1 in result     # highest relevance always selected first
    assert 3 in result     # diverse doc preferred over near-duplicate doc 2
    assert 2 not in result


def test_mmr_rerank_falls_back_to_score_order_without_vecs(tmp_path):
    scores = {1: 0.9, 2: 0.7, 3: 0.5}
    result = mmr_rerank(scores, {}, query_vec=None, top_k=2, lambda_=0.5)
    assert result == [1, 2]


def test_sync_search_end_to_end(tmp_path):
    db = _make_db(tmp_path, {"MEMORY.md": [
        Chunk("The user prefers dark mode interfaces.", 1, 1, time.time()),
        Chunk("Python is a programming language.", 2, 2, time.time()),
        Chunk("The user lives in New York.", 3, 3, time.time()),
    ]})
    results = sync_search(db, "dark mode preference", None, 2, False, 0.5, 90.0)
    assert len(results) <= 2
    assert any("dark mode" in r.text for r in results)


def test_sync_search_returns_empty_for_no_match(tmp_path):
    db = _make_db(tmp_path, {"MEMORY.md": [
        Chunk("Unrelated content about cats.", 1, 1, time.time()),
    ]})
    results = sync_search(db, "xyzzy_nonexistent", None, 5, False, 0.5, 90.0)
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_index/test_search.py -v
```

Expected: `ImportError` — `nanobot.memory_index.search` doesn't exist.

- [ ] **Step 3: Create `nanobot/memory_index/search.py`**

```python
"""Hybrid BM25+vector search with RRF fusion, temporal decay, and MMR."""

from __future__ import annotations

import math
import sqlite3
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_RRF_K = 60


@dataclass
class SearchResult:
    text: str
    source: str
    start_line: int
    end_line: int
    score: float


def bm25_search(conn: sqlite3.Connection, query: str, k: int) -> list[int]:
    """Return up to k chunk IDs ranked by BM25 (best first)."""
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            [query, k],
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.OperationalError:
        return []


def vector_search(conn: sqlite3.Connection, query_vec: list[float], k: int) -> list[int]:
    """Return up to k chunk IDs ranked by cosine distance (best first)."""
    query_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunks_vec WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            [query_bytes, k],
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def rrf_merge(bm25_ids: list[int], vec_ids: list[int]) -> dict[int, float]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[int, float] = defaultdict(float)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] += 1.0 / (_RRF_K + rank)
    for rank, doc_id in enumerate(vec_ids):
        scores[doc_id] += 1.0 / (_RRF_K + rank)
    return dict(scores)


def apply_temporal_decay(
    scores: dict[int, float],
    conn: sqlite3.Connection,
    half_life_days: float,
) -> dict[int, float]:
    """Multiply each score by exp(-age_days / half_life_days)."""
    if not scores or half_life_days <= 0:
        return scores
    now = time.time()
    placeholders = ",".join("?" * len(scores))
    rows = conn.execute(
        f"SELECT id, created_at FROM chunks WHERE id IN ({placeholders})",
        list(scores.keys()),
    ).fetchall()
    id_to_ts = {r[0]: r[1] for r in rows}
    return {
        doc_id: score * math.exp(-(now - id_to_ts.get(doc_id, now)) / 86400 / half_life_days)
        for doc_id, score in scores.items()
    }


def mmr_rerank(
    scores: dict[int, float],
    chunk_vecs: dict[int, list[float]],
    query_vec: list[float] | None,
    top_k: int,
    lambda_: float,
) -> list[int]:
    """Maximal Marginal Relevance: diversify top-k results."""
    if not scores:
        return []
    if query_vec is None or not chunk_vecs:
        return sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

    candidates = list(scores.keys())
    selected: list[int] = []

    while len(selected) < top_k and candidates:
        best_id, best_score = None, float("-inf")
        for cid in candidates:
            relevance = scores[cid]
            max_sim = (
                max(_dot(chunk_vecs[cid], chunk_vecs[s]) for s in selected if s in chunk_vecs)
                if selected else 0.0
            )
            mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr_score > best_score:
                best_score, best_id = mmr_score, cid
        if best_id is None:
            break
        selected.append(best_id)
        candidates.remove(best_id)

    return selected


def sync_search(
    db_path: Path,
    query: str,
    query_vec: list[float] | None,
    top_k: int,
    vec_available: bool,
    mmr_lambda: float,
    half_life_days: float,
) -> list[SearchResult]:
    """Full search pipeline: BM25 + vector → RRF → decay → MMR → results."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if vec_available:
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            vec_available = False

    try:
        candidate_k = top_k * 4
        bm25_ids = bm25_search(conn, query, candidate_k)
        vec_ids = (
            vector_search(conn, query_vec, candidate_k)
            if (vec_available and query_vec)
            else []
        )

        if not bm25_ids and not vec_ids:
            return []

        merged = rrf_merge(bm25_ids, vec_ids)
        decayed = apply_temporal_decay(merged, conn, half_life_days)

        # Retrieve stored embeddings for MMR (from chunks.embedding BLOB)
        chunk_vecs: dict[int, list[float]] = {}
        if vec_available and query_vec:
            dim = len(query_vec)
            placeholders = ",".join("?" * len(decayed))
            rows = conn.execute(
                f"SELECT id, embedding FROM chunks WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
                list(decayed.keys()),
            ).fetchall()
            for r in rows:
                if r["embedding"]:
                    chunk_vecs[r["id"]] = list(struct.unpack(f"{dim}f", r["embedding"]))

        ordered = mmr_rerank(decayed, chunk_vecs, query_vec, top_k, mmr_lambda)

        if not ordered:
            return []

        placeholders = ",".join("?" * len(ordered))
        rows = conn.execute(
            f"SELECT c.id, c.text, c.start_line, c.end_line, f.path "
            f"FROM chunks c JOIN files f ON c.file_id = f.id "
            f"WHERE c.id IN ({placeholders})",
            ordered,
        ).fetchall()
        id_to_row = {r["id"]: r for r in rows}

        return [
            SearchResult(
                text=id_to_row[doc_id]["text"],
                source=Path(id_to_row[doc_id]["path"]).name,
                start_line=id_to_row[doc_id]["start_line"],
                end_line=id_to_row[doc_id]["end_line"],
                score=decayed.get(doc_id, 0.0),
            )
            for doc_id in ordered
            if doc_id in id_to_row
        ]
    finally:
        conn.close()


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
```

- [ ] **Step 4: Run all search tests to verify they pass**

```bash
pytest tests/test_memory_index/test_search.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add nanobot/memory_index/search.py tests/test_memory_index/test_search.py
git commit -m "feat(memory_index): add hybrid search (BM25+RRF+decay+MMR)"
```

---

## Task 7: MemorySearchTool

**Files:**
- Create: `nanobot/agent/tools/memory_search.py`
- Create: `tests/test_memory_index/test_tool.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_memory_index/test_tool.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from nanobot.agent.tools.memory_search import MemorySearchTool
from nanobot.memory_index.search import SearchResult


def _make_tool(results: list[SearchResult]) -> MemorySearchTool:
    index = MagicMock()
    index.search = AsyncMock(return_value=results)
    return MemorySearchTool(index)


async def test_tool_formats_results():
    results = [
        SearchResult("The user prefers dark mode.", "MEMORY.md", 5, 5, 0.9),
        SearchResult("The user lives in Berlin.", "MEMORY.md", 10, 10, 0.7),
    ]
    tool = _make_tool(results)
    output = await tool.execute(query="preferences")
    assert "MEMORY.md" in output
    assert "dark mode" in output
    assert "Berlin" in output
    assert "---" in output  # separator between results


async def test_tool_returns_no_results_message():
    tool = _make_tool([])
    output = await tool.execute(query="something obscure")
    assert output == "No relevant memory found."


async def test_tool_passes_top_k():
    index = MagicMock()
    index.search = AsyncMock(return_value=[])
    tool = MemorySearchTool(index)
    await tool.execute(query="test", top_k=3)
    index.search.assert_called_once_with("test", top_k=3)


async def test_tool_default_top_k():
    index = MagicMock()
    index.search = AsyncMock(return_value=[])
    tool = MemorySearchTool(index)
    await tool.execute(query="test")
    index.search.assert_called_once_with("test", top_k=5)


def test_tool_schema():
    index = MagicMock()
    tool = MemorySearchTool(index)
    assert tool.name == "memory_search"
    schema = tool.to_schema()
    assert schema["function"]["name"] == "memory_search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory_index/test_tool.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `nanobot/agent/tools/memory_search.py`**

```python
"""MemorySearchTool — semantic search over indexed memory files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.memory_index.index import MemoryIndex


class MemorySearchTool(Tool):
    """Search long-term memory and conversation history semantically."""

    name = "memory_search"
    description = (
        "Search long-term memory and conversation history semantically. "
        "Use when recalling facts, preferences, or past context that may "
        "not be in the current session."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, index: MemoryIndex) -> None:
        self.index = index

    async def execute(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        results = await self.index.search(query, top_k=top_k)
        if not results:
            return "No relevant memory found."
        return "\n\n---\n\n".join(
            f"[{r.source} L{r.start_line}–{r.end_line}]\n{r.text}"
            for r in results
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory_index/test_tool.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add nanobot/agent/tools/memory_search.py tests/test_memory_index/test_tool.py
git commit -m "feat(memory_index): add MemorySearchTool"
```

---

## Task 8: AgentLoop + ContextBuilder Integration

**Files:**
- Modify: `nanobot/agent/loop.py`
- Modify: `nanobot/agent/context.py`
- Test: `tests/agent/` (add integration smoke tests)

- [ ] **Step 1: Write failing tests**

Check what's in `tests/agent/`:

```bash
ls tests/agent/
```

Add a new file `tests/agent/test_memory_index_integration.py`:

```python
# tests/agent/test_memory_index_integration.py

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from nanobot.config.schema import MemoryIndexConfig


def test_context_builder_shows_pointer_note_when_enabled(tmp_path):
    from nanobot.agent.context import ContextBuilder
    ctx = ContextBuilder(tmp_path, memory_index_enabled=True)
    prompt = ctx.build_system_prompt()
    assert "memory_search" in prompt
    assert "tool" in prompt.lower()
    # Full MEMORY.md content should NOT be inlined
    assert "Long-term Memory" not in prompt


def test_context_builder_loads_memory_file_when_disabled(tmp_path):
    from nanobot.agent.context import ContextBuilder
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("## Facts\n\nUser loves coffee.\n")
    ctx = ContextBuilder(tmp_path, memory_index_enabled=False)
    prompt = ctx.build_system_prompt()
    assert "User loves coffee" in prompt


async def test_agent_loop_registers_tool_when_enabled(tmp_path):
    # Must be async: AgentLoop.__init__ calls asyncio.create_task for startup_index
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from unittest.mock import MagicMock

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    cfg = MemoryIndexConfig()
    cfg.enabled = True

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        memory_index_config=cfg,
    )
    assert loop.tools.get("memory_search") is not None


def test_agent_loop_does_not_register_tool_when_disabled(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        memory_index_config=MemoryIndexConfig(),  # enabled=False
    )
    assert loop.tools.get("memory_search") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agent/test_memory_index_integration.py -v
```

Expected: failures — `ContextBuilder` has no `memory_index_enabled` param, `AgentLoop` has no `memory_index_config` param.

- [ ] **Step 3: Modify `nanobot/agent/context.py`**

In `ContextBuilder.__init__()`, add `memory_index_enabled: bool = False`:

```python
def __init__(self, workspace: Path, timezone: str | None = None, memory_index_enabled: bool = False):
    self.workspace = workspace
    self.timezone = timezone
    self.memory_index_enabled = memory_index_enabled
    self.memory = MemoryStore(workspace)
    self.skills = SkillsLoader(workspace)
```

In `build_system_prompt()`, replace the memory block:

```python
        if self.memory_index_enabled:
            parts.append(
                "# Memory\n\n"
                "Use the `memory_search` tool to recall facts, preferences, or past context "
                "before answering questions that depend on prior sessions."
            )
        else:
            memory = self.memory.get_memory_context()
            if memory:
                parts.append(f"# Memory\n\n{memory}")
```

- [ ] **Step 4: Modify `nanobot/agent/loop.py`**

In `AgentLoop.__init__()`, add the parameter after `timezone`:

```python
        memory_index_config: MemoryIndexConfig | None = None,
```

Add the TYPE_CHECKING import at the top:

```python
if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, MemoryIndexConfig, WebSearchConfig
```

Store it on self and pass flag to ContextBuilder:

```python
        self.memory_index_config = memory_index_config
        self.context = ContextBuilder(
            workspace,
            timezone=timezone,
            memory_index_enabled=bool(memory_index_config and memory_index_config.enabled),
        )
```

At the end of `_register_default_tools()`:

```python
        if self.memory_index_config and self.memory_index_config.enabled:
            from nanobot.memory_index import MemoryIndex
            from nanobot.agent.tools.memory_search import MemorySearchTool
            _index = MemoryIndex(self.workspace, self.memory_index_config)
            self._schedule_background(_index.startup_index())
            self.tools.register(MemorySearchTool(_index))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/agent/test_memory_index_integration.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
pytest --tb=short -q
```

Expected: all pre-existing tests still pass. The only new failures should be pre-existing skips.

- [ ] **Step 7: Commit**

```bash
git add nanobot/agent/context.py nanobot/agent/loop.py tests/agent/test_memory_index_integration.py
git commit -m "feat(memory_index): wire MemorySearchTool into AgentLoop and ContextBuilder"
```

---

## Task 9: pyproject.toml Optional Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `memory` optional dependency group**

In `pyproject.toml`, after the existing `[project.optional-dependencies]` entries, add:

```toml
memory = [
    "sqlite-vec>=0.1.0",
]
```

- [ ] **Step 2: Verify install works**

```bash
pip install -e ".[memory]" --dry-run
```

Expected: resolves without errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(memory_index): add [memory] optional dependency for sqlite-vec"
```

---

## Task 10: Final Verification

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v --tb=short
```

Expected: all memory_index tests pass; no regressions in existing tests.

- [ ] **Step 2: Lint**

```bash
ruff check nanobot/memory_index/ nanobot/agent/tools/memory_search.py nanobot/agent/context.py nanobot/agent/loop.py nanobot/config/schema.py
```

Expected: no errors. Fix any reported issues before committing.

- [ ] **Step 3: Format**

```bash
ruff format nanobot/memory_index/ nanobot/agent/tools/memory_search.py
```

- [ ] **Step 4: Commit any lint/format fixes**

```bash
git add -u
git commit -m "style: apply ruff format to memory_index"
```

- [ ] **Step 5: Verify the feature end-to-end with a manual smoke test**

```python
# Run interactively: python -c "..."
import asyncio
from pathlib import Path
from nanobot.config.schema import MemoryIndexConfig
from nanobot.memory_index import MemoryIndex

async def smoke():
    workspace = Path("~/.nanobot/workspace").expanduser()
    cfg = MemoryIndexConfig()
    cfg.enabled = True
    idx = MemoryIndex(workspace, cfg)
    await idx.startup_index()
    results = await idx.search("user preferences", top_k=3)
    for r in results:
        print(f"[{r.source} L{r.start_line}] score={r.score:.3f}")
        print(r.text[:200])
        print("---")

asyncio.run(smoke())
```

Expected: prints relevant chunks from MEMORY.md (BM25-only since Ollama not configured). No exceptions.
