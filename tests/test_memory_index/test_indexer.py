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
