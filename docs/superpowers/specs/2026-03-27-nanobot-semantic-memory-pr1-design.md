# Nanobot Semantic Memory — PR 1 Design Spec

**Date:** 2026-03-27
**Target branch:** `nightly`
**Scope:** Standalone `memory_index/` module + `MemorySearchTool` — the agent can explicitly search its memory semantically. Index rebuilt at gateway startup.

---

## Overview

This is PR 1 of a 3-PR series:

| PR | Scope |
|----|-------|
| **PR 1 (this spec)** | `memory_index/` module, `MemorySearchTool`, startup re-index |
| PR 2 | Pre-retrieval injection (context hook) + file watcher for live updates |
| PR 3 | QMD sidecar (query expansion + LLM re-ranking) |

Each PR is useful independently; together they form a full semantic memory system.

### Problem

Nanobot's current memory system loads `MEMORY.md` wholesale into every system prompt. As the file grows, this bloats context and crowds out useful working space. Keyword grep misses semantically related content when phrasing differs. A preference written as "I like concise explanations" won't be found by a grep for "brevity."

### Solution (PR 1)

A self-contained `nanobot/memory_index/` package that:
1. Indexes `MEMORY.md` and `HISTORY.md` into a local SQLite database using BM25 full-text search (FTS5) and cosine vector search (sqlite-vec)
2. Exposes a `memory_search` tool the agent can call explicitly to retrieve the top-k most relevant chunks for a query
3. Replaces the full MEMORY.md context injection with a brief pointer note when enabled

Off by default. Zero behavior change for users who don't opt in.

---

## Architecture

### Approach: Fully Standalone Module

`nanobot/memory_index/` has zero unconditional imports into core files. The `MemorySearchTool` is the single integration point. Changes to existing files are minimal and clearly gated.

### Module Structure

```
nanobot/memory_index/
├── __init__.py          # Public API: MemoryIndex class
├── index.py             # MemoryIndex — lifecycle, startup re-index
├── indexer.py           # File reading, chunking, embedding, SQLite write pipeline
├── search.py            # BM25 query, vector query, RRF merge, temporal decay, MMR
├── embeddings.py        # EmbeddingProvider ABC + OllamaEmbeddingProvider
└── schema.sql           # CREATE TABLE / CREATE VIRTUAL TABLE statements

nanobot/agent/tools/
└── memory_search.py     # MemorySearchTool (Tool subclass)
```

### Startup Flow

1. `AgentLoop.__init__()` checks `config.memory_index.enabled`
2. If true, constructs `MemoryIndex(workspace, config.memory_index)` and `MemorySearchTool(index)`
3. Registers the tool via `self.tools.register()`
4. `MemoryIndex` runs a startup re-index as a background task via `asyncio.to_thread` — scans MEMORY.md + HISTORY.md, hashes each file, skips anything unchanged since last index

The index lives at `workspace/memory/.index.db` — alongside the files it indexes, obviously a cache artifact.

---

## SQLite Schema

Four tables. Embedding dimension is stored in `index_meta` so the schema can adapt when the model changes without hardcoding a dimension.

```sql
-- Tracks config at index creation time; mismatches trigger full rebuild
CREATE TABLE index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Rows: ("embedding_model", "nomic-embed-text"), ("embedding_dim", "768")

-- Tracks indexed files; skip re-index if mtime + size + hash unchanged
CREATE TABLE files (
    id    INTEGER PRIMARY KEY,
    path  TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size  INTEGER NOT NULL,
    hash  TEXT NOT NULL       -- sha256 of file content
);

-- Source of truth for all chunks
CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER REFERENCES files(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    start_line INTEGER,
    end_line   INTEGER,
    created_at REAL NOT NULL  -- unix timestamp; used for temporal decay
);

-- BM25 full-text search via FTS5
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);

-- Vector search via sqlite-vec (dimension interpolated from index_meta at CREATE time)
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding FLOAT[{dim}]
);
```

When `embedding_model` or `embedding_dim` in `index_meta` mismatches config, `index.py` drops and recreates the entire database, then re-indexes from scratch.

---

## Chunking Strategy

### MEMORY.md

Split on `##` section headers (the standard MEMORY.md template already uses them — chunk boundaries align with semantic units). Each `##` block is one chunk. If a block exceeds ~400 tokens (`len(text) // 4`), split further on double-newline with 50-token overlap. Minimum chunk size: 50 tokens — merge short trailing blocks into the previous chunk.

`created_at` defaults to file mtime (no embedded timestamp in MEMORY.md).

### HISTORY.md

Split on `[YYYY-MM-DD HH:MM]` entry boundaries. Group consecutive entries into ~400-token windows. Each window is one chunk. Timestamps extracted from entries feed directly into `created_at` — more accurate than file mtime and enables precise temporal decay.

---

## Embeddings

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class OllamaEmbeddingProvider(EmbeddingProvider):
    # POST {base_url}/api/embeddings
    # Batches calls up to config.batch_size (default 16)
    # In-memory cache keyed by sha256(text + model) — skips re-embedding unchanged chunks
    # On connection error: raises EmbeddingUnavailableError
```

Uses `httpx.AsyncClient` (already a hard dependency via `web.py`).

The in-memory cache is warm for the common case: startup re-index where most chunks are unchanged. Hits skip the HTTP round-trip entirely.

---

## Hybrid Search Pipeline

### Score Normalization: Reciprocal Rank Fusion

The proposal's raw weighted addition (`vec * 0.7 + bm25 * 0.3`) is incorrect — FTS5 BM25 scores are unnormalized floats with no fixed range, while cosine scores are [-1, 1]. Raw addition lets BM25 dominate unpredictably.

**RRF only needs rank position, not raw scores**, making it immune to scale mismatches:

```
rrf_score(rank) = 1 / (60 + rank)    # 60 is the standard RRF constant
```

### Full Pipeline

```
search(query, top_k):
    query_vec  = embed(query)                           # None if Ollama unavailable
    bm25_hits  = fts5_bm25_search(query, k=top_k*4)    # ranked list of chunk ids
    vec_hits   = cosine_search(query_vec, k=top_k*4)   # ranked list; skipped if None

    merged = defaultdict(float)
    for rank, id in enumerate(bm25_hits): merged[id] += rrf(rank)
    for rank, id in enumerate(vec_hits):  merged[id] += rrf(rank)

    decayed   = apply_temporal_decay(merged)    # score *= exp(-age_days / half_life)
    reranked  = mmr(decayed, query_vec, top_k)  # diversify results
    return fetch_chunk_texts(reranked)
```

### Temporal Decay

```
decayed_score = score * exp(-age_days / half_life_days)
```

`half_life_days` defaults to 90 (configurable). `age_days` derived from `chunks.created_at`. A chunk from 180 days ago at the same RRF score as a chunk from today will score ~50% lower.

### MMR (Maximal Marginal Relevance)

Iteratively selects the next chunk maximizing:

```
λ * relevance_to_query - (1 - λ) * max_similarity_to_already_selected
```

Implemented in ~20 lines of pure Python using dot products. Chunk embeddings are retrieved from `chunks_vec` alongside the chunk text as part of the search result fetch — no extra DB round-trip. No numpy required. `λ` defaults to 0.5 (configurable).

### BM25-Only Fallback

If `query_vec` is None (Ollama unreachable), the vector branch is skipped. RRF runs on BM25 results only. This is silent — no error returned to the agent.

---

## MemorySearchTool

New file: `nanobot/agent/tools/memory_search.py`

```python
class MemorySearchTool(Tool):
    name = "memory_search"
    description = (
        "Search long-term memory and conversation history semantically. "
        "Use when recalling facts, preferences, or past context that may "
        "not be in the current session."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, top_k: int = 5, **kwargs) -> str:
        results = await self.index.search(query, top_k=top_k)
        if not results:
            return "No relevant memory found."
        return "\n\n---\n\n".join(
            f"[{r.source} L{r.start_line}–{r.end_line}]\n{r.text}"
            for r in results
        )
```

---

## Config Schema

New models added to `nanobot/config/schema.py`, one new field on `Config`:

```python
class MemoryIndexEmbeddingConfig(Base):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    batch_size: int = 16
    dim: int = 768                          # must match model output dimension

class MemoryIndexQueryConfig(Base):
    top_k: int = 5
    mmr_lambda: float = 0.5
    temporal_decay_half_life_days: float = 90.0

class MemoryIndexConfig(Base):
    enabled: bool = False                   # opt-in, off by default
    embedding: MemoryIndexEmbeddingConfig = Field(default_factory=MemoryIndexEmbeddingConfig)
    query: MemoryIndexQueryConfig = Field(default_factory=MemoryIndexQueryConfig)

# Added to Config root:
memory_index: MemoryIndexConfig = Field(default_factory=MemoryIndexConfig)
```

Env var override pattern follows existing convention:
`NANOBOT_MEMORY_INDEX__ENABLED=true`
`NANOBOT_MEMORY_INDEX__EMBEDDING__MODEL=mxbai-embed-large`
`NANOBOT_MEMORY_INDEX__EMBEDDING__DIM=1024`

---

## Changes to Existing Files

### `nanobot/agent/loop.py` — `_register_default_tools()`

~6 lines added at the end of the method, fully gated:

```python
if self.memory_index_config and self.memory_index_config.enabled:
    from nanobot.memory_index import MemoryIndex
    from nanobot.agent.tools.memory_search import MemorySearchTool
    index = MemoryIndex(self.workspace, self.memory_index_config)
    self._schedule_background(index.startup_index())
    self.tools.register(MemorySearchTool(index))
```

`memory_index_config` is threaded through `AgentLoop.__init__()` as an optional parameter (same pattern as `web_search_config`, `exec_config`, etc.).

### `nanobot/agent/context.py` — `build_system_prompt()`

The `memory_index_enabled` flag is passed to `ContextBuilder` at construction time by `AgentLoop`. The memory section becomes:

```python
if self.memory_index_enabled:
    parts.append(
        "# Memory\n\n"
        "Use the `memory_search` tool to recall facts, preferences, or past context "
        "before answering questions that depend on prior sessions."
    )
else:
    memory = self.memory.get_memory_context()   # existing path, unchanged
    if memory:
        parts.append(f"# Memory\n\n{memory}")
```

All other context assembly is unchanged.

---

## Error Handling & Graceful Degradation

The invariant: **the tool never raises an exception to the agent**. All failures return either partial results or `"No relevant memory found."` The gateway never crashes due to the index.

| Failure | Degradation |
|---|---|
| `sqlite-vec` fails to load (platform/static SQLite) | BM25-only; `_vec_available = False` set once at init, logged as warning |
| Ollama unreachable during indexing | Chunks stored with NULL embedding; those chunks skipped in vector search |
| Ollama unreachable during search | Skip vector branch; RRF on BM25 results only |
| Index DB corrupt or missing | Delete and rebuild from source files; log warning |
| MEMORY.md / HISTORY.md absent | Skip gracefully; empty index until files exist |

`sqlite-vec` availability is checked once in `MemoryIndex.__init__()` and stored as `self._vec_available: bool`. Search paths branch on this flag — no exception catching in the hot path.

---

## Dependencies

Added as optional under `[project.optional-dependencies]` → `memory` in `pyproject.toml`:

| Package | Purpose | New? |
|---|---|---|
| `sqlite-vec` | Vector search extension for SQLite | Yes |
| `httpx` | Async HTTP for Ollama API | No — already a hard dep |

`watchdog` and `numpy` are explicitly excluded: `watchdog` is PR 2's concern; MMR is implemented in pure Python.

Install: `pip install -e ".[memory]"`

---

## Testing

All tests in `tests/test_memory_index/`, using `pytest-asyncio` (`asyncio_mode = "auto"`) and real SQLite in `tmp_path`:

```
tests/test_memory_index/
├── test_chunking.py      # unit: ## header split, timestamp split, overlap, min-size merge
├── test_search.py        # unit: RRF merge math, temporal decay, MMR diversity
├── test_indexer.py       # integration: round-trip index+search in tmp SQLite
├── test_embeddings.py    # unit: OllamaEmbeddingProvider with httpx mock
└── test_tool.py          # unit: MemorySearchTool.execute() with mock MemoryIndex
```

Key cases:
- Chunking splits correctly on `##` headers and `[YYYY-MM-DD]` timestamps
- RRF correctly boosts docs appearing in both BM25 and vector results
- Temporal decay: 180-day-old chunk scores ~50% lower than same-content chunk from today
- MMR: top-k results are more diverse than top-k by score alone
- BM25-only path activates when `_vec_available = False`
- Tool returns formatted output with source + line range; empty index returns "no results" string
- No mocks for the database layer — real SQLite in every integration test

---

## What This PR Does Not Include

- Session JSONL indexing — sessions are distilled into memory/history by design; indexing them would undermine that
- File watcher for live updates — PR 2
- Pre-retrieval injection — PR 2
- QMD sidecar — PR 3
- Cloud embedding providers — can follow separately; Ollama covers the local use case

---

## Follow-on PRs (for reference)

**PR 2:** Introduces `IndexService` lifecycle object (mirroring `CronService` pattern), file watcher via `watchdog`, and pre-retrieval injection in `ContextBuilder.build_messages()`. The `build_messages` method becomes async; `_process_message` in `loop.py` awaits it.

**PR 3:** `memory_index/qmd.py` subprocess wrapper for QMD CLI. Activated when `config.memory_index.backend = "qmd"`. Graceful fallback to builtin SQLite backend when QMD binary is absent.
