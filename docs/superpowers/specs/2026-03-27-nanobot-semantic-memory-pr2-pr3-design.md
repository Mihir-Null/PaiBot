# Nanobot Semantic Memory — PR 2 & PR 3 Design Spec

**Date:** 2026-03-27
**Prerequisite:** PR 1 merged (`nanobot/memory_index/` module + `MemorySearchTool`)

---

## Context

PR 1 built a standalone index with an explicit `memory_search` tool. The agent must call the tool to retrieve memory. PR 2 makes retrieval automatic (pre-retrieval injection on every turn) and keeps the index live (file watcher). PR 3 adds an optional QMD sidecar for query expansion + LLM re-ranking.

---

## PR 2 — Pre-retrieval Injection + File Watcher

### What It Does

1. **Auto-inject** top-k relevant memory chunks into every agent turn's context — no explicit tool call required
2. **Live updates** — MEMORY.md/HISTORY.md changes on disk trigger an incremental re-index without gateway restart

### Architecture

#### New: `IndexService` (`nanobot/memory_index/service.py`)

Lifecycle object that owns `MemoryIndex` + the file watcher. Mirrors `nanobot/cron/service.py` structure.

```python
class IndexService:
    def __init__(self, workspace: Path, cfg: MemoryIndexConfig) -> None:
        self.index = MemoryIndex(workspace, cfg)
        self._cfg = cfg
        self._observer: Observer | None = None   # watchdog

    async def start(self) -> None:
        await self.index.startup_index()
        if self._cfg.watch_files:
            self._start_watcher()

    async def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def _start_watcher(self) -> None:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        # On MEMORY.md or HISTORY.md modify/create: schedule _index_file via asyncio.run_coroutine_threadsafe
        ...
```

#### Modified: `ContextBuilder.build_messages()` → async

`build_messages()` currently builds the message list synchronously. It already receives `current_message`. Change it to `async def` so it can await `index.search()` for pre-retrieval injection.

```python
async def build_messages(
    self,
    current_message: InboundMessage,
    history: list[dict],
    index: IndexService | None = None,
) -> list[dict]:
    messages = [...]  # existing logic
    if index and self._cfg.inject_top_k > 0:
        query = current_message.content
        results = await index.index.search(query, top_k=self._cfg.inject_top_k)
        if results:
            injection = "Relevant memory:\n\n" + "\n\n---\n\n".join(
                f"[{r.source} L{r.start_line}–{r.end_line}]\n{r.text}" for r in results
            )
            # Prepend as a system message or inject into the last user message
            messages.insert(-1, {"role": "system", "content": injection})
    return messages
```

`_process_message` in `loop.py` changes `build_messages(...)` → `await build_messages(...)`.

#### Config additions to `MemoryIndexConfig`

```python
class MemoryIndexConfig(Base):
    # existing fields unchanged...
    inject_top_k: int = 3       # chunks auto-injected per turn; 0 = disabled
    watch_files: bool = True    # enable watchdog file watcher
```

#### pyproject.toml

Add to `[memory]` extras:
```toml
memory = [
    "sqlite-vec>=0.1.0",
    "watchdog>=3.0.0",
]
```

### Files Changed

| File | Change |
|---|---|
| `nanobot/memory_index/service.py` | **New** — `IndexService` |
| `nanobot/agent/context.py` | `build_messages()` → `async def`; accepts `index` param |
| `nanobot/agent/loop.py` | Constructs `IndexService`; awaits `build_messages()` |
| `nanobot/config/schema.py` | Add `inject_top_k`, `watch_files` to `MemoryIndexConfig` |
| `pyproject.toml` | Add `watchdog>=3.0.0` to `[memory]` |

### Integration Point Detail

Current `loop.py` call site (in `_process_message`):
```python
messages = self.context.build_messages(inbound, history)
```

After PR 2:
```python
messages = await self.context.build_messages(inbound, history, index=self._index_service)
```

`self._index_service` is set in `_register_default_tools()` when `memory_index_config.enabled` is True (replaces current `self._memory_index` temporary storage). `startup_index()` is called via `IndexService.start()` in `run()`/`process_direct()` (same deferred pattern as PR 1).

### Testing

- `IndexService.start()` / `stop()` lifecycle
- File change event triggers re-index (mock `watchdog`)
- `build_messages()` injects results when index has matches
- `build_messages()` is unchanged when index is None or `inject_top_k=0`
- No injection when index returns empty results
- `_process_message` correctly awaits `build_messages()`

---

## PR 3 — QMD Sidecar

### What It Does

Optional backend swap: replaces the builtin SQLite search pipeline with a QMD (query-time model decomposition) subprocess for query expansion + LLM re-ranking. Gracefully falls back to the PR 1 SQLite backend when QMD binary is absent.

### Architecture

#### New: `nanobot/memory_index/qmd.py`

```python
class QMDBackend:
    """Thin subprocess wrapper for the QMD CLI."""

    def __init__(self, binary: str, index_path: Path) -> None:
        self._binary = binary
        self._index_path = index_path

    @classmethod
    def is_available(cls, binary: str) -> bool:
        import shutil
        return shutil.which(binary) is not None

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        proc = await asyncio.create_subprocess_exec(
            self._binary, "search",
            "--index", str(self._index_path),
            "--query", query,
            "--top-k", str(top_k),
            "--format", "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        # Parse JSON output → list[SearchResult]
        ...
```

#### Modified: `MemoryIndex.search()` — backend dispatch

```python
async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
    if self._qmd is not None:
        return await self._qmd.search(query, top_k)
    # existing SQLite path...
```

`self._qmd` is set in `__init__` when `cfg.backend == "qmd"` AND `QMDBackend.is_available(cfg.qmd_binary)`.

#### Config additions to `MemoryIndexConfig`

```python
class MemoryIndexConfig(Base):
    # existing fields unchanged...
    backend: str = "sqlite"      # "sqlite" | "qmd"
    qmd_binary: str = "qmd"      # path or name for shutil.which lookup
```

### Files Changed

| File | Change |
|---|---|
| `nanobot/memory_index/qmd.py` | **New** — `QMDBackend` subprocess wrapper |
| `nanobot/memory_index/index.py` | Backend dispatch in `search()`; `QMDBackend` init in `__init__` |
| `nanobot/config/schema.py` | Add `backend`, `qmd_binary` to `MemoryIndexConfig` |

### Testing

- `QMDBackend.is_available()` returns False when binary absent
- `search()` routes to QMD backend when `backend="qmd"` and binary present
- `search()` falls back to SQLite when QMD binary absent (even if `backend="qmd"`)
- Mock subprocess for QMD output parsing tests
- No regression to SQLite path when `backend="sqlite"` (default)

---

## What Does NOT Change in PR 2 or PR 3

- `nanobot/memory_index/indexer.py` — chunking logic unchanged
- `nanobot/memory_index/embeddings.py` — embedding provider unchanged
- `nanobot/memory_index/search.py` — search functions unchanged
- `nanobot/memory_index/schema.sql` — schema unchanged
- `nanobot/agent/tools/memory_search.py` — tool unchanged
- The explicit `memory_search` tool remains registered and callable — PR 2 adds auto-injection on top, does not remove explicit search
