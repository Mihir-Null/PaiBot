# Graphiti Memory Lobe — Design Spec

**Date:** 2026-03-29
**Status:** Approved for implementation planning

## Goal

Replace nanobot's flat-file memory system (MEMORY.md + HISTORY.md + LLM consolidation) with a temporal knowledge graph backed by Graphiti. Memory becomes an ambient "lobe" — automatically capturing facts from every conversation turn and pre-retrieving relevant context before every turn — rather than a file the agent reads and writes manually.

---

## Background & Motivation

nanobot's current memory system has three structural weaknesses:

1. **Unbounded context bloat.** The entire MEMORY.md is injected into every system prompt. As memory grows, attention degrades and token costs rise linearly.
2. **LLM-dependent consolidation.** `MemoryConsolidator` calls the LLM after every consolidation window to summarize conversations into MEMORY.md. Failures produce stale or missing memories with no audit trail.
3. **No temporal reasoning.** Facts are overwritten or appended with no record of when they became true or false. A corrected fact coexists with its contradiction.

Graphiti (Apache 2.0, `graphiti-core` on PyPI) addresses all three:
- Retrieval is query-scoped: only relevant facts are injected per turn (solves #1)
- Extraction runs on raw episodes, not LLM summaries, and marks stale facts invalid rather than deleting them (solves #2 and #3)
- LongMemEval benchmark: Graphiti 63.8% vs mem0 49.0% vs full-context baseline 60.2% at 29s latency

---

## Architecture

### Data Flow

```
Inbound turn:
  GraphitiBackend.retrieve(current_message, group_id)
    → hybrid semantic + BM25 search over temporal graph
    → top-k facts injected as a context block before LLM sees the message

Outbound turn (background, non-blocking):
  GraphitiBackend.consolidate(turn_messages, group_id)
    → graphiti.add_episode() extracts entities + relationships
    → new facts added; stale facts marked invalid with timestamp
    → raw episode preserved in Graphiti's episode store (replaces HISTORY.md)
```

### What Is Replaced

| Old component | Replaced by |
|---|---|
| `MemoryStore` (MEMORY.md + HISTORY.md writes) | `GraphitiMemoryBackend.consolidate()` |
| `MemoryStore.get_memory_context()` (full-file injection) | `GraphitiMemoryBackend.retrieve()` (query-scoped injection) |
| `MemoryConsolidator` LLM summarization call | Graphiti's entity extraction on raw episodes |
| `MemoryConsolidator.maybe_consolidate_by_tokens()` LLM step | Simple session message drop (facts already in graph) |
| HISTORY.md grep audit trail | Graphiti episode store (richer, timestamped, queryable) |

Token-based session pruning (dropping old messages when context window fills) is **retained** but simplified: it no longer calls the LLM since facts are captured continuously by Graphiti.

---

## Component 1: `MemoryBackend` ABC (nanobot core)

Added to `nanobot/agent/memory.py`. Minimal interface — five methods:

```python
class MemoryBackend(ABC):
    async def start(self, provider: LLMProvider) -> None: ...  # lifecycle: connect, build indices
    async def stop(self) -> None: ...                          # lifecycle: graceful shutdown
    async def consolidate(
        self,
        messages: list[dict],
        session_key: str,
    ) -> None: ...                              # post-turn: extract + store facts
    async def retrieve(
        self,
        query: str,
        session_key: str,
        top_k: int = 5,
    ) -> str: ...                               # pre-turn: return formatted context block
    def get_tools(self) -> list[Tool]: ...      # tools this backend exposes to the agent (default: [])

    @property
    def consolidates_per_turn(self) -> bool:    # True = consolidate every turn; False = token-pressure only
        return False
```

`start()` receives the nanobot `LLMProvider` instance so backends that need LLM access (e.g. Graphiti's entity extraction) reuse the same configured provider rather than requiring a separate API key.

`MemoryStore` implements `MemoryBackend` so existing users see no behavior change when the plugin is not installed.

Entry-point group: `nanobot.memory`. `AgentLoop` scans at startup; config field `memory.backend` (string) selects which registered backend to use. Falls back to the built-in `MemoryStore` implementation if the field is absent or `"default"`.

---

## Component 2: nanobot Core Changes

**`nanobot/agent/memory.py`**
- Add `MemoryBackend` ABC (as above)
- `MemoryStore` implements `MemoryBackend`: `consolidates_per_turn = False`; `get_tools()` returns `[]`; `start()` ignores the provider (file-based, no LLM needed at init); `consolidate()` and `retrieve()` wrap the existing `MemoryStore.consolidate()` and `get_memory_context()` methods — zero behavior change for existing users
- `MemoryConsolidator.maybe_consolidate_by_tokens()`: session message-drop loop is retained unchanged; the LLM summarization call (`store.consolidate()`) is gated on `backend.consolidates_per_turn == False` — when a per-turn backend (Graphiti) is active, the pruning loop only drops old messages without calling the LLM, since facts were already captured episode-by-episode

**`nanobot/agent/context.py`**
- `ContextBuilder.__init__` accepts `memory_backend: MemoryBackend | None = None`
- `build_messages()` becomes `async`; when `memory_backend` is set, calls `await memory_backend.retrieve(current_message, session_key, top_k)` and injects the returned string as a system-role block immediately before the conversation history
- `build_system_prompt()` no longer reads or injects MEMORY.md; workspace description in `_get_identity()` drops MEMORY.md/HISTORY.md mentions and instead describes the memory tool surface:
  > Relevant memories from past conversations are automatically surfaced. Use `memory_search` for targeted recall, `memory_forget` to correct errors, `memory_list` to audit stored facts.

**`nanobot/agent/loop.py`**
- `AgentLoop.__init__` discovers backend via `nanobot.memory` entry-point and `config.memory.backend`; calls `await backend.start(self.provider)` in `run()`/`process_direct()` (same deferred-init pattern as `startup_index()` in the QMD branch — not in `__init__` to avoid event loop issues); passes backend to `ContextBuilder`; registers `backend.get_tools()` into the tool registry
- `_process_message()`: when `backend.consolidates_per_turn` is True, fires `backend.consolidate(turn_messages, session_key)` as a background task after the agent loop completes; when False, the existing `maybe_consolidate_by_tokens()` path governs (no change for `MemoryStore`)
- `await self.context.build_messages(...)` — updated call site for the now-async signature

**`nanobot/config/schema.py`**
- Add `MemoryConfig(Base)` with `backend: str = "default"`
- Add `Config.memory: MemoryConfig = Field(default_factory=MemoryConfig)`

---

## Component 3: `nanobot-graphiti` Plugin Package

Separate installable package. Installs alongside nanobot:

```
pip install nanobot-graphiti
```

Registers under `nanobot.memory` entry-point in its `pyproject.toml`:

```toml
[project.entry-points."nanobot.memory"]
graphiti = "nanobot_graphiti:GraphitiMemoryBackend"
```

### Package Structure

```
nanobot_graphiti/
  __init__.py        # exports GraphitiMemoryBackend
  backend.py         # GraphitiMemoryBackend(MemoryBackend)
  tools.py           # MemorySearchTool, MemoryForgetTool, MemoryListTool
  config.py          # GraphitiConfig pydantic model
```

### `GraphitiMemoryBackend`

- **`start(provider)`**: receives nanobot's `LLMProvider`; constructs a Graphiti-compatible LLM/embedding client adapter from it (Graphiti accepts any OpenAI-compatible endpoint, which nanobot's `openai_compat_provider` already speaks); constructs `graphiti_core.Graphiti` client with configured graph DB; calls `await graphiti.build_indices_and_constraints()`
- **`consolidate(messages, session_key)`**: formats the turn's messages into a `graphiti_core.EpisodeBody`; calls `await graphiti.add_episode(name=session_key, episode_body=..., group_id=group_id_from(session_key))`; errors are caught and logged — never block the main loop
- **`retrieve(query, session_key, top_k)`**: calls `await graphiti.search(query, group_ids=[group_id], num_results=top_k)`; formats results as:
  ```
  [Memory — {N} relevant facts]
  • {fact text} ({source date})
  ...
  ```
  Returns empty string if no results — no block injected

### Session Key → `group_id` Scoping

Session key format: `channel:chat_id` (e.g., `telegram:123456`).

Default (`scope: "user"`): `group_id = chat_id` — memories are **user-level**, shared across channels. A user's Telegram and Discord conversations accumulate one memory graph.

Alternative (`scope: "session"`): `group_id = session_key` — memories are **session-scoped**, isolated per channel+chat. Useful for multi-user or shared-channel deployments.

Configured via `memory.graphiti.scope`.

### Config Schema (`GraphitiConfig`)

```python
class GraphitiConfig(Base):
    graph_db: Literal["kuzu", "neo4j", "falkordb"] = "kuzu"
    kuzu_path: str = "~/.nanobot/workspace/memory/graph"  # kuzu only
    neo4j_uri: str = "bolt://localhost:7687"               # neo4j only
    neo4j_user: str = "neo4j"                              # neo4j only
    neo4j_password: str = ""                               # neo4j only
    falkordb_host: str = "localhost"                       # falkordb only
    falkordb_port: int = 6379                              # falkordb only
    top_k: int = 5
    scope: Literal["user", "session"] = "user"
```

**Kuzu is the default** — embedded graph DB (no server process, file on disk). Users who want a persistent server-backed graph switch to `neo4j` or `falkordb`.

In `config.yaml`:

```yaml
memory:
  backend: graphiti
  graphiti:
    graph_db: kuzu
    top_k: 5
    scope: user
```

---

## Component 4: Tool Surface

Three tools returned by `GraphitiMemoryBackend.get_tools()` and registered by `AgentLoop` via `self.tools.register(t)` for each tool in `backend.get_tools()` during `_register_default_tools()`. The base `MemoryBackend.get_tools()` returns `[]`, so no tools are added for the default `MemoryStore` backend.

| Tool | Description | Key parameters |
|---|---|---|
| `memory_search` | Semantic search over the memory graph | `query: str`, `top_k: int = 10` |
| `memory_forget` | Delete a specific fact by its graph node ID | `fact_id: str`, `reason: str` |
| `memory_list` | List all stored facts for the current user | `limit: int = 50` |

`memory_forget` and `memory_list` use the session's `group_id` to scope results to the current user.

The agent uses these tools reactively — e.g., when a user says "you have that wrong" or "what do you remember about me?" The automatic pre-retrieval injection handles the common case without any tool call.

---

## What Is Not in Scope (v1)

- Graph memory visualization / admin UI
- Multi-agent shared memory (cross-session writes)
- Custom entity type schemas
- Memory export/import tooling
- FalkorDB or Amazon Neptune as tested backends (supported by config, not validated)
- Gradual migration path from existing MEMORY.md files (out of scope; users start fresh)

---

## Testing Strategy

**nanobot core tests** (no new dependencies):
- `MemoryBackend` ABC can be implemented and registered via entry-point
- `MemoryStore` implements `MemoryBackend`: `consolidates_per_turn` is False, `get_tools()` returns `[]`
- `build_messages()` async signature produces correct message list with and without an injected context block
- `AgentLoop` falls back to `MemoryStore` when no `nanobot.memory` entry-point is registered
- `AgentLoop` fires `backend.consolidate()` post-turn only when `backend.consolidates_per_turn` is True
- Token-based session pruning skips LLM call when `backend.consolidates_per_turn` is True; retains it when False

**`nanobot-graphiti` plugin tests** (mocked Graphiti client):
- `GraphitiMemoryBackend.consolidates_per_turn` is True
- `GraphitiMemoryBackend.get_tools()` returns exactly three tools: `memory_search`, `memory_forget`, `memory_list`
- `GraphitiMemoryBackend.consolidate()` calls `graphiti.add_episode()` with correct `group_id`
- `GraphitiMemoryBackend.retrieve()` calls `graphiti.search()` and formats output correctly
- `retrieve()` returns empty string when search returns no results
- `consolidate()` errors are swallowed and logged, never raised
- `scope: "user"` strips channel prefix from session key; `scope: "session"` preserves it
- `GraphitiConfig` defaults: `graph_db="kuzu"`, `top_k=5`, `scope="user"`
- Tool tests: `memory_search`, `memory_forget`, `memory_list` call correct Graphiti methods with session-scoped `group_id`
