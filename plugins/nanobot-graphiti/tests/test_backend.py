"""Tests for GraphitiMemoryBackend and GraphitiConfig."""

import pytest


# ── Config ──────────────────────────────────────────────────────────────────

def test_graphiti_config_defaults():
    from nanobot_graphiti.config import GraphitiConfig

    cfg = GraphitiConfig()
    assert cfg.graph_db == "kuzu"
    assert cfg.kuzu_path == "~/.nanobot/workspace/memory/graph"
    assert cfg.top_k == 5
    assert cfg.scope == "user"
    assert cfg.embedding_model == "text-embedding-3-small"


def test_graphiti_config_accepts_neo4j():
    from nanobot_graphiti.config import GraphitiConfig

    cfg = GraphitiConfig(graph_db="neo4j", neo4j_uri="bolt://myhost:7687", neo4j_password="secret")
    assert cfg.graph_db == "neo4j"
    assert cfg.neo4j_uri == "bolt://myhost:7687"


def test_graphiti_config_from_nanobot_config_kuzu():
    """_from_nanobot_config() parses memory.model_extra["graphiti"] section."""
    from unittest.mock import MagicMock
    from nanobot_graphiti.config import GraphitiConfig

    nanobot_config = MagicMock()
    nanobot_config.memory.model_extra = {"graphiti": {"graph_db": "kuzu", "top_k": 10}}

    cfg = GraphitiConfig._from_nanobot_config(nanobot_config)
    assert cfg.graph_db == "kuzu"
    assert cfg.top_k == 10


def test_graphiti_config_from_nanobot_config_missing_section():
    """_from_nanobot_config() falls back to defaults when section absent."""
    from unittest.mock import MagicMock
    from nanobot_graphiti.config import GraphitiConfig

    nanobot_config = MagicMock()
    nanobot_config.memory.model_extra = {}

    cfg = GraphitiConfig._from_nanobot_config(nanobot_config)
    assert cfg.graph_db == "kuzu"
    assert cfg.top_k == 5


# ── Backend contract ─────────────────────────────────────────────────────────

def test_graphiti_backend_is_memory_backend():
    from nanobot.agent.memory import MemoryBackend
    from nanobot_graphiti.backend import GraphitiMemoryBackend

    assert issubclass(GraphitiMemoryBackend, MemoryBackend)


def test_graphiti_backend_consolidates_per_turn_is_true():
    from nanobot_graphiti.backend import GraphitiMemoryBackend
    from nanobot_graphiti.config import GraphitiConfig

    backend = GraphitiMemoryBackend(GraphitiConfig())
    assert backend.consolidates_per_turn is True


async def test_graphiti_backend_start_calls_build_indices(mock_graphiti, mock_provider):
    from nanobot_graphiti.backend import GraphitiMemoryBackend
    from nanobot_graphiti.config import GraphitiConfig

    backend = GraphitiMemoryBackend(GraphitiConfig(), _graphiti_factory=lambda **kw: mock_graphiti)
    await backend.start(mock_provider)

    mock_graphiti.build_indices_and_constraints.assert_awaited_once()


async def test_graphiti_backend_stop_closes_client(mock_graphiti, mock_provider):
    from nanobot_graphiti.backend import GraphitiMemoryBackend
    from nanobot_graphiti.config import GraphitiConfig

    backend = GraphitiMemoryBackend(GraphitiConfig(), _graphiti_factory=lambda **kw: mock_graphiti)
    await backend.start(mock_provider)
    await backend.stop()

    mock_graphiti.close.assert_awaited_once()


async def test_graphiti_backend_stop_is_safe_before_start():
    """stop() before start() must not raise."""
    from nanobot_graphiti.backend import GraphitiMemoryBackend
    from nanobot_graphiti.config import GraphitiConfig

    backend = GraphitiMemoryBackend(GraphitiConfig())
    await backend.stop()  # no exception
