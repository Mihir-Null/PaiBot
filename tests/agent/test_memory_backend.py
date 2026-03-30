"""Tests for memory backend configuration."""

import pytest


def test_memory_config_defaults():
    from nanobot.config.schema import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.backend == "default"


def test_memory_config_backend_can_be_set():
    from nanobot.config.schema import MemoryConfig

    cfg = MemoryConfig(backend="graphiti")
    assert cfg.backend == "graphiti"


def test_config_has_memory_field():
    from nanobot.config.schema import Config

    cfg = Config()
    assert cfg.memory.backend == "default"


def test_memory_backend_abc_cannot_be_instantiated_directly():
    from nanobot.agent.memory import MemoryBackend

    with pytest.raises(TypeError):
        MemoryBackend()


def test_memory_store_is_memory_backend(tmp_path):
    from nanobot.agent.memory import MemoryBackend, MemoryStore

    store = MemoryStore(tmp_path)
    assert isinstance(store, MemoryBackend)


def test_memory_store_consolidates_per_turn_is_false(tmp_path):
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(tmp_path)
    assert store.consolidates_per_turn is False


def test_memory_store_get_tools_returns_empty_list(tmp_path):
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(tmp_path)
    assert store.get_tools() == []


async def test_memory_store_retrieve_returns_memory_context(tmp_path):
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(tmp_path)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(exist_ok=True)
    (memory_dir / "MEMORY.md").write_text("I like coffee")

    result = await store.retrieve("anything", "telegram:123")
    assert "I like coffee" in result


async def test_memory_store_retrieve_returns_empty_string_when_no_file(tmp_path):
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(tmp_path)
    result = await store.retrieve("anything", "telegram:123")
    assert result == ""


async def test_memory_store_consolidate_is_noop(tmp_path):
    from nanobot.agent.memory import MemoryStore

    store = MemoryStore(tmp_path)
    # Should not raise and should not create any files
    await store.consolidate([{"role": "user", "content": "hello"}], "telegram:123")
    assert not (tmp_path / "memory" / "MEMORY.md").exists()
