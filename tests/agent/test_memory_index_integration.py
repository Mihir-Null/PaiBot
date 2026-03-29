from unittest.mock import MagicMock


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


async def test_agent_loop_creates_index_service_when_enabled(tmp_path):
    """AgentLoop._index_service is an IndexService instance when memory_index is enabled."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import MemoryIndexConfig
    from nanobot.memory_index.service import IndexService

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    cfg = MemoryIndexConfig()
    cfg.enabled = True
    cfg.watch_files = False

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        memory_index_config=cfg,
    )

    assert isinstance(loop._index_service, IndexService)
    assert loop.tools.get("memory_search") is not None


def test_agent_loop_does_not_register_tool_when_disabled(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import MemoryIndexConfig

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        memory_index_config=MemoryIndexConfig(),  # enabled=False
    )
    assert loop.tools.get("memory_search") is None
    assert loop._index_service is None


async def test_agent_loop_start_schedules_index_service_start(tmp_path):
    """run() schedules IndexService.start() as a background task exactly once."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import MemoryIndexConfig

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    cfg = MemoryIndexConfig()
    cfg.enabled = True
    cfg.watch_files = False

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        memory_index_config=cfg,
    )
    loop._running = False  # prevent run() from looping

    start_calls = []

    async def fake_start():
        start_calls.append(1)

    loop._index_service.start = fake_start

    with patch.object(loop, "_connect_mcp", new_callable=AsyncMock):
        try:
            await asyncio.wait_for(loop.run(), timeout=0.5)
        except asyncio.TimeoutError:
            pass

    # Drain background tasks so fake_start actually runs
    if loop._background_tasks:
        await asyncio.gather(*loop._background_tasks, return_exceptions=True)

    assert len(start_calls) == 1


async def test_agent_loop_start_schedules_only_once(tmp_path):
    """Calling process_direct() twice does not schedule IndexService.start() twice."""
    import asyncio
    from unittest.mock import AsyncMock

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import MemoryIndexConfig

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 4096
    provider.chat_with_retry = AsyncMock(return_value=MagicMock(
        has_tool_calls=False,
        content="Hi",
        finish_reason="stop",
        tool_calls=[],
        usage={},
        reasoning_content=None,
        thinking_blocks=None,
    ))

    cfg = MemoryIndexConfig()
    cfg.enabled = True
    cfg.watch_files = False

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        memory_index_config=cfg,
    )

    start_calls = []

    async def fake_start():
        start_calls.append(1)

    loop._index_service.start = fake_start

    await loop.process_direct("hello")
    await loop.process_direct("world")

    if loop._background_tasks:
        await asyncio.gather(*loop._background_tasks, return_exceptions=True)

    assert len(start_calls) == 1  # only once, not twice
