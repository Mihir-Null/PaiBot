"""Tests for pre-retrieval injection in build_messages()."""

from unittest.mock import AsyncMock, MagicMock

from nanobot.memory_index.search import SearchResult


def _make_result(text: str) -> SearchResult:
    return SearchResult(text=text, source="MEMORY.md", start_line=1, end_line=2, score=0.9)


async def test_build_messages_injects_results_when_index_provided(tmp_path):
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path, memory_index_enabled=True)

    mock_service = MagicMock()
    mock_service.inject_top_k = 3
    mock_service.search = AsyncMock(return_value=[_make_result("User prefers dark mode.")])

    messages = await ctx.build_messages(
        history=[],
        current_message="What theme do I like?",
        index=mock_service,
    )

    injected = [m for m in messages if m["role"] == "system" and "Relevant memory" in m.get("content", "")]
    assert len(injected) == 1
    assert "dark mode" in injected[0]["content"]


async def test_injection_message_is_inserted_before_user_message(tmp_path):
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path, memory_index_enabled=True)

    mock_service = MagicMock()
    mock_service.inject_top_k = 3
    mock_service.search = AsyncMock(return_value=[_make_result("Some fact.")])

    messages = await ctx.build_messages(
        history=[],
        current_message="hello",
        index=mock_service,
    )

    # Injection must come before the last (user) message
    injection_idx = next(
        i for i, m in enumerate(messages)
        if m["role"] == "system" and "Relevant memory" in m.get("content", "")
    )
    assert injection_idx == len(messages) - 2  # second-to-last


async def test_injection_includes_source_and_line_range(tmp_path):
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path, memory_index_enabled=True)

    result = SearchResult(
        text="User likes coffee.",
        source="MEMORY.md",
        start_line=5,
        end_line=7,
        score=0.8,
    )
    mock_service = MagicMock()
    mock_service.inject_top_k = 3
    mock_service.search = AsyncMock(return_value=[result])

    messages = await ctx.build_messages(
        history=[],
        current_message="What do I drink?",
        index=mock_service,
    )

    injected = [m for m in messages if m["role"] == "system" and "Relevant memory" in m.get("content", "")]
    assert "[MEMORY.md L5–7]" in injected[0]["content"]
    assert "User likes coffee." in injected[0]["content"]


async def test_no_injection_when_index_is_none(tmp_path):
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path, memory_index_enabled=False)

    messages = await ctx.build_messages(
        history=[],
        current_message="hello",
        index=None,
    )

    injected = [m for m in messages if m["role"] == "system" and "Relevant memory" in m.get("content", "")]
    assert len(injected) == 0


async def test_no_injection_when_empty_results(tmp_path):
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path, memory_index_enabled=True)

    mock_service = MagicMock()
    mock_service.inject_top_k = 3
    mock_service.search = AsyncMock(return_value=[])

    messages = await ctx.build_messages(
        history=[],
        current_message="anything",
        index=mock_service,
    )

    injected = [m for m in messages if m["role"] == "system" and "Relevant memory" in m.get("content", "")]
    assert len(injected) == 0


async def test_no_injection_when_inject_top_k_is_zero(tmp_path):
    from nanobot.agent.context import ContextBuilder

    ctx = ContextBuilder(tmp_path, memory_index_enabled=True)

    mock_service = MagicMock()
    mock_service.inject_top_k = 0
    mock_service.search = AsyncMock(return_value=[_make_result("Should not appear.")])

    messages = await ctx.build_messages(
        history=[],
        current_message="anything",
        index=mock_service,
    )

    mock_service.search.assert_not_called()
    injected = [m for m in messages if m["role"] == "system" and "Relevant memory" in m.get("content", "")]
    assert len(injected) == 0


async def test_estimate_session_prompt_tokens_is_async(tmp_path):
    """MemoryConsolidator.estimate_session_prompt_tokens() must be awaitable."""
    from nanobot.agent.memory import MemoryConsolidator

    async def fake_build(**kwargs):
        return [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    consolidator = MemoryConsolidator(
        workspace=tmp_path,
        provider=MagicMock(),
        model="test-model",
        sessions=MagicMock(),
        context_window_tokens=8192,
        build_messages=fake_build,
        get_tool_definitions=lambda: [],
    )

    mock_session = MagicMock()
    mock_session.messages = [{"role": "user", "content": "hi"}]
    mock_session.key = "cli:test"
    mock_session.get_history.return_value = []

    tokens, source = await consolidator.estimate_session_prompt_tokens(mock_session)
    assert isinstance(tokens, int)
    assert isinstance(source, str)
