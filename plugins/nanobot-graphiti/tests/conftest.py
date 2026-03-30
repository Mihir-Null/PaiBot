"""Shared fixtures for nanobot-graphiti tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_graphiti():
    """A fully mocked Graphiti client."""
    g = MagicMock()
    g.build_indices_and_constraints = AsyncMock()
    g.add_episode = AsyncMock()
    g.search = AsyncMock(return_value=[])
    g.close = AsyncMock()
    g.driver = MagicMock()
    return g


@pytest.fixture
def mock_provider():
    """A minimal nanobot LLMProvider mock."""
    p = MagicMock()
    p.api_key = "test-key"
    p.api_base = "https://api.openai.com/v1"
    p.get_default_model.return_value = "gpt-4o-mini"
    return p


@pytest.fixture
def session_key():
    return "telegram:123456"
