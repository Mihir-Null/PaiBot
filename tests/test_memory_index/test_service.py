from nanobot.config.schema import MemoryIndexConfig


def test_memory_index_config_defaults_include_injection_fields():
    cfg = MemoryIndexConfig()
    assert cfg.inject_top_k == 3
    assert cfg.watch_files is True


def test_memory_index_config_inject_top_k_can_be_zero():
    cfg = MemoryIndexConfig(inject_top_k=0)
    assert cfg.inject_top_k == 0


def test_memory_index_config_watch_files_can_be_disabled():
    cfg = MemoryIndexConfig(watch_files=False)
    assert cfg.watch_files is False


async def test_index_service_start_indexes_memory_file(tmp_path):
    from nanobot.memory_index.service import IndexService

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("## Facts\n\nUser likes Python programming.\n")

    cfg = MemoryIndexConfig()
    cfg.watch_files = False  # disable watcher — not tested here
    service = IndexService(tmp_path, cfg)
    await service.start()

    results = await service.index.search("Python programming", top_k=1)
    assert len(results) >= 1
    assert "Python" in results[0].text

    await service.stop()


async def test_index_service_stop_is_idempotent(tmp_path):
    from nanobot.memory_index.service import IndexService

    cfg = MemoryIndexConfig()
    cfg.watch_files = False
    service = IndexService(tmp_path, cfg)
    await service.start()
    await service.stop()
    await service.stop()  # must not raise


async def test_index_service_stop_without_start_is_safe(tmp_path):
    from nanobot.memory_index.service import IndexService

    cfg = MemoryIndexConfig()
    cfg.watch_files = False
    service = IndexService(tmp_path, cfg)
    await service.stop()  # must not raise — observer is None


async def test_start_watcher_creates_and_starts_observer(tmp_path):
    """_start_watcher() schedules an Observer on the memory directory."""
    from unittest.mock import MagicMock, patch

    from nanobot.memory_index.service import IndexService

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    cfg = MemoryIndexConfig()
    cfg.watch_files = False  # prevent auto-start in .start()
    service = IndexService(tmp_path, cfg)

    mock_observer = MagicMock()
    with patch("watchdog.observers.Observer", return_value=mock_observer):
        service._start_watcher()

        mock_observer.schedule.assert_called_once()
        # Second positional arg to schedule is the watched directory path
        _handler, watched_dir, *_ = mock_observer.schedule.call_args[0]
        assert watched_dir == str(tmp_path / "memory")
        mock_observer.start.assert_called_once()
        assert service._observer is mock_observer


async def test_stop_stops_and_joins_observer(tmp_path):
    """stop() calls observer.stop() and observer.join()."""
    from unittest.mock import MagicMock

    from nanobot.memory_index.service import IndexService

    cfg = MemoryIndexConfig()
    cfg.watch_files = False
    service = IndexService(tmp_path, cfg)

    mock_observer = MagicMock()
    service._observer = mock_observer

    await service.stop()

    mock_observer.stop.assert_called_once()
    mock_observer.join.assert_called_once()
    assert service._observer is None


async def test_watcher_only_triggers_for_memory_files(tmp_path):
    """Handler on_modified only schedules re-index for MEMORY.md and HISTORY.md."""
    from unittest.mock import MagicMock, patch

    from watchdog.events import FileModifiedEvent

    from nanobot.memory_index.service import IndexService

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    cfg = MemoryIndexConfig()
    cfg.watch_files = False
    service = IndexService(tmp_path, cfg)

    scheduled = []

    def fake_run_threadsafe(coro, loop):
        # Close the coroutine to avoid ResourceWarning
        coro.close()
        scheduled.append(True)

    captured_handler = None

    def capture_schedule(handler, path, recursive=False):
        nonlocal captured_handler
        captured_handler = handler

    mock_observer = MagicMock()
    mock_observer.schedule.side_effect = capture_schedule

    with patch("watchdog.observers.Observer", return_value=mock_observer), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_threadsafe):
        service._start_watcher()

        assert captured_handler is not None

        # Trigger on MEMORY.md — should schedule
        captured_handler.on_modified(FileModifiedEvent(str(memory_dir / "MEMORY.md")))
        assert len(scheduled) == 1

        # Trigger on HISTORY.md — should schedule
        captured_handler.on_modified(FileModifiedEvent(str(memory_dir / "HISTORY.md")))
        assert len(scheduled) == 2

        # Trigger on unrelated file — should NOT schedule
        captured_handler.on_modified(FileModifiedEvent(str(memory_dir / "notes.txt")))
        assert len(scheduled) == 2  # still 2
