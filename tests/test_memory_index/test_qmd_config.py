def test_memory_index_config_defaults_backend_sqlite():
    from nanobot.config.schema import MemoryIndexConfig

    cfg = MemoryIndexConfig()
    assert cfg.backend == "sqlite"
    assert cfg.qmd_binary == "qmd"


def test_memory_index_config_backend_can_be_qmd():
    from nanobot.config.schema import MemoryIndexConfig

    cfg = MemoryIndexConfig(backend="qmd")
    assert cfg.backend == "qmd"


def test_memory_index_config_qmd_binary_can_be_custom_path():
    from nanobot.config.schema import MemoryIndexConfig

    cfg = MemoryIndexConfig(backend="qmd", qmd_binary="/usr/local/bin/qmd")
    assert cfg.qmd_binary == "/usr/local/bin/qmd"
