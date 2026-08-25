"""MCP-обёртка: модуль грузится без пакета mcp; build_server даёт понятную ошибку."""
import sys

import pytest


def test_module_imports_without_mcp():
    import realmemory.api.mcp_server as m

    assert callable(m.build_server)
    assert callable(m.main)


def test_build_server_wire_or_clear_error(tmp_path, tiny_cfg, clock):
    from realmemory import Hippocampus

    hippo = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    try:
        import realmemory.api.mcp_server as m

        try:
            import fastmcp  # noqa: F401
            has_fastmcp = True
        except ImportError:
            has_fastmcp = False

        if has_fastmcp:
            server = m.build_server(hippo)
            assert hasattr(server, "run")
        else:
            sys.modules.pop("fastmcp", None)
            with pytest.raises(ImportError, match=r"realmemory\[mcp\]"):
                m.build_server(hippo)
    finally:
        hippo.close()
