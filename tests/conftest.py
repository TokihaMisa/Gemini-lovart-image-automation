import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_preflight: run webui.run_process with its real pre-start checks"
    )


@pytest.fixture(autouse=True)
def _bypass_webui_preflight(request, monkeypatch):
    """Dashboard tests fake the child process; the pre-start checks have their own tests."""
    webui = sys.modules.get("webui")
    if webui is None or request.node.get_closest_marker("real_preflight"):
        return
    monkeypatch.setattr(webui, "_preflight_problems", lambda *_args, **_kwargs: [])
