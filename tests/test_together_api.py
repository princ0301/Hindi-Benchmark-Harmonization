import sys

import pytest


def _import_together_api(monkeypatch, *, env_value=None):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    if env_value is None:
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TOGETHER_API_KEY", env_value)
    sys.modules.pop("scripts.together_api", None)
    import scripts.together_api as together_api
    return together_api


def test_together_api_uses_env_key(monkeypatch):
    together_api = _import_together_api(monkeypatch, env_value="test-key")

    class FakeTogether:
        def __init__(self, api_key=None):
            self.api_key = api_key

    monkeypatch.setattr(together_api, "Together", FakeTogether)
    client = together_api.get_together_client()

    assert client.api_key == "test-key"


def test_together_api_requires_key(monkeypatch):
    together_api = _import_together_api(monkeypatch, env_value=None)

    with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
        together_api.get_together_client()
