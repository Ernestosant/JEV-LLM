import importlib.util
from pathlib import Path
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location(
    "session_guard", Path(__file__).resolve().parents[1] / "tools/colab/session_guard.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class Client:
    def __init__(self, present=True, fail=False):
        self.present, self.fail = present, fail
        self.released = []

    def list_assignments(self):
        if self.fail:
            raise TimeoutError("secret-bearing errors must not be printed")
        return [SimpleNamespace(endpoint="owned")] if self.present else []

    def unassign(self, endpoint):
        self.released.append(endpoint)
        self.present = False


def test_release_verifies_server_and_removes_only_owned_session(tmp_path):
    removed = []
    client = Client()
    assert guard.release(client, SimpleNamespace(remove=removed.append), "ours", "owned", tmp_path)
    assert client.released == ["owned"] and removed == ["ours"]


def test_already_released_is_idempotent(tmp_path):
    client = Client(present=False)
    assert guard.release(client, SimpleNamespace(remove=lambda name: None), "ours", "owned", tmp_path)
    assert client.released == []


def test_api_failure_is_unknown_not_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(guard.time, "sleep", lambda seconds: None)
    client = Client(fail=True)
    removed = []
    assert not guard.release(client, SimpleNamespace(remove=removed.append), "ours", "owned", tmp_path)
    assert removed == [] and client.released == []
    assert "secret-bearing" not in (tmp_path / "session_lifecycle.jsonl").read_text()
