import json
import os

from jagimage import desktop, server


def test_remembered_keys_come_back_next_time(tmp_path, monkeypatch):
    for env in ("GEMINI_API_KEY", "OPENAI_API_KEY", "JAGIMAGE_RUNPOD_TOKEN", "JAGIMAGE_RUNPOD_URL"):
        monkeypatch.delenv(env, raising=False)
    keys = tmp_path / "Jag Studio" / "keys.json"
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    server.save_keys(keys)
    assert json.loads(keys.read_text()) == {"GEMINI_API_KEY": "g-key"}
    if os.name != "nt":
        assert keys.stat().st_mode & 0o077 == 0  # only the user can read it
    monkeypatch.delenv("GEMINI_API_KEY")
    keys.write_text(json.dumps({"GEMINI_API_KEY": "g-key", "PATH": "/evil"}))
    path_before = os.environ.get("PATH")
    server.load_keys(keys)
    assert os.environ["GEMINI_API_KEY"] == "g-key" and os.environ.get("PATH") == path_before  # only API keys
    server.load_keys(tmp_path / "missing.json")  # no file yet: nothing happens


def test_desktop_app_starts_and_answers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(desktop, "settings_dir", lambda: tmp_path / "settings")
    assert desktop.main(["--smoke", "--out", str(tmp_path / "photos")]) == 0
    assert "smoke test passed" in capsys.readouterr().out and (tmp_path / "photos").is_dir()
