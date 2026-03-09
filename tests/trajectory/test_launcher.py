import os
import shlex
import sys
import time
from pathlib import Path

import pytest
import yaml

from trajectory.launcher import MASLauncher


def _make_template() -> dict:
    return {
        "llm": {
            "base_url": "http://original:8000/v1",
            "model": "Qwen3-4B",
            "max_tokens": 1024,
        },
        "agents": {
            "verifier": {"temperature": 0.2},
            "searcher": {"temperature": 0.6},
            "answerer": {"temperature": 0.4},
        },
        "metadata": {
            "experiment": "task-5",
        },
    }


def test_prepare_config_replaces_base_url():
    launcher = MASLauncher()
    try:
        out = launcher.prepare_config(
            config_template=_make_template(),
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )
        result = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert result["llm"]["base_url"] == "http://127.0.0.1:19000/v1"
    finally:
        launcher.cleanup()


def test_prepare_config_injects_agent_model_names():
    launcher = MASLauncher()
    try:
        out = launcher.prepare_config(
            config_template=_make_template(),
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )
        result = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert result["agents"]["verifier"]["model"] == "verifier"
        assert result["agents"]["searcher"]["model"] == "searcher"
        assert result["agents"]["answerer"]["model"] == "answerer"
    finally:
        launcher.cleanup()


def test_prepare_config_preserves_other_fields():
    template = _make_template()
    launcher = MASLauncher()
    try:
        out = launcher.prepare_config(
            config_template=template,
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )
        result = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert result["llm"]["model"] == "Qwen3-4B"
        assert result["llm"]["max_tokens"] == 1024
        assert result["agents"]["verifier"]["temperature"] == 0.2
        assert result["agents"]["searcher"]["temperature"] == 0.6
        assert result["metadata"]["experiment"] == "task-5"
        assert "model" not in template["agents"]["verifier"]
    finally:
        launcher.cleanup()


def test_prepare_config_rewrites_agent_llm_base_urls():
    template = _make_template()
    template["agents"]["verifier"]["llm"] = {
        "base_url": "http://verifier:8000/v1",
        "api_key": "EMPTY",
    }
    template["agents"]["searcher"]["llm"] = {
        "base_url": "http://searcher:8000/v1",
        "api_key": "EMPTY",
    }

    launcher = MASLauncher()
    try:
        out = launcher.prepare_config(
            config_template=template,
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )
        result = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert result["agents"]["verifier"]["llm"]["base_url"] == "http://127.0.0.1:19000/v1"
        assert result["agents"]["searcher"]["llm"]["base_url"] == "http://127.0.0.1:19000/v1"
        assert result["agents"]["verifier"]["llm"]["api_key"] == "EMPTY"
        assert result["agents"]["searcher"]["llm"]["api_key"] == "EMPTY"
    finally:
        launcher.cleanup()


def test_launch_and_wait_success():
    launcher = MASLauncher()
    command = f"{shlex.quote(sys.executable)} -c \"print('hello')\""
    process = launcher.launch(command=command)
    exit_code = launcher.wait(process, timeout=10.0)
    assert exit_code == 0


def test_launch_and_wait_timeout_kills_process():
    launcher = MASLauncher()
    command = f"{shlex.quote(sys.executable)} -c \"import time; time.sleep(60)\""
    process = launcher.launch(command=command)
    exit_code = launcher.wait(process, timeout=0.3)
    assert exit_code != 0
    assert process.poll() is not None


def test_wait_timeout_kills_process_group_children(tmp_path):
    launcher = MASLauncher()
    child_pid_path = tmp_path / "child.pid"
    parent_code = (
        "import pathlib, subprocess, sys, time;"
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding='utf-8');"
        "time.sleep(60)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"
    process = launcher.launch(command=command)

    deadline = time.time() + 5.0
    while time.time() < deadline and not child_pid_path.exists():
        time.sleep(0.05)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())

    exit_code = launcher.wait(process, timeout=0.3)
    assert exit_code != 0

    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"child process {child_pid} still alive after timeout cleanup")


def test_prepare_config_write_error_cleans_temp_file(tmp_path, monkeypatch):
    launcher = MASLauncher(work_dir=tmp_path)
    before = {p.name for p in tmp_path.iterdir()}

    def _raise_dump_error(*_args, **_kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr("trajectory.launcher.yaml.safe_dump", _raise_dump_error)

    with pytest.raises(RuntimeError, match="write failed"):
        launcher.prepare_config(
            config_template=_make_template(),
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )

    after = {p.name for p in tmp_path.iterdir()}
    assert after == before


def test_cleanup_keeps_failed_paths_for_retry(tmp_path, monkeypatch):
    launcher = MASLauncher(work_dir=tmp_path)
    path_1 = launcher.prepare_config(
        config_template=_make_template(),
        monitor_url="http://127.0.0.1:19000/v1",
        agent_roles=["verifier", "searcher", "answerer"],
    )
    path_2 = launcher.prepare_config(
        config_template=_make_template(),
        monitor_url="http://127.0.0.1:19000/v1",
        agent_roles=["verifier", "searcher", "answerer"],
    )

    original_unlink = Path.unlink
    state = {"failed_once": False}

    def _flaky_unlink(self, missing_ok=False):
        if self == path_1 and not state["failed_once"]:
            state["failed_once"] = True
            raise OSError("busy")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)

    launcher.cleanup()
    assert path_1.exists()
    assert not path_2.exists()

    launcher.cleanup()
    assert not path_1.exists()
