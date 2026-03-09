import asyncio
import time
import sys
from pathlib import Path

import pytest

from trajectory.backend import InferenceBackend
from trajectory.datatypes import (
    EpisodeResult,
    EpisodeTrajectory,
    InteractionRecord,
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
    TurnData,
)
from trajectory import pipe as pipe_module
from trajectory.pipe import AgentPipe, AgentPipeConfig
from trajectory.reward import FunctionRewardProvider

pytestmark = pytest.mark.asyncio


class EchoBackend(InferenceBackend):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"echo:{request.agent_role}",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )


def _reward_fn(trajectory: EpisodeTrajectory) -> dict[str, object]:
    return {
        "agent_rewards": {role: 1.0 for role in trajectory.agent_trajectories},
        "final_reward": 1.0,
    }


def _make_config() -> AgentPipeConfig:
    return AgentPipeConfig(
        mas_command_template="echo ignored {config_path} {prompt}",
        config_template={
            "llm": {"base_url": "http://placeholder/v1"},
            "agents": {"verifier": {}},
        },
        model_mapping={"verifier": ModelMappingEntry()},
        timeout=5.0,
    )


async def test_agent_pipe_end_to_end(tmp_path: Path) -> None:
    script = tmp_path / "tiny_mas.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
base_url = config["llm"]["base_url"]
prompt = sys.argv[2]

for role in ("verifier", "answerer"):
    payload = json.dumps(
        {
            "model": role,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    _ = data["choices"][0]["message"]["content"]
""".strip(),
        encoding="utf-8",
    )

    config_template = {
        "llm": {
            "base_url": "http://placeholder/v1",
            "model": "default-model",
        },
        "agents": {
            "verifier": {"temperature": 0.2},
            "answerer": {"temperature": 0.4},
        },
    }
    config = AgentPipeConfig(
        mas_command_template=f"{sys.executable} {script} {{config_path}} {{prompt}}",
        config_template=config_template,
        model_mapping={
            "verifier": ModelMappingEntry(),
            "answerer": ModelMappingEntry(),
        },
        timeout=30.0,
    )
    pipe = AgentPipe(config=config, backend=EchoBackend())
    reward_provider = FunctionRewardProvider(_reward_fn)

    result = await pipe.run(prompt="What is 2 + 2 ?", reward_provider=reward_provider)

    assert isinstance(result, EpisodeResult)
    assert set(result.trajectory.agent_trajectories.keys()) == {"verifier", "answerer"}
    assert len(result.trajectory.agent_trajectories["verifier"]) == 1
    assert len(result.trajectory.agent_trajectories["answerer"]) == 1
    assert result.trajectory.agent_trajectories["verifier"][0].response_text == "echo:verifier"
    assert result.trajectory.agent_trajectories["answerer"][0].response_text == "echo:answerer"
    assert result.trajectory.agent_trajectories["verifier"][0].messages[0]["content"] == "What is 2 + 2 ?"
    assert result.rewards["verifier"] == 1.0
    assert result.rewards["answerer"] == 1.0
    assert result.final_reward == 1.0
    assert result.metadata["exit_code"] == 0


async def test_agent_pipe_run_raises_on_nonzero_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMonitor:
        def __init__(self, **_kwargs):
            self._buffer: list[InteractionRecord] = []

        async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
            return 19000

        async def stop(self) -> None:
            return None

        def get_buffer(self) -> list[InteractionRecord]:
            return self._buffer

    class FakeLauncher:
        def __init__(self, **_kwargs):
            pass

        def prepare_config(
            self,
            config_template: dict[str, object],
            monitor_url: str,
            agent_roles: list[str],
        ) -> Path:
            return Path("/tmp/fake.yaml")

        def launch(self, command: str) -> object:
            return object()

        def wait(self, process: object, timeout: float | None = None) -> int:
            return 9

        def cleanup(self) -> None:
            return None

    monkeypatch.setattr(pipe_module, "ModelMonitor", FakeMonitor)
    monkeypatch.setattr(pipe_module, "MASLauncher", FakeLauncher)

    pipe = AgentPipe(config=_make_config(), backend=EchoBackend())
    reward_provider = FunctionRewardProvider(_reward_fn)

    with pytest.raises(RuntimeError, match="exit code 9"):
        await pipe.run(prompt="irrelevant", reward_provider=reward_provider)


async def test_agent_pipe_run_offloads_reward_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMonitor:
        def __init__(self, **_kwargs):
            pass

        async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
            return 19001

        async def stop(self) -> None:
            return None

        def get_buffer(self) -> list[InteractionRecord]:
            return []

    class FakeLauncher:
        def __init__(self, **_kwargs):
            pass

        def prepare_config(
            self,
            config_template: dict[str, object],
            monitor_url: str,
            agent_roles: list[str],
        ) -> Path:
            return Path("/tmp/fake.yaml")

        def launch(self, command: str) -> object:
            return object()

        def wait(self, process: object, timeout: float | None = None) -> int:
            return 0

        def cleanup(self) -> None:
            return None

    class FakeCollector:
        def build(self, buffer: list[InteractionRecord], episode_id: str) -> EpisodeTrajectory:
            return EpisodeTrajectory(
                episode_id=episode_id,
                agent_trajectories={
                    "verifier": [
                        TurnData(
                            agent_role="verifier",
                            turn_index=0,
                            messages=[{"role": "user", "content": "hi"}],
                            response_text="echo:verifier",
                            token_ids=[1],
                            logprobs=[-0.1],
                            finish_reason="stop",
                            timestamp=1.0,
                            metadata={},
                        )
                    ]
                },
                metadata={},
            )

    monkeypatch.setattr(pipe_module, "ModelMonitor", FakeMonitor)
    monkeypatch.setattr(pipe_module, "MASLauncher", FakeLauncher)
    monkeypatch.setattr(pipe_module, "TrajectoryCollector", FakeCollector)

    def slow_reward(trajectory: EpisodeTrajectory) -> dict[str, object]:
        time.sleep(0.5)
        return {
            "agent_rewards": {role: 1.0 for role in trajectory.agent_trajectories},
            "final_reward": 1.0,
        }

    start = time.perf_counter()
    result1, result2 = await asyncio.gather(
        AgentPipe(config=_make_config(), backend=EchoBackend()).run(
            prompt="p1",
            reward_provider=FunctionRewardProvider(slow_reward),
        ),
        AgentPipe(config=_make_config(), backend=EchoBackend()).run(
            prompt="p2",
            reward_provider=FunctionRewardProvider(slow_reward),
        ),
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.9
    assert result1.final_reward == 1.0
    assert result2.final_reward == 1.0


async def test_agent_pipe_run_cleanup_executes_when_stop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"cleanup_called": False}

    class FakeMonitor:
        def __init__(self, **_kwargs):
            pass

        async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
            return 19002

        async def stop(self) -> None:
            raise RuntimeError("stop boom")

        def get_buffer(self) -> list[InteractionRecord]:
            return []

    class FakeLauncher:
        def __init__(self, **_kwargs):
            pass

        def prepare_config(
            self,
            config_template: dict[str, object],
            monitor_url: str,
            agent_roles: list[str],
        ) -> Path:
            return Path("/tmp/fake.yaml")

        def launch(self, command: str) -> object:
            return object()

        def wait(self, process: object, timeout: float | None = None) -> int:
            return 7

        def cleanup(self) -> None:
            state["cleanup_called"] = True

    monkeypatch.setattr(pipe_module, "ModelMonitor", FakeMonitor)
    monkeypatch.setattr(pipe_module, "MASLauncher", FakeLauncher)

    pipe = AgentPipe(config=_make_config(), backend=EchoBackend())
    with pytest.raises(RuntimeError, match="exit code 7") as exc_info:
        await pipe.run(
            prompt="irrelevant",
            reward_provider=FunctionRewardProvider(_reward_fn),
        )

    assert "stop boom" not in str(exc_info.value)
    assert state["cleanup_called"] is True
