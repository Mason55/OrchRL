import asyncio
from collections import Counter
from pathlib import Path

import pytest

from trajectory import parallel as parallel_module
from trajectory import pipe as pipe_module
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
from trajectory.parallel import parallel_rollout
from trajectory.pipe import AgentPipeConfig
from trajectory.reward import FunctionRewardProvider

pytestmark = pytest.mark.asyncio


class NoopBackend(InferenceBackend):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"noop:{request.agent_role}",
            token_ids=[1],
            logprobs=[-0.1],
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
        monitor_port=0,
    )


async def test_parallel_rollout_unique_episode_ids_and_monitor_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"next_port": 25000, "ports": []}

    class FakeMonitor:
        def __init__(self, **kwargs: object) -> None:
            self._episode_id = str(kwargs["episode_id"])
            self._buffer: list[InteractionRecord] = []

        async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
            assert host == "127.0.0.1"
            assert port == 0
            assigned = state["next_port"]
            state["next_port"] += 1
            state["ports"].append(assigned)
            return assigned

        async def stop(self) -> None:
            return None

        def get_buffer(self) -> list[InteractionRecord]:
            return self._buffer

    class FakeLauncher:
        def __init__(self, **_kwargs: object) -> None:
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
                            messages=[{"role": "user", "content": "p"}],
                            response_text="ok",
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

    class FakeRewardWorker:
        def compute(
            self,
            trajectory: EpisodeTrajectory,
            reward_provider: object,
        ) -> EpisodeResult:
            return EpisodeResult(
                trajectory=trajectory,
                rewards={"verifier": 1.0},
                final_reward=1.0,
                metadata={},
            )

    monkeypatch.setattr(pipe_module, "ModelMonitor", FakeMonitor)
    monkeypatch.setattr(pipe_module, "MASLauncher", FakeLauncher)
    monkeypatch.setattr(pipe_module, "TrajectoryCollector", FakeCollector)
    monkeypatch.setattr(pipe_module, "RewardWorker", FakeRewardWorker)

    results = await parallel_rollout(
        prompts=["same-prompt"],
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=NoopBackend(),
        n_samples_per_prompt=5,
    )

    episode_ids = [result.trajectory.episode_id for result in results]
    assert len(episode_ids) == 5
    assert len(set(episode_ids)) == 5
    assert len(state["ports"]) == 5
    assert len(set(state["ports"])) == 5


async def test_parallel_rollout_respects_max_concurrent(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"active": 0, "max_active": 0, "idx": 0}

    class FakeAgentPipe:
        def __init__(self, config: AgentPipeConfig, backend: InferenceBackend) -> None:
            pass

        async def run(self, prompt: str, reward_provider: object) -> EpisodeResult:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            try:
                await asyncio.sleep(0.05)
                state["idx"] += 1
                ep_id = f"ep-{state['idx']}"
                return EpisodeResult(
                    trajectory=EpisodeTrajectory(
                        episode_id=ep_id,
                        agent_trajectories={},
                        metadata={},
                    ),
                    rewards={},
                    final_reward=0.0,
                    metadata={"prompt": prompt},
                )
            finally:
                state["active"] -= 1

    monkeypatch.setattr(parallel_module, "AgentPipe", FakeAgentPipe)

    results = await parallel_rollout(
        prompts=["p1", "p2", "p3"],
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=NoopBackend(),
        n_samples_per_prompt=2,
        max_concurrent=2,
    )

    assert len(results) == 6
    assert state["max_active"] == 2


async def test_parallel_rollout_supports_prompt_batching(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgentPipe:
        def __init__(self, config: AgentPipeConfig, backend: InferenceBackend) -> None:
            self._counter = 0

        async def run(self, prompt: str, reward_provider: object) -> EpisodeResult:
            self._counter += 1
            return EpisodeResult(
                trajectory=EpisodeTrajectory(
                    episode_id=f"{prompt}-ep-{self._counter}",
                    agent_trajectories={},
                    metadata={},
                ),
                rewards={},
                final_reward=0.0,
                metadata={"prompt": prompt},
            )

    monkeypatch.setattr(parallel_module, "AgentPipe", FakeAgentPipe)

    results = await parallel_rollout(
        prompts=["q1", "q2", "q3"],
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=NoopBackend(),
        n_samples_per_prompt=3,
    )

    assert len(results) == 9
    prompt_counts = Counter(result.metadata["prompt"] for result in results)
    assert prompt_counts == {"q1": 3, "q2": 3, "q3": 3}


async def test_parallel_rollout_collects_success_when_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgentPipe:
        def __init__(self, config: AgentPipeConfig, backend: InferenceBackend) -> None:
            pass

        async def run(self, prompt: str, reward_provider: object) -> EpisodeResult:
            await asyncio.sleep(0.01)
            if prompt == "bad":
                raise RuntimeError("boom")
            return EpisodeResult(
                trajectory=EpisodeTrajectory(
                    episode_id=f"{prompt}-ok",
                    agent_trajectories={},
                    metadata={},
                ),
                rewards={},
                final_reward=0.0,
                metadata={"prompt": prompt},
            )

    monkeypatch.setattr(parallel_module, "AgentPipe", FakeAgentPipe)

    results = await parallel_rollout(
        prompts=["ok-a", "bad", "ok-b"],
        reward_provider=FunctionRewardProvider(_reward_fn),
        config=_make_config(),
        backend=NoopBackend(),
        n_samples_per_prompt=1,
    )

    prompts = sorted(result.metadata["prompt"] for result in results)
    assert prompts == ["ok-a", "ok-b"]
