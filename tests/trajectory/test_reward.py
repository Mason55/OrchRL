import pytest

from trajectory.datatypes import EpisodeResult, EpisodeTrajectory, TurnData
from trajectory.reward import FunctionRewardProvider, RewardWorker


def _make_trajectory(roles: list[str]) -> EpisodeTrajectory:
    agent_trajectories: dict[str, list[TurnData]] = {}
    for role in roles:
        agent_trajectories[role] = [
            TurnData(
                agent_role=role,
                turn_index=0,
                messages=[],
                response_text="out",
                token_ids=None,
                logprobs=None,
                finish_reason="stop",
                timestamp=1.0,
                metadata={},
            )
        ]
    return EpisodeTrajectory(episode_id="ep1", agent_trajectories=agent_trajectories)


def test_function_reward_provider_wraps_function_call() -> None:
    def reward_fn(traj: EpisodeTrajectory) -> dict[str, object]:
        return {
            "agent_rewards": {role: 1.0 for role in traj.agent_trajectories},
            "final_reward": 1.0,
        }

    provider = FunctionRewardProvider(reward_fn)
    traj = _make_trajectory(["verifier", "searcher"])

    result = provider.compute(traj)

    assert result["final_reward"] == 1.0
    assert result["agent_rewards"]["verifier"] == 1.0


def test_reward_worker_compute_returns_episode_result() -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {
            "agent_rewards": {"verifier": 0.5, "answerer": 1.0},
            "final_reward": 1.0,
        }

    worker = RewardWorker()
    traj = _make_trajectory(["verifier", "answerer"])

    result = worker.compute(traj, FunctionRewardProvider(reward_fn))

    assert isinstance(result, EpisodeResult)
    assert result.rewards["verifier"] == 0.5
    assert result.rewards["answerer"] == 1.0
    assert result.final_reward == 1.0
    assert result.trajectory is traj


def test_reward_worker_compute_supports_per_turn_rewards() -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {
            "agent_rewards": {"verifier": [0.3, 0.7]},
            "final_reward": 0.7,
        }

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    result = worker.compute(traj, FunctionRewardProvider(reward_fn))

    assert result.rewards["verifier"] == [0.3, 0.7]


def test_reward_worker_compute_raises_when_agent_rewards_missing() -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {"final_reward": 1.0}

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    with pytest.raises(ValueError, match="agent_rewards"):
        worker.compute(traj, FunctionRewardProvider(reward_fn))


def test_reward_worker_compute_raises_when_agent_rewards_value_type_invalid() -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {
            "agent_rewards": {"verifier": "bad"},
            "final_reward": 1.0,
        }

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    with pytest.raises(TypeError, match="agent_rewards"):
        worker.compute(traj, FunctionRewardProvider(reward_fn))


def test_reward_worker_compute_raises_when_final_reward_missing() -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {"agent_rewards": {"verifier": 0.5}}

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    with pytest.raises(ValueError, match="final_reward"):
        worker.compute(traj, FunctionRewardProvider(reward_fn))


def test_reward_worker_compute_wraps_provider_exceptions_with_context() -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        raise RuntimeError("boom")

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    with pytest.raises(RuntimeError, match="Reward provider failed for episode") as exc_info:
        worker.compute(traj, FunctionRewardProvider(reward_fn))
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"


@pytest.mark.parametrize("bad_scalar", [float("nan"), float("inf"), float("-inf")])
def test_reward_worker_rejects_non_finite_agent_reward_scalar(bad_scalar: float) -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {
            "agent_rewards": {"verifier": bad_scalar},
            "final_reward": 1.0,
        }

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    with pytest.raises(TypeError, match="agent_rewards"):
        worker.compute(traj, FunctionRewardProvider(reward_fn))


@pytest.mark.parametrize("bad_item", [float("nan"), float("inf"), float("-inf")])
def test_reward_worker_rejects_non_finite_agent_reward_list_item(bad_item: float) -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {
            "agent_rewards": {"verifier": [0.2, bad_item]},
            "final_reward": 1.0,
        }

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    with pytest.raises(TypeError, match="agent_rewards"):
        worker.compute(traj, FunctionRewardProvider(reward_fn))


@pytest.mark.parametrize("bad_final", [float("nan"), float("inf"), float("-inf")])
def test_reward_worker_rejects_non_finite_final_reward(bad_final: float) -> None:
    def reward_fn(_: EpisodeTrajectory) -> dict[str, object]:
        return {
            "agent_rewards": {"verifier": 0.5},
            "final_reward": bad_final,
        }

    worker = RewardWorker()
    traj = _make_trajectory(["verifier"])

    with pytest.raises(TypeError, match="final_reward"):
        worker.compute(traj, FunctionRewardProvider(reward_fn))
