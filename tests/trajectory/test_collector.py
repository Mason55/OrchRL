from trajectory.collector import TrajectoryCollector
from trajectory.datatypes import EpisodeTrajectory, InteractionRecord


def _record(
    *,
    agent_role: str,
    turn_index: int,
    response_text: str = "resp",
    token_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
) -> InteractionRecord:
    return InteractionRecord(
        agent_role=agent_role,
        turn_index=turn_index,
        timestamp=1234.5 + turn_index,
        messages=[{"role": "user", "content": f"msg-{agent_role}-{turn_index}"}],
        generation_params={"temperature": 0.1},
        response_text=response_text,
        token_ids=token_ids,
        logprobs=logprobs,
        finish_reason="stop",
        episode_id="buffer-ep",
        metadata={"source": "test"},
    )


def test_build_groups_by_agent_role():
    buffer = [
        _record(agent_role="searcher", turn_index=0),
        _record(agent_role="verifier", turn_index=0),
        _record(agent_role="searcher", turn_index=1),
    ]

    trajectory = TrajectoryCollector().build(buffer=buffer, episode_id="ep-1")

    assert isinstance(trajectory, EpisodeTrajectory)
    assert trajectory.episode_id == "ep-1"
    assert set(trajectory.agent_trajectories.keys()) == {"searcher", "verifier"}
    assert len(trajectory.agent_trajectories["searcher"]) == 2
    assert len(trajectory.agent_trajectories["verifier"]) == 1


def test_build_sorts_each_group_by_turn_index():
    buffer = [
        _record(agent_role="searcher", turn_index=2),
        _record(agent_role="searcher", turn_index=0),
        _record(agent_role="searcher", turn_index=1),
        _record(agent_role="verifier", turn_index=1),
        _record(agent_role="verifier", turn_index=0),
    ]

    trajectory = TrajectoryCollector().build(buffer=buffer, episode_id="ep-2")

    assert [turn.turn_index for turn in trajectory.agent_trajectories["searcher"]] == [0, 1, 2]
    assert [turn.turn_index for turn in trajectory.agent_trajectories["verifier"]] == [0, 1]


def test_build_handles_empty_buffer():
    trajectory = TrajectoryCollector().build(buffer=[], episode_id="ep-empty")

    assert isinstance(trajectory, EpisodeTrajectory)
    assert trajectory.episode_id == "ep-empty"
    assert trajectory.agent_trajectories == {}


def test_build_preserves_record_fields():
    record = _record(
        agent_role="answerer",
        turn_index=3,
        response_text="final answer",
        token_ids=[11, 22, 33],
        logprobs=[-0.11, -0.22, -0.33],
    )
    trajectory = TrajectoryCollector().build(buffer=[record], episode_id="ep-fidelity")
    turn = trajectory.agent_trajectories["answerer"][0]

    assert turn.agent_role == "answerer"
    assert turn.turn_index == 3
    assert turn.messages == record.messages
    assert turn.response_text == "final answer"
    assert turn.token_ids == [11, 22, 33]
    assert turn.logprobs == [-0.11, -0.22, -0.33]
    assert turn.finish_reason == "stop"
    assert turn.timestamp == record.timestamp
    assert turn.metadata == {
        "source": "test",
        "episode_id": "ep-fidelity",
        "agent_role": "answerer",
        "turn_index": 3,
        "timestamp": record.timestamp,
    }
    assert record.metadata == {"source": "test"}
