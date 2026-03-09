from trajectory.datatypes import (
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
    InteractionRecord,
    TurnData,
    EpisodeTrajectory,
    EpisodeResult,
)


def test_model_request_creation():
    req = ModelRequest(
        request_id="r1",
        agent_role="verifier",
        messages=[{"role": "user", "content": "hello"}],
        generation_params={"temperature": 0.7},
    )
    assert req.agent_role == "verifier"
    assert req.request_id == "r1"


def test_model_response_with_none_tokens():
    resp = ModelResponse(content="hi", token_ids=None, logprobs=None, finish_reason="stop")
    assert resp.token_ids is None
    assert resp.finish_reason == "stop"


def test_interaction_record_fields():
    rec = InteractionRecord(
        agent_role="searcher",
        turn_index=0,
        timestamp=1000.0,
        messages=[{"role": "user", "content": "q"}],
        generation_params={"temperature": 0.6},
        response_text="answer",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
        episode_id="ep1",
        metadata={},
    )
    assert rec.agent_role == "searcher"
    assert len(rec.token_ids) == 3


def test_episode_trajectory_agent_grouping():
    t1 = TurnData(
        agent_role="verifier",
        turn_index=0,
        messages=[],
        response_text="no",
        token_ids=None,
        logprobs=None,
        finish_reason="stop",
        timestamp=1.0,
        metadata={},
    )
    t2 = TurnData(
        agent_role="searcher",
        turn_index=0,
        messages=[],
        response_text="query",
        token_ids=None,
        logprobs=None,
        finish_reason="stop",
        timestamp=2.0,
        metadata={},
    )
    traj = EpisodeTrajectory(
        episode_id="ep1",
        agent_trajectories={"verifier": [t1], "searcher": [t2]},
        metadata={},
    )
    assert set(traj.agent_trajectories.keys()) == {"verifier", "searcher"}


def test_episode_result_with_per_turn_rewards():
    traj = EpisodeTrajectory(episode_id="ep1", agent_trajectories={}, metadata={})
    result = EpisodeResult(
        trajectory=traj,
        rewards={"verifier": [0.5, 0.8], "searcher": 1.0},
        final_reward=1.0,
        metadata={},
    )
    assert isinstance(result.rewards["verifier"], list)
    assert isinstance(result.rewards["searcher"], float)
