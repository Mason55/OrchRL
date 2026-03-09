#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trajectory import (
    AgentPipeConfig,
    FunctionRewardProvider,
    ModelMappingEntry,
    VLLMBackend,
    parallel_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect trajectories from the OrchRL Search MAS.")
    parser.add_argument("--vllm-url", required=True, help="OpenAI-compatible vLLM endpoint, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="Model name served by the vLLM endpoint.")
    parser.add_argument("--mas-dir", type=Path, required=True, help="Path to OrchRL/examples/mas_app/search.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the Search MAS YAML config.")
    parser.add_argument("--prompt", default="who got the first nobel prize in physics?")
    parser.add_argument("--expected-answer", default=None, help="Optional golden answer used for 0/1 reward.")
    parser.add_argument("--substring-match", action="store_true", help="Use substring EM when --expected-answer is provided.")
    parser.add_argument("--n-samples", type=int, default=2, help="Episodes to collect for the prompt.")
    parser.add_argument("--output", type=Path, default=Path("trajectory_output.json"))
    return parser.parse_args()


def build_reward_provider(
    mas_dir: Path,
    expected_answer: str | None,
    use_substring: bool,
) -> FunctionRewardProvider:
    mas_dir = mas_dir.resolve()
    if str(mas_dir) not in sys.path:
        sys.path.insert(0, str(mas_dir))

    from search_mas.apps.search.evaluator import extract_answer, is_search_answer_correct

    def reward_fn(trajectory):
        answer_turns = trajectory.agent_trajectories.get("answerer", [])
        final_text = answer_turns[-1].response_text if answer_turns else ""
        prediction = extract_answer(final_text) or final_text.strip() or None

        if expected_answer is not None:
            final_reward = 1.0 if is_search_answer_correct(
                prediction,
                expected_answer,
                use_substring=use_substring,
            ) else 0.0
        else:
            final_reward = 1.0 if prediction else 0.0

        agent_rewards = {
            role: final_reward if role == "answerer" else 0.0
            for role in trajectory.agent_trajectories
        }
        return {
            "agent_rewards": agent_rewards,
            "final_reward": final_reward,
        }

    return FunctionRewardProvider(reward_fn)


async def main() -> None:
    args = parse_args()
    config_template = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    roles = list(config_template.get("agents", {}).keys()) or ["verifier", "searcher", "answerer"]

    backend = VLLMBackend(backend_url=args.vllm_url, actual_model=args.model)
    pipe_config = AgentPipeConfig(
        mas_command_template=(
            f"{sys.executable} scripts/run_search_mas.py "
            "--config {config_path} --question {prompt}"
        ),
        config_template=config_template,
        model_mapping={role: ModelMappingEntry(actual_model=args.model) for role in roles},
        timeout=180.0,
        mas_work_dir=args.mas_dir,
    )

    results = await parallel_rollout(
        prompts=[args.prompt],
        reward_provider=build_reward_provider(
            args.mas_dir,
            expected_answer=args.expected_answer,
            use_substring=args.substring_match,
        ),
        config=pipe_config,
        backend=backend,
        n_samples_per_prompt=args.n_samples,
        max_concurrent=args.n_samples,
    )

    payload = [asdict(result) for result in results]
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Collected {len(results)} episodes -> {args.output}")
    for result in results:
        role_names = ",".join(result.trajectory.agent_trajectories.keys())
        print(
            f"episode={result.trajectory.episode_id[:8]} "
            f"reward={result.final_reward} roles={role_names}",
        )


if __name__ == "__main__":
    asyncio.run(main())
