"""AKG Kernel Gen — rollout collection example using MateRolloutAdapter.

This script wires together:
  - KernelBenchLoader   (prompt_loader)
  - AKGKernelRewardProvider  (reward_provider)
  - MateRolloutAdapter  (OrchRL's existing rollout engine)

Run:
    python examples/akg_kernel_gen/run_akg_rollout.py \
        --kernelbench-dir /path/to/KernelBench \
        --vllm-url http://localhost:8000/v1 \
        --policy-name kernel_gen \
        --steps 10

Prerequisites:
  1. vLLM server running the model to be trained
  2. KernelBench dataset downloaded (see README)
  3. CUDA GPU available for KernelVerifier
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from orchrl.trainer.mate_rollout_adapter import MateRolloutAdapter
from examples.akg_kernel_gen.orchrl_glue.kernelbench_loader import KernelBenchLoader
from examples.akg_kernel_gen.orchrl_glue.akg_kernel_reward import AKGKernelRewardProvider

logger = logging.getLogger(__name__)


def _build_verifier_factory(config_path: str):
    """Returns a callable that creates a KernelVerifier instance.

    The factory is called once per reward computation, so the verifier
    is stateless and safe to reuse across episodes.
    """
    import uuid
    from pathlib import Path as _Path
    from akg_agents.op.verifier.kernel_verifier import KernelVerifier
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.core.worker.manager import get_worker_manager

    resolved = _Path(config_path)
    if resolved.exists():
        base_config = load_config(config_path=str(resolved))
    else:
        base_config = load_config(dsl="triton_cuda", workflow="default")
    base_config["log_dir"] = base_config.get("log_dir", "/tmp/akg_rl_verifier_logs")

    def factory():
        wm = get_worker_manager()
        worker = wm.get_worker(backend="cuda", arch="a100")
        return KernelVerifier(
            op_name="reward_check",
            framework_code="",   # set per-call via verifier.framework_code
            task_id=uuid.uuid4().hex[:8],
            framework="torch",
            dsl="triton_cuda",
            backend="cuda",
            arch="a100",
            config=base_config,
            worker=worker,
        )

    return factory


def build_adapter(
    kernelbench_dir: str,
    designer_vllm_url: str,
    gen_vllm_url: str,
    designer_model: str,
    gen_model: str,
    policy_designer: str,
    policy_gen: str,
    config_template_path: str,
    akg_config_path: str,
) -> MateRolloutAdapter:
    """Assemble MateRolloutAdapter from AKG components."""
    with open(config_template_path, encoding="utf-8") as f:
        subprocess_template = yaml.safe_load(f)

    max_iterations = subprocess_template.get("task", {}).get("max_iterations", 5)

    config = {
        "mas_command_template": (
            "python examples/akg_kernel_gen/mas_entry/akg_rl_entry.py"
            " --config {config_path} --task {prompt}"
        ),
        "roles": ["kernel_designer", "kernel_gen"],
        "config_template": subprocess_template,
        "batch_size": 8,
        "n_samples_per_prompt": 4,
        "max_concurrent_episodes": 8,
        "timeout": 300,
    }

    loader = KernelBenchLoader(kernelbench_dir, level="level1", shuffle=True, seed=42)
    reward_provider = AKGKernelRewardProvider(
        alpha=1.0,
        beta=0.3,
        gamma=0.1,
        enable_profiling=False,
        max_turns=max_iterations,
        verifier_factory=_build_verifier_factory(akg_config_path),
    )

    return MateRolloutAdapter(
        config=config,
        prompt_loader=loader,
        reward_provider=reward_provider,
        server_address_dict={
            policy_designer: designer_vllm_url,
            policy_gen: gen_vllm_url,
        },
        role_policy_mapping={
            "kernel_designer": policy_designer,
            "kernel_gen": policy_gen,
        },
        policy_server_name_mapping={
            policy_designer: designer_model,
            policy_gen: gen_model,
        },
    )


async def _run_steps(adapter: MateRolloutAdapter, n_steps: int) -> None:
    total_episodes = 0
    total_reward = 0.0
    for step in range(n_steps):
        episodes = await adapter.collect_step_rollouts(step_idx=step)
        rewards = [ep.final_reward for ep in episodes if ep.final_reward is not None]
        step_mean = sum(rewards) / len(rewards) if rewards else 0.0
        total_episodes += len(episodes)
        total_reward += sum(rewards)
        logger.info(
            "step=%d episodes=%d mean_reward=%.3f",
            step, len(episodes), step_mean,
        )
    logger.info(
        "Done. total_episodes=%d overall_mean_reward=%.3f",
        total_episodes,
        total_reward / total_episodes if total_episodes else 0.0,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="AKG Kernel Gen rollout collection")
    parser.add_argument("--kernelbench-dir", required=True,
                        help="Path to KernelBench root (contains level1/ etc.)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1",
                        help="vLLM server URL for the training model")
    parser.add_argument("--policy-name", default="kernel_gen",
                        help="Policy name registered in OrchRL trainer")
    parser.add_argument("--designer-vllm-url", default="http://localhost:9000/v1",
                        help="vLLM server URL for kernel_designer")
    parser.add_argument("--designer-model", default="Qwen/Qwen2.5-Coder-7B",
                        help="Model name for kernel_designer")
    parser.add_argument("--gen-vllm-url", default="http://localhost:8000/v1",
                        help="vLLM server URL for kernel_gen")
    parser.add_argument("--gen-model", default="Qwen/Qwen2.5-Coder-7B",
                        help="Model name for kernel_gen")
    parser.add_argument("--policy-designer", default="kernel_designer",
                        help="Policy name for designer")
    parser.add_argument("--policy-gen", default="kernel_gen",
                        help="Policy name for gen")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of rollout steps to collect")
    parser.add_argument(
        "--config-template",
        default=str(Path(__file__).parent / "configs" / "akg_config_template.yaml"),
    )
    parser.add_argument(
        "--akg-config",
        default="python/akg_agents/op/config/triton_cuda_config.yaml",
        help="AKG LangGraphTask config path (relative to akg_agents root)",
    )
    args = parser.parse_args()

    adapter = build_adapter(
        kernelbench_dir=args.kernelbench_dir,
        designer_vllm_url=args.designer_vllm_url,
        gen_vllm_url=args.gen_vllm_url,
        designer_model=args.designer_model,
        gen_model=args.gen_model,
        policy_designer=args.policy_designer,
        policy_gen=args.policy_gen,
        config_template_path=args.config_template,
        akg_config_path=args.akg_config,
    )
    asyncio.run(_run_steps(adapter, args.steps))


if __name__ == "__main__":
    main()
