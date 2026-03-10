"""AKG RL Entry Point — launched by MASLauncher as a subprocess.

Usage:
    python akg_rl_entry.py --config /tmp/trajectory_mas_xxx.yaml \
                           --task '{"op_name":"1_relu","task_desc":"..."}'

IMPORTANT: This script always exits 0.
OrchRL drops any episode with a non-zero exit code (before reward
computation), which would remove negative examples needed for GRPO.
Correctness is determined by AKGKernelRewardProvider, not by this script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _inject_agent_env_vars(config: dict[str, Any]) -> None:
    """Set AKG_AGENTS_* env vars to route LLM calls through ModelMonitor.

    akg_agents reads AKG_AGENTS_BASE_URL / AKG_AGENTS_API_KEY / AKG_AGENTS_MODEL_NAME
    in single-model mode and applies them to all model levels (complex/standard/fast).
    """
    agents_cfg = config.get("agents", {})
    gen_cfg = agents_cfg.get("kernel_gen", {})
    gen_llm = gen_cfg.get("llm", {})
    base_url = gen_llm.get("base_url", "")
    model_name = gen_cfg.get("model", "kernel_gen")

    if not base_url:
        logger.warning("[akg_rl_entry] kernel_gen base_url is empty; LLM calls will fail")
        return

    os.environ["AKG_AGENTS_BASE_URL"] = base_url
    os.environ["AKG_AGENTS_API_KEY"] = "dummy"
    os.environ["AKG_AGENTS_MODEL_NAME"] = model_name
    logger.info("[akg_rl_entry] LLM routed to %s (model=%s)", base_url, model_name)


def _build_task(
    op_name: str,
    task_desc: str,
    task_cfg: dict[str, Any],
    akg_config: dict[str, Any],
) -> Any:
    """Construct a LangGraphTask instance. Extracted for testability."""
    from akg_agents.op.langgraph_op.task import LangGraphTask

    task_id = uuid.uuid4().hex[:8]
    return LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id=task_id,
        backend=task_cfg.get("backend", "cuda"),
        arch=task_cfg.get("arch", "a100"),
        dsl=task_cfg.get("dsl", "triton_cuda"),
        framework=task_cfg.get("framework", "torch"),
        workflow="coder_only",
        config=akg_config,
    )


def run(config_path: str, task_json: str) -> int:
    """Core logic; separated from main() for testability. Always returns 0."""
    try:
        # 1. Load YAML config written by prepare_config()
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 2. Set AKG env vars → redirect LLM calls to ModelMonitor
        _inject_agent_env_vars(config)

        # 3. Parse task
        task_info = json.loads(task_json)
        op_name = task_info["op_name"]
        task_desc = task_info["task_desc"]
        task_cfg = config.get("task", {})

        # 4. Load akg_agents config (uses env vars set above for LLM routing)
        from akg_agents.op.config.config_validator import load_config
        dsl = task_cfg.get("dsl", "triton_cuda")
        backend = task_cfg.get("backend", "cuda")
        akg_config = load_config(dsl=dsl, backend=backend, workflow="coder_only")
        akg_config["log_dir"] = akg_config.get("log_dir", "/tmp/akg_rl_logs")
        akg_config["task_label"] = op_name
        akg_config["skip_kernel_gen"] = True   # no Skill system needed for RL

        # 5. Build and run LangGraphTask
        task = _build_task(op_name=op_name, task_desc=task_desc, task_cfg=task_cfg, akg_config=akg_config)
        op_name_out, success, final_state = asyncio.run(task.run())
        logger.info("[akg_rl_entry] op=%s success=%s", op_name_out, success)
    except Exception as exc:
        # Log but do NOT propagate — we must exit 0 for GRPO to see this episode
        logger.error("[akg_rl_entry] episode failed with %s: %s", type(exc).__name__, exc)

    return 0  # always


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="AKG RL subprocess entry point")
    parser.add_argument("--config", required=True, help="Path to YAML config written by prepare_config()")
    parser.add_argument("--task", required=True, help="JSON string: {op_name, task_desc}")
    args = parser.parse_args()
    sys.exit(run(config_path=args.config, task_json=args.task))


if __name__ == "__main__":
    main()
