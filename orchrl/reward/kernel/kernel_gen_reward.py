"""Kernel Gen reward function for MATE integration.

Extracts code from the kernel_gen agent's last turn, attempts to compile it,
and returns a reward based on compilation success + content quality.

Reward formula:
    reward = r_compile + r_quality_bonus

    r_compile       = 1.0 if code compiles (valid Python AST), else 0.0
    r_quality_bonus = 0.2 if code contains triton/cuda/torch keywords, else 0.0
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(response_text: str) -> str:
    match = _CODE_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def _check_compile(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _check_has_kernel_content(code: str) -> bool:
    lowered = code.lower()
    return "triton" in lowered or "cuda" in lowered or "torch" in lowered


def compute_reward(trajectory: Any) -> dict[str, Any]:
    kernel_gen_turns = trajectory.agent_trajectories.get("kernel_gen", [])

    if not kernel_gen_turns:
        return {
            "agent_rewards": {role: 0.0 for role in trajectory.agent_trajectories},
            "final_reward": 0.0,
        }

    last_response = getattr(kernel_gen_turns[-1], "response_text", "")
    code = _extract_code(last_response)

    r_compile = 1.0 if _check_compile(code) else 0.0
    r_quality = 0.2 if (r_compile and _check_has_kernel_content(code)) else 0.0
    final_reward = r_compile + r_quality

    return {
        "agent_rewards": {role: final_reward for role in trajectory.agent_trajectories},
        "final_reward": final_reward,
    }
