"""Kernel Gen reward function for MATE integration.

Extracts code from the kernel_gen agent's last turn, attempts to compile it,
and returns a reward based on compilation success + content quality.

Reward formula:
    r_compile       = 1.0 if code compiles (valid Python AST), else 0.0
    r_quality_bonus = 0.2 if code contains triton/cuda/torch keywords, else 0.0
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_BLOCK_CLOSED_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_CODE_BLOCK_OPEN_RE = re.compile(r"```(?:python)?\n(.*)", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_code(response_text: str) -> str:
    text = _strip_thinking(response_text)
    match = _CODE_BLOCK_CLOSED_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _CODE_BLOCK_OPEN_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


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

    logger.info(
        "[kernel_gen_reward] response_len=%d code_len=%d compile=%s quality=%s reward=%.1f first100=%r",
        len(last_response), len(code), r_compile > 0, r_quality > 0, final_reward,
        code[:100] if code else "<empty>",
    )

    return {
        "agent_rewards": {role: final_reward for role in trajectory.agent_trajectories},
        "final_reward": final_reward,
    }
