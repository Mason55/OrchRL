# AKG Kernel Gen Example Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `examples/akg_kernel_gen/` — an OrchRL example that trains AKG's KernelGen agent on KernelBench using GRPO, with AKG running as a MATE black-box subprocess.

**Architecture:** `KernelBenchLoader` feeds task descriptions to `MateRolloutAdapter`; `MASLauncher` launches `akg_rl_entry.py` as subprocess per episode; `ModelMonitor` intercepts all LLM calls and records token_ids; `AKGKernelRewardProvider` re-runs `KernelVerifier` on the final generated code to compute multi-objective reward. Zero changes to OrchRL core or AKG core.

**Tech Stack:** Python 3.10+, OrchRL (`trajectory`, `orchrl.trainer.mate_rollout_adapter`), AKG Agents (`akg_agents.op.langgraph_op.task.LangGraphTask`, `akg_agents.op.verifier.kernel_verifier.KernelVerifier`), KernelBench dataset, pytest, PyYAML.

---

## File Map

| File | Role |
|------|------|
| `examples/akg_kernel_gen/__init__.py` | Package marker |
| `examples/akg_kernel_gen/orchrl_glue/__init__.py` | Package marker |
| `examples/akg_kernel_gen/orchrl_glue/kernelbench_loader.py` | `KernelBenchLoader` — implements `prompt_loader` duck-type for `MateRolloutAdapter` |
| `examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py` | `AKGKernelRewardProvider` — implements `RewardProvider` Protocol |
| `examples/akg_kernel_gen/mas_entry/__init__.py` | Package marker |
| `examples/akg_kernel_gen/mas_entry/akg_rl_entry.py` | AKG subprocess entry point; always exits 0 |
| `examples/akg_kernel_gen/configs/akg_config_template.yaml` | MAS command template + task config injected by `prepare_config()` |
| `examples/akg_kernel_gen/run_akg_rollout.py` | Assembly script: wires loader + reward + `MateRolloutAdapter` |
| `examples/akg_kernel_gen/requirements.txt` | Extra deps beyond OrchRL |
| `examples/akg_kernel_gen/README.md` | Usage guide |
| `examples/akg_kernel_gen/tests/__init__.py` | Test package marker |
| `examples/akg_kernel_gen/tests/test_kernelbench_loader.py` | Unit tests for loader |
| `examples/akg_kernel_gen/tests/test_akg_kernel_reward.py` | Unit tests for reward provider |
| `examples/akg_kernel_gen/tests/test_akg_rl_entry.py` | Integration tests for entry script |

---

## Chunk 1: KernelBench Prompt Loader

**What:** `KernelBenchLoader` scans a KernelBench directory for `.py` task files, shuffles with fixed seed, and returns batches in the `{"prompt": json_str, "raw": {...}}` shape that `MateRolloutAdapter._collect_single_job()` expects.

**Files:**
- Create: `examples/akg_kernel_gen/orchrl_glue/__init__.py`
- Create: `examples/akg_kernel_gen/orchrl_glue/kernelbench_loader.py`
- Create: `examples/akg_kernel_gen/tests/__init__.py`
- Create: `examples/akg_kernel_gen/tests/test_kernelbench_loader.py`

### Task 1.1: Scaffold directories and write failing tests

- [ ] Create the directory tree:

```bash
mkdir -p examples/akg_kernel_gen/orchrl_glue
mkdir -p examples/akg_kernel_gen/mas_entry
mkdir -p examples/akg_kernel_gen/configs
mkdir -p examples/akg_kernel_gen/tests
touch examples/akg_kernel_gen/__init__.py
touch examples/akg_kernel_gen/orchrl_glue/__init__.py
touch examples/akg_kernel_gen/mas_entry/__init__.py
touch examples/akg_kernel_gen/tests/__init__.py
```

- [ ] Write `examples/akg_kernel_gen/tests/test_kernelbench_loader.py`:

```python
"""Unit tests for KernelBenchLoader — no GPU, no AKG imports required."""
import json
import os
import textwrap
import pytest
from pathlib import Path

from examples.akg_kernel_gen.orchrl_glue.kernelbench_loader import KernelBenchLoader


@pytest.fixture()
def dataset_dir(tmp_path):
    """Minimal fake KernelBench level1 directory with 5 tasks."""
    level_dir = tmp_path / "level1"
    level_dir.mkdir()
    for i in range(1, 6):
        task_file = level_dir / f"{i}_op_{i}.py"
        task_file.write_text(
            textwrap.dedent(f"""\
                import torch
                import torch.nn as nn

                class Model(nn.Module):
                    def forward(self, x):
                        return x + {i}

                def get_inputs():
                    return [torch.randn(4, 4)]

                def get_init_inputs():
                    return []
            """)
        )
    return str(tmp_path)


def test_get_step_batch_returns_list_of_dicts(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch = loader.get_step_batch(step_idx=0, batch_size=3)
    assert isinstance(batch, list)
    assert len(batch) == 3
    for item in batch:
        assert "prompt" in item
        assert "raw" in item


def test_prompt_is_valid_json_with_required_keys(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch = loader.get_step_batch(step_idx=0, batch_size=1)
    parsed = json.loads(batch[0]["prompt"])
    assert "op_name" in parsed
    assert "task_desc" in parsed
    assert isinstance(parsed["op_name"], str)
    assert isinstance(parsed["task_desc"], str)
    assert "class Model" in parsed["task_desc"]


def test_raw_contains_op_name_and_level(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch = loader.get_step_batch(step_idx=0, batch_size=1)
    raw = batch[0]["raw"]
    assert "op_name" in raw
    assert "level" in raw
    assert raw["level"] == "level1"


def test_batch_size_respected(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    for batch_size in [1, 3, 5]:
        batch = loader.get_step_batch(step_idx=0, batch_size=batch_size)
        assert len(batch) == batch_size


def test_batch_cycles_when_step_exceeds_dataset(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    # 5 tasks, batch_size=3: step 0 → [0,1,2], step 1 → [3,4,0]
    batch0 = loader.get_step_batch(step_idx=0, batch_size=3)
    batch1 = loader.get_step_batch(step_idx=1, batch_size=3)
    # All prompts should be valid JSON
    for item in batch0 + batch1:
        json.loads(item["prompt"])


def test_reproducible_order_with_same_seed(dataset_dir):
    loader_a = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=42)
    loader_b = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=42)
    batch_a = loader_a.get_step_batch(step_idx=0, batch_size=5)
    batch_b = loader_b.get_step_batch(step_idx=0, batch_size=5)
    assert [json.loads(x["prompt"])["op_name"] for x in batch_a] == \
           [json.loads(x["prompt"])["op_name"] for x in batch_b]


def test_different_seeds_give_different_order(dataset_dir):
    loader_a = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=42)
    loader_b = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=99)
    names_a = [json.loads(x["prompt"])["op_name"]
               for x in loader_a.get_step_batch(0, 5)]
    names_b = [json.loads(x["prompt"])["op_name"]
               for x in loader_b.get_step_batch(0, 5)]
    # Very unlikely to be equal with 5 elements
    assert names_a != names_b


def test_raises_if_dataset_dir_missing():
    with pytest.raises((FileNotFoundError, ValueError)):
        KernelBenchLoader("/nonexistent/path/dataset", level="level1")
```

- [ ] Run tests to confirm they all fail (module not found):

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -m pytest examples/akg_kernel_gen/tests/test_kernelbench_loader.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'examples.akg_kernel_gen.orchrl_glue.kernelbench_loader'`

### Task 1.2: Implement KernelBenchLoader

- [ ] Write `examples/akg_kernel_gen/orchrl_glue/kernelbench_loader.py`:

```python
"""KernelBench prompt loader for MateRolloutAdapter.

Implements the duck-typed prompt_loader interface:
    get_step_batch(step_idx: int, batch_size: int) -> list[dict]

Each returned dict has shape:
    {
        "prompt": json.dumps({"op_name": str, "task_desc": str}),
        "raw":    {"op_name": str, "task_desc": str, "level": str},
    }
"""
from __future__ import annotations

import json
import random
from pathlib import Path


class KernelBenchLoader:
    """Loads KernelBench tasks and serves them as rolling batches.

    Args:
        dataset_dir: Root directory containing ``level1/``, ``level2/`` etc.
        level:       Sub-directory to load (default ``"level1"``).
        shuffle:     Whether to shuffle tasks before serving (default ``True``).
        seed:        Random seed for reproducible shuffling (default ``42``).
    """

    def __init__(
        self,
        dataset_dir: str,
        level: str = "level1",
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        level_path = Path(dataset_dir) / level
        if not level_path.exists():
            raise FileNotFoundError(
                f"KernelBench level directory not found: {level_path}"
            )

        task_files = sorted(level_path.glob("*.py"))
        if not task_files:
            raise ValueError(f"No .py task files found in {level_path}")

        self._tasks: list[dict] = []
        for f in task_files:
            op_name = f.stem  # e.g. "1_element_wise_addition"
            task_desc = f.read_text(encoding="utf-8")
            self._tasks.append(
                {"op_name": op_name, "task_desc": task_desc, "level": level}
            )

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._tasks)

    def get_step_batch(self, step_idx: int, batch_size: int) -> list[dict]:
        """Return *batch_size* tasks for the given training step.

        Cycles through the dataset using ``step_idx * batch_size`` as the
        starting offset, wrapping around as needed.
        """
        n = len(self._tasks)
        start = (step_idx * batch_size) % n
        # Build indices with wrap-around
        indices = [(start + i) % n for i in range(batch_size)]
        result = []
        for idx in indices:
            task = self._tasks[idx]
            result.append(
                {
                    "prompt": json.dumps(
                        {
                            "op_name": task["op_name"],
                            "task_desc": task["task_desc"],
                        }
                    ),
                    "raw": {
                        "op_name": task["op_name"],
                        "task_desc": task["task_desc"],
                        "level": task["level"],
                    },
                }
            )
        return result
```

- [ ] Run tests and confirm all pass:

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -m pytest examples/akg_kernel_gen/tests/test_kernelbench_loader.py -v
```

Expected: `7 passed`

- [ ] Commit:

```bash
git add examples/akg_kernel_gen/
git commit -m "feat(examples): add KernelBenchLoader for AKG rollout example"
```

---

## Chunk 2: AKG Kernel Reward Provider

**What:** `AKGKernelRewardProvider` implements OrchRL's `RewardProvider` Protocol. It extracts the final generated code from `trajectory.agent_trajectories["kernel_gen"][-1].response_text`, re-runs `KernelVerifier` for correctness, and returns a multi-objective reward. `verifier_factory` is injected so tests can mock it without a GPU.

**Files:**
- Create: `examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py`
- Create: `examples/akg_kernel_gen/tests/test_akg_kernel_reward.py`

### Task 2.1: Write failing reward tests

- [ ] Write `examples/akg_kernel_gen/tests/test_akg_kernel_reward.py`:

```python
"""Unit tests for AKGKernelRewardProvider — no GPU required (verifier is mocked)."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from trajectory.datatypes import EpisodeTrajectory, TurnData

from examples.akg_kernel_gen.orchrl_glue.akg_kernel_reward import AKGKernelRewardProvider


def _make_turn(response_text: str, turn_index: int = 0) -> TurnData:
    return TurnData(
        agent_role="kernel_gen",
        turn_index=turn_index,
        messages=[],
        response_text=response_text,
        token_ids=[1, 2, 3],
        logprobs=None,
        finish_reason="stop",
        timestamp=0.0,
    )


def _make_trajectory(
    kernel_gen_turns: list[TurnData],
    extra_roles: list[str] | None = None,
) -> EpisodeTrajectory:
    agent_trajectories: dict = {"kernel_gen": kernel_gen_turns}
    for role in (extra_roles or []):
        agent_trajectories[role] = [_make_turn("some output", 0)]
    return EpisodeTrajectory(
        episode_id="test-episode",
        agent_trajectories=agent_trajectories,
    )


def _mock_verifier_factory(success: bool):
    """Returns a factory callable that yields a mock KernelVerifier."""
    verifier = MagicMock()
    verifier.run = AsyncMock(return_value=(success, "mock log"))
    def factory():
        return verifier
    return factory


# ── correctness reward ────────────────────────────────────────────────────────

def test_returns_required_keys():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    code_block = "```python\ndef foo(): pass\n```"
    traj = _make_trajectory([_make_turn(code_block)])
    result = provider.compute(traj)
    assert "agent_rewards" in result
    assert "final_reward" in result


def test_correct_code_gives_nonzero_reward():
    provider = AKGKernelRewardProvider(
        alpha=1.0, beta=0.0, gamma=0.0,
        verifier_factory=_mock_verifier_factory(True),
    )
    traj = _make_trajectory([_make_turn("```python\npass\n```")])
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(1.0)


def test_incorrect_code_gives_zero_reward():
    provider = AKGKernelRewardProvider(
        alpha=1.0, beta=0.0, gamma=0.0,
        verifier_factory=_mock_verifier_factory(False),
    )
    traj = _make_trajectory([_make_turn("```python\npass\n```")])
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.0)


def test_empty_trajectory_gives_zero_reward():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    traj = _make_trajectory([])  # no kernel_gen turns
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.0)


# ── iter efficiency ───────────────────────────────────────────────────────────

def test_single_turn_maximises_iter_reward():
    """1 turn out of max_turns=5 → r_iter = 0.8."""
    provider = AKGKernelRewardProvider(
        alpha=0.0, beta=0.0, gamma=1.0, max_turns=5,
        verifier_factory=_mock_verifier_factory(False),
    )
    traj = _make_trajectory([_make_turn("code")])
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.8)


def test_max_turns_gives_zero_iter_reward():
    provider = AKGKernelRewardProvider(
        alpha=0.0, beta=0.0, gamma=1.0, max_turns=3,
        verifier_factory=_mock_verifier_factory(False),
    )
    turns = [_make_turn("code", i) for i in range(3)]
    traj = _make_trajectory(turns)
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.0)


# ── multi-role reward ────────────────────────────────────────────────────────

def test_all_roles_receive_same_final_reward():
    provider = AKGKernelRewardProvider(
        alpha=1.0, beta=0.0, gamma=0.0,
        verifier_factory=_mock_verifier_factory(True),
    )
    traj = _make_trajectory(
        [_make_turn("```python\npass\n```")],
        extra_roles=["kernel_designer"],
    )
    result = provider.compute(traj)
    rewards = result["agent_rewards"]
    assert set(rewards.keys()) == {"kernel_gen", "kernel_designer"}
    assert rewards["kernel_gen"] == rewards["kernel_designer"]


# ── code extraction ───────────────────────────────────────────────────────────

def test_extracts_code_from_markdown_block():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    response = "Here is the code:\n```python\nclass ModelNew: pass\n```\nDone."
    traj = _make_trajectory([_make_turn(response)])
    # Should not raise and should return a result
    result = provider.compute(traj)
    assert "final_reward" in result


def test_falls_back_to_raw_text_when_no_code_block():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    traj = _make_trajectory([_make_turn("class ModelNew: pass")])
    result = provider.compute(traj)
    assert "final_reward" in result
```

- [ ] Run to confirm failure:

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -m pytest examples/akg_kernel_gen/tests/test_akg_kernel_reward.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'examples.akg_kernel_gen.orchrl_glue.akg_kernel_reward'`

### Task 2.2: Implement AKGKernelRewardProvider

- [ ] Write `examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py`:

```python
"""AKG Kernel Reward Provider — implements OrchRL's RewardProvider Protocol.

Computes a multi-objective reward from an EpisodeTrajectory:
    reward = α · r_correct + β · r_perf · r_correct + γ · r_iter

Where:
  r_correct  — 1.0 if KernelVerifier passes, 0.0 otherwise
  r_perf     — speedup vs PyTorch reference (Phase 2, disabled by default)
  r_iter     — 1.0 - n_turns / max_turns  (efficiency bonus)

The verifier is injected via ``verifier_factory`` to allow mocking in tests.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable

from trajectory.datatypes import EpisodeTrajectory

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(response_text: str) -> str:
    """Extract code from a Markdown code block, or return the raw text."""
    match = _CODE_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


class AKGKernelRewardProvider:
    """Multi-objective reward provider for AKG kernel generation.

    Args:
        alpha:            Weight for correctness reward (default 1.0).
        beta:             Weight for performance reward (default 0.3).
        gamma:            Weight for iteration-efficiency reward (default 0.1).
        enable_profiling: Whether to run the profiler for r_perf (default False).
        max_turns:        Maximum expected turns per episode (default 5).
        verifier_factory: Callable[[] -> KernelVerifier]. Required in production;
                          can be replaced with a mock in tests.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        enable_profiling: bool = False,
        max_turns: int = 5,
        verifier_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.enable_profiling = enable_profiling
        self.max_turns = max_turns
        self._verifier_factory = verifier_factory

    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]:
        """Compute reward for one episode.

        Returns:
            {"agent_rewards": {role: float, ...}, "final_reward": float}
        """
        kernel_gen_turns = trajectory.agent_trajectories.get("kernel_gen", [])

        if not kernel_gen_turns:
            logger.warning(
                "[AKGKernelRewardProvider] No kernel_gen turns in trajectory %s",
                trajectory.episode_id,
            )
            final_reward = 0.0
            return {
                "agent_rewards": {
                    role: final_reward
                    for role in trajectory.agent_trajectories
                },
                "final_reward": final_reward,
            }

        # Extract generated code from the last turn
        last_response = kernel_gen_turns[-1].response_text
        code = _extract_code(last_response)

        # r_correct: re-run verifier independently of the subprocess
        r_correct = self._compute_correctness(code)

        # r_perf: speedup ratio (Phase 2 feature, off by default)
        r_perf = 0.0
        if r_correct and self.enable_profiling:
            r_perf = self._compute_performance(code)

        # r_iter: efficiency bonus
        n_turns = len(kernel_gen_turns)
        r_iter = max(0.0, 1.0 - n_turns / self.max_turns)

        final_reward = (
            self.alpha * r_correct
            + self.beta * r_perf * r_correct
            + self.gamma * r_iter
        )

        logger.debug(
            "[AKGKernelRewardProvider] episode=%s r_correct=%.2f r_perf=%.2f "
            "r_iter=%.2f final=%.3f",
            trajectory.episode_id, r_correct, r_perf, r_iter, final_reward,
        )

        return {
            "agent_rewards": {
                role: final_reward for role in trajectory.agent_trajectories
            },
            "final_reward": final_reward,
        }

    def _compute_correctness(self, code: str) -> float:
        if not code or self._verifier_factory is None:
            logger.warning(
                "[AKGKernelRewardProvider] No verifier_factory configured; "
                "returning r_correct=0.0"
            )
            return 0.0
        try:
            verifier = self._verifier_factory()
            loop = asyncio.new_event_loop()
            try:
                success, log = loop.run_until_complete(verifier.run(code))
            finally:
                loop.close()
            return 1.0 if success else 0.0
        except Exception as exc:
            logger.warning(
                "[AKGKernelRewardProvider] Verifier raised %s: %s", type(exc).__name__, exc
            )
            return 0.0

    def _compute_performance(self, code: str) -> float:
        """Phase 2: compute speedup ratio. Not implemented yet."""
        return 0.0
```

- [ ] Run tests and confirm all pass:

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -m pytest examples/akg_kernel_gen/tests/test_akg_kernel_reward.py -v
```

Expected: `10 passed`

- [ ] Commit:

```bash
git add examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py \
        examples/akg_kernel_gen/tests/test_akg_kernel_reward.py
git commit -m "feat(examples): add AKGKernelRewardProvider with multi-objective reward"
```

---

## Chunk 3: AKG RL Entry Script

**What:** `akg_rl_entry.py` is the subprocess launched by `MASLauncher`. It reads the YAML config written by `prepare_config()`, sets `AKG_AGENTS_BASE_URL` and `AKG_AGENTS_MODEL_NAME` env vars to redirect AKG's LLM calls to `ModelMonitor`, then runs `LangGraphTask`. It **always exits 0** so OrchRL never drops an episode — failed generations are handled by the reward provider.

**Files:**
- Create: `examples/akg_kernel_gen/mas_entry/akg_rl_entry.py`
- Create: `examples/akg_kernel_gen/tests/test_akg_rl_entry.py`

### Task 3.1: Write failing entry script tests

- [ ] Write `examples/akg_kernel_gen/tests/test_akg_rl_entry.py`:

```python
"""Integration tests for akg_rl_entry.py.

LangGraphTask is mocked so tests run without a GPU or LLM server.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


ENTRY_SCRIPT = str(
    Path(__file__).parent.parent / "mas_entry" / "akg_rl_entry.py"
)


def _write_config(tmp_path: Path, base_url: str = "http://127.0.0.1:9999/v1") -> Path:
    config = {
        "agents": {
            "kernel_gen": {
                "model": "kernel_gen",
                "llm": {"base_url": base_url, "api_key": "dummy"},
            },
            "kernel_designer": {
                "model": "kernel_designer",
                "llm": {"base_url": base_url, "api_key": "dummy"},
            },
        },
        "task": {
            "framework": "torch",
            "backend": "cuda",
            "arch": "a100",
            "dsl": "triton_cuda",
            "max_iterations": 2,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _task_arg(op_name: str = "1_relu", task_desc: str = "class Model: pass") -> str:
    return json.dumps({"op_name": op_name, "task_desc": task_desc})


# ── always exits 0 ────────────────────────────────────────────────────────────

def test_exits_zero_on_successful_task(tmp_path):
    config_path = _write_config(tmp_path)
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("1_relu", True, {}))

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        return_value=mock_task,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        code = akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert code == 0


def test_exits_zero_even_when_task_fails(tmp_path):
    config_path = _write_config(tmp_path)
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("1_relu", False, {}))

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        return_value=mock_task,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        code = akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert code == 0


def test_exits_zero_even_when_task_raises(tmp_path):
    config_path = _write_config(tmp_path)
    mock_task = MagicMock()
    mock_task.run = AsyncMock(side_effect=RuntimeError("GPU exploded"))

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        return_value=mock_task,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        code = akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert code == 0


# ── env var injection ─────────────────────────────────────────────────────────

def test_sets_akg_env_vars_from_config(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, base_url="http://monitor:1234/v1")
    captured_env = {}
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("1_relu", True, {}))

    def capture_and_build(*args, **kwargs):
        captured_env["base_url"] = os.environ.get("AKG_AGENTS_BASE_URL")
        captured_env["model_name"] = os.environ.get("AKG_AGENTS_MODEL_NAME")
        return mock_task

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        side_effect=capture_and_build,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert captured_env["base_url"] == "http://monitor:1234/v1"
    assert captured_env["model_name"] == "kernel_gen"


# ── task JSON parsing ─────────────────────────────────────────────────────────

def test_parses_task_json_and_passes_to_build_task(tmp_path):
    config_path = _write_config(tmp_path)
    received_kwargs: dict = {}
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("custom_op", True, {}))

    def capture_build(**kwargs):
        received_kwargs.update(kwargs)
        return mock_task

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        side_effect=capture_build,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        akg_rl_entry.run(
            config_path=str(config_path),
            task_json=json.dumps({"op_name": "my_relu", "task_desc": "class M: pass"}),
        )
    assert received_kwargs.get("op_name") == "my_relu"
    assert received_kwargs.get("task_desc") == "class M: pass"
```

- [ ] Run to confirm failure:

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -m pytest examples/akg_kernel_gen/tests/test_akg_rl_entry.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'examples.akg_kernel_gen.mas_entry.akg_rl_entry'`

### Task 3.2: Implement akg_rl_entry.py

- [ ] Write `examples/akg_kernel_gen/mas_entry/akg_rl_entry.py`:

```python
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
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _inject_akg_env_vars(config: dict[str, Any]) -> None:
    """Set AKG_AGENTS_* env vars so AKG's LLM client calls ModelMonitor."""
    agents_cfg = config.get("agents", {})
    kernel_gen_cfg = agents_cfg.get("kernel_gen", {})
    llm_cfg = kernel_gen_cfg.get("llm", {})
    base_url = llm_cfg.get("base_url", "")
    model_name = kernel_gen_cfg.get("model", "kernel_gen")

    if base_url:
        os.environ["AKG_AGENTS_BASE_URL"] = base_url
        os.environ["AKG_AGENTS_MODEL_NAME"] = model_name
        logger.info(
            "[akg_rl_entry] LLM redirected to ModelMonitor: %s (model=%s)",
            base_url, model_name,
        )
    else:
        logger.warning("[akg_rl_entry] base_url is empty; AKG will use its default LLM config.")


def _build_task(
    op_name: str,
    task_desc: str,
    task_cfg: dict[str, Any],
) -> Any:
    """Construct a LangGraphTask instance. Extracted for testability."""
    import uuid
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.op.config.config_validator import load_config

    task_id = uuid.uuid4().hex[:8]
    config = load_config()  # loads default config; base_url already set via env vars
    config["log_dir"] = config.get("log_dir", "/tmp/akg_rl_logs")

    return LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id=task_id,
        backend=task_cfg.get("backend", "cuda"),
        arch=task_cfg.get("arch", "a100"),
        dsl=task_cfg.get("dsl", "triton_cuda"),
        framework=task_cfg.get("framework", "torch"),
        config=config,
    )


def run(config_path: str, task_json: str) -> int:
    """Core logic; separated from main() for testability. Always returns 0."""
    # 1. Load YAML config written by prepare_config()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Set AKG env vars → redirect LLM calls to ModelMonitor
    _inject_akg_env_vars(config)

    # 3. Parse task
    task_info = json.loads(task_json)
    op_name = task_info["op_name"]
    task_desc = task_info["task_desc"]
    task_cfg = config.get("task", {})

    # 4. Build and run LangGraphTask
    try:
        task = _build_task(op_name=op_name, task_desc=task_desc, task_cfg=task_cfg)
        op_name_out, success, final_state = asyncio.run(task.run())
        logger.info(
            "[akg_rl_entry] op=%s success=%s", op_name_out, success
        )
    except Exception as exc:
        # Log but do NOT propagate — we must exit 0 for GRPO to see this episode
        logger.error("[akg_rl_entry] LangGraphTask raised %s: %s", type(exc).__name__, exc)

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
```

- [ ] Run tests and confirm all pass:

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -m pytest examples/akg_kernel_gen/tests/test_akg_rl_entry.py -v
```

Expected: `6 passed`

- [ ] Commit:

```bash
git add examples/akg_kernel_gen/mas_entry/akg_rl_entry.py \
        examples/akg_kernel_gen/tests/test_akg_rl_entry.py
git commit -m "feat(examples): add akg_rl_entry.py subprocess entry point (always exits 0)"
```

---

## Chunk 4: Config, Assembly Script, and README

**What:** Wire everything together. `akg_config_template.yaml` is the config template passed to `MateRolloutAdapter`; `run_akg_rollout.py` is the runnable assembly script; `README.md` documents usage and prerequisites.

**Files:**
- Create: `examples/akg_kernel_gen/configs/akg_config_template.yaml`
- Create: `examples/akg_kernel_gen/run_akg_rollout.py`
- Create: `examples/akg_kernel_gen/requirements.txt`
- Create: `examples/akg_kernel_gen/README.md`

### Task 4.1: Write config template

- [ ] Write `examples/akg_kernel_gen/configs/akg_config_template.yaml`:

```yaml
# AKG Kernel Gen — MateRolloutAdapter config template
# OrchRL's prepare_config() injects base_url into agents.*.llm.base_url at runtime.

mas_command_template: >-
  python examples/akg_kernel_gen/mas_entry/akg_rl_entry.py
  --config {config_path}
  --task {prompt}

# ── agent roles ────────────────────────────────────────────────────────────────
# model field = role name sent in LLM request → ModelMonitor routes by model name
agents:
  kernel_gen:
    model: kernel_gen        # the policy being trained
    llm:
      base_url: ""           # injected by prepare_config()
      api_key: "dummy"
  kernel_designer:
    model: kernel_designer   # frozen in Phase 1; add to training in Phase 2
    llm:
      base_url: ""           # injected by prepare_config()
      api_key: "dummy"

# ── task config (passed through to LangGraphTask) ─────────────────────────────
task:
  framework: torch
  backend: cuda
  arch: a100
  dsl: triton_cuda
  max_iterations: 5

# ── rollout sampling ───────────────────────────────────────────────────────────
batch_size: 8              # prompts per training step
n_samples_per_prompt: 4    # episodes per prompt (GRPO needs >1 for advantage computation)
max_concurrent_episodes: 8 # max parallel AKG subprocesses
timeout: 300               # seconds per episode before MASLauncher kills the subprocess
```

### Task 4.2: Write assembly script

- [ ] Write `examples/akg_kernel_gen/run_akg_rollout.py`:

```python
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
    from akg_agents.op.verifier.kernel_verifier import KernelVerifier
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.core.worker.manager import get_worker_manager

    base_config = load_config(config_path=config_path)
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
    vllm_url: str,
    policy_name: str,
    config_template_path: str,
    akg_config_path: str,
) -> MateRolloutAdapter:
    """Assemble MateRolloutAdapter from AKG components."""
    with open(config_template_path, encoding="utf-8") as f:
        config_template = yaml.safe_load(f)

    loader = KernelBenchLoader(kernelbench_dir, level="level1", shuffle=True, seed=42)
    reward_provider = AKGKernelRewardProvider(
        alpha=1.0,
        beta=0.3,
        gamma=0.1,
        enable_profiling=False,
        max_turns=config_template.get("task", {}).get("max_iterations", 5),
        verifier_factory=_build_verifier_factory(akg_config_path),
    )

    return MateRolloutAdapter(
        config=config_template,
        prompt_loader=loader,
        reward_provider=reward_provider,
        server_address_dict={policy_name: vllm_url},
        role_policy_mapping={"kernel_gen": policy_name, "kernel_designer": policy_name},
        policy_server_name_mapping={policy_name: "kernel_gen"},
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
        vllm_url=args.vllm_url,
        policy_name=args.policy_name,
        config_template_path=args.config_template,
        akg_config_path=args.akg_config,
    )
    asyncio.run(_run_steps(adapter, args.steps))


if __name__ == "__main__":
    main()
```

### Task 4.3: Write requirements.txt and README

- [ ] Write `examples/akg_kernel_gen/requirements.txt`:

```text
# AKG Agents (install from local source)
# pip install -e /path/to/akg/akg_agents --no-build-isolation
akg-agents>=2.0.0

# OrchRL trajectory library (install from local source)
# pip install -e /path/to/OrchRL

pyyaml>=6.0
```

- [ ] Write `examples/akg_kernel_gen/README.md`:

````markdown
# AKG Kernel Gen — OrchRL Example

Train AKG's `KernelGen` agent on [KernelBench](https://github.com/KernelBench/KernelBench)
using OrchRL's GRPO pipeline. AKG runs as a MATE black-box subprocess;
all LLM calls are intercepted by `ModelMonitor` and `token_ids` are recorded for gradient computation.

## Prerequisites

1. **vLLM server** with the model to train (e.g. `Qwen2.5-Coder-7B`):
   ```bash
   vllm serve Qwen/Qwen2.5-Coder-7B --port 8000
   ```

2. **KernelBench dataset** (PyTorch, Level 1):
   ```bash
   cd /path/to/akg/akg_agents
   git submodule update --init "thirdparty/KernelBench"
   ```
   Tasks are at `akg_agents/thirdparty/KernelBench/KernelBench/level1/`.

3. **Register a CUDA worker** with AKG's WorkerManager before running:
   ```python
   from akg_agents.core.worker.manager import register_local_worker
   import asyncio
   asyncio.run(register_local_worker([0], backend="cuda", arch="a100"))
   ```

4. **Install dependencies**:
   ```bash
   pip install -e /path/to/akg/akg_agents --no-build-isolation
   pip install -e /path/to/OrchRL
   ```

## Quickstart

```bash
cd /path/to/OrchRL

python examples/akg_kernel_gen/run_akg_rollout.py \
    --kernelbench-dir /path/to/akg/akg_agents/thirdparty/KernelBench/KernelBench \
    --vllm-url http://localhost:8000/v1 \
    --policy-name kernel_gen \
    --steps 10
```

## Architecture

```
MateRolloutAdapter
  └─ per episode: AgentPipe.run()
       ├─ ModelMonitor (aiohttp proxy, intercepts LLM calls)
       └─ MASLauncher → akg_rl_entry.py --config /tmp/xxx.yaml --task '...'
             ├─ KernelDesigner → LLM (model=kernel_designer) → ModelMonitor
             ├─ KernelGen      → LLM (model=kernel_gen)      → ModelMonitor
             └─ KernelVerifier → local GPU (no LLM)
  └─ AKGKernelRewardProvider.compute(trajectory)
       ├─ extract final code from kernel_gen[-1].response_text
       ├─ re-run KernelVerifier → r_correct
       └─ reward = α·r_correct + β·r_perf·r_correct + γ·r_iter
```

**Why `akg_rl_entry.py` always exits 0:**
OrchRL drops any episode with non-zero exit code before reward computation,
removing the negative examples GRPO needs. Correctness is measured by
`AKGKernelRewardProvider`, not by the subprocess exit code.

## Reward Formula

```
reward = α·r_correct + β·r_perf·r_correct + γ·r_iter

r_correct  = 1.0 if KernelVerifier passes, else 0.0
r_perf     = speedup vs PyTorch reference (Phase 2, disabled by default)
r_iter     = 1.0 - n_turns / max_turns

Phase 1 defaults: α=1.0  β=0.3  γ=0.1  enable_profiling=False
```

## Phased Delivery

| Phase | Scope | Changes needed |
|-------|-------|---------------|
| 1 | KernelBench Level 1, CUDA, correctness only | This example as-is |
| 2 | Add performance reward, train KernelDesigner | Set `enable_profiling=True`; add KernelDesigner to training policy |
| 3 | Ascend backend | Change `backend=ascend`, `dsl=triton_ascend` in config template |

## Running Tests

```bash
cd /path/to/OrchRL
python -m pytest examples/akg_kernel_gen/tests/ -v
```
````

### Task 4.4: Smoke test and final commit

- [ ] Run all tests:

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -m pytest examples/akg_kernel_gen/tests/ -v
```

Expected: all tests pass (no GPU required — verifier is mocked in tests).

- [ ] Verify the assembly script can at least be imported without error:

```bash
cd /data1/lmy/agentic-rl/OrchRL
python -c "from examples.akg_kernel_gen.run_akg_rollout import build_adapter; print('OK')"
```

Expected: `OK` (or import error only for optional AKG deps, not for OrchRL glue code).

- [ ] Final commit:

```bash
git add examples/akg_kernel_gen/
git commit -m "feat(examples): complete akg_kernel_gen example with config, assembly, and README"
```
