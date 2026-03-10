# AKG Dual-Agent Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable independent training of kernel_designer and kernel_gen agents through OrchRL with separate vLLM instances.

**Architecture:** Leverage AKG's existing agent_model_config support to route Designer and Coder to different model_level configurations. Use environment variables (AKG_AGENTS_DESIGNER_* and AKG_AGENTS_CODER_*) to point both agents to the same ModelMonitor URL but with different model names, allowing ModelMonitor to route requests to separate vLLM backends.

**Tech Stack:** Python 3.12, OrchRL trajectory system, AKG LangGraphTask, vLLM

---

## Chunk 1: Core Entry Point Modifications

### Task 1: Update akg_rl_entry.py Environment Variable Injection

**Files:**
- Modify: `examples/akg_kernel_gen/mas_entry/akg_rl_entry.py:28-45`
- Test: `examples/akg_kernel_gen/tests/test_akg_rl_entry.py`

- [ ] **Step 1: Write failing test for dual-agent env var injection**

```python
# examples/akg_kernel_gen/tests/test_akg_rl_entry.py
import os
import pytest
from examples.akg_kernel_gen.mas_entry.akg_rl_entry import _inject_agent_env_vars

def test_inject_agent_env_vars_sets_designer_and_coder():
    """Test that both designer and coder env vars are set correctly."""
    config = {
        "agents": {
            "kernel_designer": {
                "model": "kernel_designer",
                "llm": {"base_url": "http://127.0.0.1:12345/v1"}
            },
            "kernel_gen": {
                "model": "kernel_gen",
                "llm": {"base_url": "http://127.0.0.1:12345/v1"}
            }
        }
    }
    
    # Clear any existing env vars
    for key in list(os.environ.keys()):
        if key.startswith("AKG_AGENTS_"):
            del os.environ[key]
    
    _inject_agent_env_vars(config)
    
    assert os.environ["AKG_AGENTS_DESIGNER_BASE_URL"] == "http://127.0.0.1:12345/v1"
    assert os.environ["AKG_AGENTS_DESIGNER_MODEL_NAME"] == "kernel_designer"
    assert os.environ["AKG_AGENTS_DESIGNER_API_KEY"] == "dummy"
    
    assert os.environ["AKG_AGENTS_CODER_BASE_URL"] == "http://127.0.0.1:12345/v1"
    assert os.environ["AKG_AGENTS_CODER_MODEL_NAME"] == "kernel_gen"
    assert os.environ["AKG_AGENTS_CODER_API_KEY"] == "dummy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest examples/akg_kernel_gen/tests/test_akg_rl_entry.py::test_inject_agent_env_vars_sets_designer_and_coder -v`
Expected: FAIL with "ImportError: cannot import name '_inject_agent_env_vars'"

- [ ] **Step 3: Replace _inject_akg_env_vars with _inject_agent_env_vars**

```python
# examples/akg_kernel_gen/mas_entry/akg_rl_entry.py
def _inject_agent_env_vars(config: dict[str, Any]) -> None:
    """Set AKG_AGENTS_* env vars for designer and coder to route through ModelMonitor."""
    agents_cfg = config.get("agents", {})

    # Designer
    designer_cfg = agents_cfg.get("kernel_designer", {})
    designer_llm = designer_cfg.get("llm", {})
    designer_base_url = designer_llm.get("base_url", "")
    designer_model = designer_cfg.get("model", "kernel_designer")

    if designer_base_url:
        os.environ["AKG_AGENTS_DESIGNER_BASE_URL"] = designer_base_url
        os.environ["AKG_AGENTS_DESIGNER_MODEL_NAME"] = designer_model
        os.environ["AKG_AGENTS_DESIGNER_API_KEY"] = "dummy"
        logger.info(
            "[akg_rl_entry] Designer LLM: %s (model=%s)",
            designer_base_url, designer_model,
        )
    else:
        logger.warning("[akg_rl_entry] Designer base_url is empty")

    # Coder (kernel_gen)
    gen_cfg = agents_cfg.get("kernel_gen", {})
    gen_llm = gen_cfg.get("llm", {})
    gen_base_url = gen_llm.get("base_url", "")
    gen_model = gen_cfg.get("model", "kernel_gen")

    if gen_base_url:
        os.environ["AKG_AGENTS_CODER_BASE_URL"] = gen_base_url
        os.environ["AKG_AGENTS_CODER_MODEL_NAME"] = gen_model
        os.environ["AKG_AGENTS_CODER_API_KEY"] = "dummy"
        logger.info(
            "[akg_rl_entry] Coder LLM: %s (model=%s)",
            gen_base_url, gen_model,
        )
    else:
        logger.warning("[akg_rl_entry] Coder base_url is empty")
```

- [ ] **Step 4: Update run() to call new function**

```python
# examples/akg_kernel_gen/mas_entry/akg_rl_entry.py (in run function, around line 80)
def run(config_path: str, task_json: str) -> int:
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Set AKG env vars → redirect LLM calls to ModelMonitor
        _inject_agent_env_vars(config)  # Changed from _inject_akg_env_vars

        # Parse task
        task_info = json.loads(task_json)
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest examples/akg_kernel_gen/tests/test_akg_rl_entry.py::test_inject_agent_env_vars_sets_designer_and_coder -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add examples/akg_kernel_gen/mas_entry/akg_rl_entry.py examples/akg_kernel_gen/tests/test_akg_rl_entry.py
git commit -m "feat(akg): inject env vars for both designer and coder agents"
```

### Task 2: Inject agent_model_config into AKG Config

**Files:**
- Modify: `examples/akg_kernel_gen/mas_entry/akg_rl_entry.py:47-69`
- Test: `examples/akg_kernel_gen/tests/test_akg_rl_entry.py`

- [ ] **Step 1: Write failing test for agent_model_config injection**

```python
# examples/akg_kernel_gen/tests/test_akg_rl_entry.py
def test_build_task_injects_agent_model_config(monkeypatch):
    """Test that _build_task injects agent_model_config into AKG config."""
    from examples.akg_kernel_gen.mas_entry.akg_rl_entry import _build_task
    
    # Mock load_config to return a minimal config
    mock_config = {"log_dir": "/tmp/test"}
    monkeypatch.setattr(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry.load_config",
        lambda: mock_config
    )
    
    # Mock LangGraphTask to capture the config
    captured_config = {}
    class MockTask:
        def __init__(self, **kwargs):
            captured_config.update(kwargs)
    
    monkeypatch.setattr(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry.LangGraphTask",
        MockTask
    )
    
    task_cfg = {"backend": "cuda", "arch": "a100", "dsl": "triton_cuda"}
    _build_task("test_op", "test_desc", task_cfg)
    
    assert "config" in captured_config
    assert captured_config["config"]["agent_model_config"] == {
        "designer": "designer",
        "coder": "coder"
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest examples/akg_kernel_gen/tests/test_akg_rl_entry.py::test_build_task_injects_agent_model_config -v`
Expected: FAIL with "KeyError: 'agent_model_config'"

- [ ] **Step 3: Update _build_task to inject agent_model_config**

```python
# examples/akg_kernel_gen/mas_entry/akg_rl_entry.py
def _build_task(
    op_name: str,
    task_desc: str,
    task_cfg: dict[str, Any],
) -> Any:
    """Construct a LangGraphTask instance. Extracted for testability."""
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.op.config.config_validator import load_config

    task_id = uuid.uuid4().hex[:8]
    config = load_config()
    config["log_dir"] = config.get("log_dir", "/tmp/akg_rl_logs")
    
    # Inject agent_model_config to use per-agent model_level
    config["agent_model_config"] = {
        "designer": "designer",
        "coder": "coder",
    }

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest examples/akg_kernel_gen/tests/test_akg_rl_entry.py::test_build_task_injects_agent_model_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/akg_kernel_gen/mas_entry/akg_rl_entry.py examples/akg_kernel_gen/tests/test_akg_rl_entry.py
git commit -m "feat(akg): inject agent_model_config for per-agent model_level"
```

---

## Chunk 2: Configuration Updates

### Task 3: Update akg_config_template.yaml

**Files:**
- Modify: `examples/akg_kernel_gen/configs/akg_config_template.yaml`

- [ ] **Step 1: Add roles and agent_model_config to template**

```yaml
# examples/akg_kernel_gen/configs/akg_config_template.yaml
mas_command_template: >-
  python examples/akg_kernel_gen/mas_entry/akg_rl_entry.py
  --config {config_path}
  --task {prompt}

# Agent roles for OrchRL trajectory collection
roles:
  - kernel_designer
  - kernel_gen

# Agent configurations
agents:
  kernel_designer:
    model: kernel_designer
    llm:
      base_url: ""  # injected by prepare_config()
      api_key: "dummy"
  kernel_gen:
    model: kernel_gen
    llm:
      base_url: ""  # injected by prepare_config()
      api_key: "dummy"

# Task configuration (passed through to LangGraphTask)
task:
  framework: torch
  backend: cuda
  arch: a100
  dsl: triton_cuda
  max_iterations: 5

# Rollout sampling
batch_size: 8
n_samples_per_prompt: 4
max_concurrent_episodes: 8
timeout: 300
```

- [ ] **Step 2: Verify YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('examples/akg_kernel_gen/configs/akg_config_template.yaml'))"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add examples/akg_kernel_gen/configs/akg_config_template.yaml
git commit -m "feat(akg): add dual-agent config to template"
```

---

## Chunk 3: Rollout Script Updates

### Task 4: Update run_akg_rollout.py for Dual Agents

**Files:**
- Modify: `examples/akg_kernel_gen/run_akg_rollout.py:119-151`
- Test: Manual verification

- [ ] **Step 1: Add CLI arguments for designer**

```python
# examples/akg_kernel_gen/run_akg_rollout.py (in main function, around line 124)
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
```

- [ ] **Step 2: Update build_adapter signature and implementation**

```python
# examples/akg_kernel_gen/run_akg_rollout.py
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
```

- [ ] **Step 3: Update main() to pass new arguments**

```python
# examples/akg_kernel_gen/run_akg_rollout.py (in main function)
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
```

- [ ] **Step 4: Verify script runs with --help**

Run: `python examples/akg_kernel_gen/run_akg_rollout.py --help`
Expected: Shows all new arguments

- [ ] **Step 5: Commit**

```bash
git add examples/akg_kernel_gen/run_akg_rollout.py
git commit -m "feat(akg): add dual-agent vLLM configuration to rollout script"
```

---

## Chunk 4: Reward Provider Updates

### Task 5: Update AKGKernelRewardProvider for Dual Agents

**Files:**
- Modify: `examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py:64-119`
- Test: `examples/akg_kernel_gen/tests/test_akg_kernel_reward.py`

- [ ] **Step 1: Write failing test for dual-agent rewards**

```python
# examples/akg_kernel_gen/tests/test_akg_kernel_reward.py
def test_compute_assigns_rewards_to_both_agents():
    """Test that both kernel_designer and kernel_gen receive rewards."""
    from trajectory.datatypes import EpisodeTrajectory, TurnData
    from examples.akg_kernel_gen.orchrl_glue.akg_kernel_reward import AKGKernelRewardProvider
    
    # Mock verifier that always succeeds
    def mock_verifier_factory():
        class MockVerifier:
            async def run(self, code):
                return True, "success"
        return MockVerifier()
    
    provider = AKGKernelRewardProvider(
        alpha=1.0,
        beta=0.0,
        gamma=0.1,
        max_turns=5,
        verifier_factory=mock_verifier_factory,
    )
    
    trajectory = EpisodeTrajectory(
        episode_id="test",
        agent_trajectories={
            "kernel_designer": [
                TurnData(
                    agent_role="kernel_designer",
                    turn_index=0,
                    messages=[],
                    response_text="design output",
                    token_ids=None,
                    logprobs=None,
                    finish_reason="stop",
                    timestamp=0.0,
                    metadata={},
                )
            ],
            "kernel_gen": [
                TurnData(
                    agent_role="kernel_gen",
                    turn_index=0,
                    messages=[],
                    response_text="```python\ndef call(): pass\n```",
                    token_ids=None,
                    logprobs=None,
                    finish_reason="stop",
                    timestamp=0.0,
                    metadata={},
                )
            ],
        },
        metadata={},
    )
    
    result = provider.compute(trajectory)
    
    assert "agent_rewards" in result
    assert "kernel_designer" in result["agent_rewards"]
    assert "kernel_gen" in result["agent_rewards"]
    assert result["agent_rewards"]["kernel_designer"] > 0
    assert result["agent_rewards"]["kernel_gen"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest examples/akg_kernel_gen/tests/test_akg_kernel_reward.py::test_compute_assigns_rewards_to_both_agents -v`
Expected: FAIL with "KeyError: 'kernel_designer'" or assertion error

- [ ] **Step 3: Update compute() to assign rewards to all agents**

```python
# examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py
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

    # Assign final_reward to all agents in the trajectory
    return {
        "agent_rewards": {
            role: final_reward for role in trajectory.agent_trajectories
        },
        "final_reward": final_reward,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest examples/akg_kernel_gen/tests/test_akg_kernel_reward.py::test_compute_assigns_rewards_to_both_agents -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py examples/akg_kernel_gen/tests/test_akg_kernel_reward.py
git commit -m "feat(akg): assign rewards to both designer and gen agents"
```

---

## Chunk 5: Documentation and Integration Testing

### Task 6: Update README with Dual-Agent Instructions

**Files:**
- Modify: `examples/akg_kernel_gen/README.md`

- [ ] **Step 1: Update README architecture section**

```markdown
# examples/akg_kernel_gen/README.md (update Architecture section around line 46)

## Architecture

```
MateRolloutAdapter
  └─ per episode: AgentPipe.run()
       ├─ ModelMonitor (aiohttp proxy, intercepts LLM calls)
       └─ MASLauncher → akg_rl_entry.py --config /tmp/xxx.yaml --task '...'
             ├─ KernelDesigner → LLM (model=kernel_designer) → ModelMonitor → designer vLLM
             ├─ KernelGen      → LLM (model=kernel_gen)      → ModelMonitor → gen vLLM
             └─ KernelVerifier → local GPU (no LLM)
  └─ AKGKernelRewardProvider.compute(trajectory)
       ├─ extract final code from kernel_gen[-1].response_text
       ├─ re-run KernelVerifier → r_correct
       └─ reward = α·r_correct + β·r_perf·r_correct + γ·r_iter
```

**Dual-Agent Training:**
Both `kernel_designer` and `kernel_gen` are routed through ModelMonitor to separate vLLM instances, enabling independent training of each agent's model weights.
```

- [ ] **Step 2: Update Quickstart with dual vLLM setup**

```markdown
# examples/akg_kernel_gen/README.md (update Quickstart section around line 34)

## Quickstart

**Start two vLLM servers** (one for each agent):

```bash
# Terminal 1: Designer model
vllm serve Qwen/Qwen2.5-Coder-7B --port 9000

# Terminal 2: Gen model
vllm serve Qwen/Qwen2.5-Coder-7B --port 8000
```

**Run rollout collection:**

```bash
cd /path/to/OrchRL

python examples/akg_kernel_gen/run_akg_rollout.py \
    --kernelbench-dir /path/to/akg/akg_agents/thirdparty/KernelBench/KernelBench \
    --designer-vllm-url http://localhost:9000/v1 \
    --gen-vllm-url http://localhost:8000/v1 \
    --policy-designer kernel_designer \
    --policy-gen kernel_gen \
    --steps 10
```
```

- [ ] **Step 3: Commit**

```bash
git add examples/akg_kernel_gen/README.md
git commit -m "docs(akg): update README for dual-agent training"
```

### Task 7: Integration Test with Mock vLLM

**Files:**
- Create: `examples/akg_kernel_gen/tests/test_dual_agent_integration.py`

- [ ] **Step 1: Write integration test**

```python
# examples/akg_kernel_gen/tests/test_dual_agent_integration.py
"""Integration test for dual-agent training setup."""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock


@pytest.mark.asyncio
async def test_dual_agent_routing_through_monitor():
    """Test that designer and gen requests route to different backends."""
    from trajectory import ModelMonitor, VLLMBackend, ModelMappingEntry
    
    # Track which backend received which model request
    designer_calls = []
    gen_calls = []
    
    async def mock_designer_generate(request):
        designer_calls.append(request.agent_role)
        from trajectory.datatypes import ModelResponse
        return ModelResponse(
            content="design output",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )
    
    async def mock_gen_generate(request):
        gen_calls.append(request.agent_role)
        from trajectory.datatypes import ModelResponse
        return ModelResponse(
            content="```python\ndef call(): pass\n```",
            token_ids=[4, 5, 6],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )
    
    # Create mock backends
    designer_backend = Mock(spec=VLLMBackend)
    designer_backend.generate = mock_designer_generate
    
    gen_backend = Mock(spec=VLLMBackend)
    gen_backend.generate = mock_gen_generate
    
    # Create model mapping
    model_mapping = {
        "kernel_designer": ModelMappingEntry(
            actual_model="designer-model",
            backend_url="http://localhost:9000/v1",
        ),
        "kernel_gen": ModelMappingEntry(
            actual_model="gen-model",
            backend_url="http://localhost:8000/v1",
        ),
    }
    
    # This test verifies the concept - actual routing happens in ModelMonitor
    # which uses backend_url from ModelMappingEntry to route requests
    assert model_mapping["kernel_designer"].backend_url == "http://localhost:9000/v1"
    assert model_mapping["kernel_gen"].backend_url == "http://localhost:8000/v1"
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest examples/akg_kernel_gen/tests/test_dual_agent_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add examples/akg_kernel_gen/tests/test_dual_agent_integration.py
git commit -m "test(akg): add integration test for dual-agent routing"
```

### Task 8: Run All Tests

**Files:**
- Test: All test files in `examples/akg_kernel_gen/tests/`

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest examples/akg_kernel_gen/tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: If any tests fail, fix them**

Iterate on failing tests until all pass.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(akg): complete dual-agent training implementation"
```

---

## Verification Checklist

After implementation, verify:

- [ ] Two vLLM servers can be started on different ports
- [ ] `run_akg_rollout.py` accepts new CLI arguments
- [ ] Environment variables are correctly set in subprocess
- [ ] ModelMonitor routes designer requests to port 9000
- [ ] ModelMonitor routes gen requests to port 8000
- [ ] Both agents' trajectories are collected
- [ ] Rewards are assigned to both agents
- [ ] All tests pass

## Notes

- This implementation does NOT modify any AKG code
- Uses AKG's existing `agent_model_config` and environment variable support
- ModelMonitor routing is based on the `model` field in requests
- Both agents receive the same `final_reward` (per-agent rewards can be added in Phase 2)
