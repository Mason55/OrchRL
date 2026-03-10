# AKG Agents × OrchRL Integration Design

**Date:** 2026-03-10
**Status:** Approved
**Scope:** `OrchRL/examples/akg_kernel_gen/` — example only, no changes to OrchRL core

---

## 1. Goal

Apply reinforcement learning (GRPO) to improve AKG Agents' kernel code generation quality. The RL training loop is provided by OrchRL; AKG's `LangGraphTask` (KernelGen + KernelDesigner + KernelVerifier) runs as the external MAS subprocess.

**Multi-objective reward:** correctness (compile + functional correctness) × performance (speedup vs PyTorch reference) × iteration efficiency.

---

## 2. Approach

**AKG as MATE black-box subprocess.** OrchRL's existing `MateRolloutAdapter` launches AKG as a subprocess, intercepts all LLM calls via `ModelMonitor` (OpenAI-compatible proxy), records `token_ids` for gradient computation, then computes reward via `AKGKernelRewardProvider`.

No changes to OrchRL core code. The example implements three OrchRL interfaces:
- `prompt_loader` — duck-typed, `get_step_batch(step_idx, batch_size)`
- `RewardProvider` — Protocol, `compute(trajectory) → dict`
- `akg_rl_entry.py` — AKG subprocess entry point

---

## 3. Architecture & Data Flow

```
OrchRL Trainer (Ray)
  │
  ├── KernelBenchLoader.get_step_batch()
  │     └── [{"prompt": '{"op_name":...,"task_desc":"..."}', "raw": {...}}, ...]
  │
  └── MateRolloutAdapter (existing OrchRL class, unchanged)
        │
        ├── [per episode] AgentPipe.run()
        │     │
        │     ├── ModelMonitor (aiohttp proxy, intercepts LLM calls, records token_ids)
        │     │
        │     └── MASLauncher → subprocess: akg_rl_entry.py --config /tmp/xxx.yaml --task '...'
        │           │
        │           ├── KernelDesigner ──→ LLM call (model="kernel_designer") ──→ ModelMonitor
        │           ├── KernelGen      ──→ LLM call (model="kernel_gen")      ──→ ModelMonitor
        │           └── KernelVerifier ──→ local GPU execution (no LLM)
        │                 └── guides ReAct loop; does NOT determine exit code
        │
        └── AKGKernelRewardProvider.compute(trajectory)
              ├── extract final code from trajectory.agent_trajectories["kernel_gen"][-1]
              ├── re-run KernelVerifier → r_correct (0 or 1)
              ├── optional: run profiler → r_perf (speedup ratio)
              └── return {"agent_rewards": {...}, "final_reward": float}

                    ↓ vLLM backend (training model)
              vLLM Server (KernelGen policy being trained)
```

### Critical design note: exit code

`AgentPipe.run()` raises `RuntimeError` on non-zero exit code, causing the episode to be dropped by `parallel_rollout`. Since GRPO requires negative examples (failed generations), `akg_rl_entry.py` **must always exit 0**. Correctness is determined independently by `AKGKernelRewardProvider`, not by the subprocess exit code.

---

## 4. Agent Roles & Model Mapping

```python
model_mapping = {
    "kernel_gen": ModelMappingEntry(
        actual_model="Qwen2.5-Coder-7B",   # model being trained
        backend_url="http://vllm-host:8000/v1"
    ),
    "kernel_designer": ModelMappingEntry(
        actual_model="Qwen2.5-Coder-7B",   # frozen in Phase 1
        backend_url="http://vllm-host:8000/v1"
    ),
}
```

AKG's `AgentBase` sets `context["agent_name"]` on every LLM request. The entry script maps `agent_name → model name` so ModelMonitor can route correctly. In Phase 1, only `kernel_gen` is trained; `kernel_designer` points to the same frozen vLLM endpoint.

---

## 5. Config Injection

OrchRL's `prepare_config()` writes a temporary YAML with `base_url = monitor_url` injected into per-role `llm` sections. The entry script reads this YAML and configures AKG's LLM client via environment variables:

```python
# akg_rl_entry.py
os.environ["AKG_AGENTS_BASE_URL"]    = config["agents"]["kernel_gen"]["llm"]["base_url"]
os.environ["AKG_AGENTS_MODEL_NAME"]  = config["agents"]["kernel_gen"]["model"]
```

**Prerequisite:** Confirm `AKG_AGENTS_BASE_URL` and `AKG_AGENTS_MODEL_NAME` are read in `akg_agents/core_v2/config/settings.py`. If not, a 2-3 line addition to AKG's settings loader is needed (separate PR to AKG repo).

Config template (stored in `configs/akg_config_template.yaml`):

```yaml
agents:
  kernel_gen:
    model: kernel_gen
    llm:
      base_url: ""        # injected by prepare_config
      api_key: "dummy"
  kernel_designer:
    model: kernel_designer
    llm:
      base_url: ""        # injected by prepare_config
      api_key: "dummy"

task:
  framework: torch
  backend: cuda
  arch: a100
  dsl: triton_cuda
  max_iterations: 5

mas_command_template: "python examples/akg_kernel_gen/mas_entry/akg_rl_entry.py --config {config_path} --task {prompt}"
batch_size: 8
n_samples_per_prompt: 4
max_concurrent_episodes: 8
timeout: 300
```

---

## 6. Reward Design

### Components

| Component | Meaning | Computation | Cost |
|-----------|---------|-------------|------|
| `r_correct` | Compile + functional correctness | Re-run KernelVerifier on final code | Medium (GPU) |
| `r_perf` | Speedup vs PyTorch reference | `actual_time / reference_time`, clipped to [0, 2] then normalized | High (GPU profiling) |
| `r_iter` | Iteration efficiency | `1.0 - n_turns / max_turns` | Zero |

### Formula

```
reward = α · r_correct + β · r_perf · r_correct + γ · r_iter
```

Phase 1 defaults: `α=1.0, β=0.3, γ=0.1, enable_profiling=False`

### Credit assignment

`credit_assignment: all_turns` — final reward broadcast to all `kernel_gen` turns in the episode. Handled by `MateDataprotoAdapter` (existing OrchRL, unchanged).

---

## 7. KernelBench Prompt Format

`prompt_loader` returns items in the shape `MateRolloutAdapter` expects:

```python
{
    "prompt": '{"op_name": "relu_1", "task_desc": "import torch\\n..."}',
    "raw": {"op_name": "relu_1", "level": 1, ...},
}
```

The `prompt` field is passed verbatim as `{prompt}` in `mas_command_template`. The entry script parses the JSON to extract `op_name` and `task_desc`.

---

## 8. File Layout

```
OrchRL/examples/akg_kernel_gen/
├── README.md
├── requirements.txt
│
├── mas_entry/
│   └── akg_rl_entry.py          # AKG subprocess entry; always exit 0
│
├── orchrl_glue/
│   ├── kernelbench_loader.py    # prompt_loader: KernelBench → MateRolloutAdapter format
│   └── akg_kernel_reward.py     # RewardProvider: extract code → verify → reward
│
├── configs/
│   └── akg_config_template.yaml # MAS command template + task config
│
└── run_akg_rollout.py           # assembly script: MateRolloutAdapter + training loop
```

---

## 9. Phased Delivery

| Phase | Scope | Reward | Roles |
|-------|-------|--------|-------|
| 1 | Full pipeline on KernelBench Level 1, CUDA | Correctness only (`enable_profiling=False`) | `kernel_gen` trained, `kernel_designer` frozen |
| 2 | KernelBench Level 1-3, add profiling reward | Correctness + performance | `kernel_gen` + `kernel_designer` jointly trained |
| 3 | Ascend backend | Same reward | Switch `dsl=triton_ascend`, reuse existing Ascend verifier adapter |

---

## 10. GPU Resource Layout (8-GPU example)

```
GPU 0-5: vLLM training server (KernelGen policy, supports 8 concurrent KV cache slots)
GPU 6-7: KernelVerifier execution pool (compile + correctness verification)
         Phase 2: also used for profiling
```

`max_concurrent_episodes=8` matches vLLM concurrent request capacity to avoid OOM.

---

## 11. Out of Scope

- OrchRL core refactoring (`RolloutSource` ABC, `MultiAgentsPPOTrainer` decoupling) — deferred
- AKG's `core_v2` architectural changes — none required
- Custom ModelMonitor modifications — none required
