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
