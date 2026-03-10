# AKG Dual-Agent Training Design

## 背景

当前 `akg_kernel_gen` 示例只训练 `kernel_gen` agent，`kernel_designer` 使用固定模型。需要将两个 agent 都接入 OrchRL 进行独立训练，使用各自的 vLLM 实例和模型权重。

**约束条件**：
- 不能修改 AKG 代码（akg_agents 仓库）
- 必须依赖 AKG 内部的 LangGraphTask 编排 designer → gen 的多轮交互
- 两个 agent 使用独立的 vLLM 实例和模型

## 核心发现

AKG 已支持 per-agent 的 model_level 配置：
- Designer 使用 `config["agent_model_config"]["designer"]`
- Coder 使用 `config["agent_model_config"]["coder"]`
- 通过环境变量 `AKG_AGENTS_{LEVEL}_BASE_URL` / `AKG_AGENTS_{LEVEL}_MODEL_NAME` 配置

## 架构设计

### 数据流

```
run_akg_rollout.py
  ├─ designer vLLM (localhost:9000)
  └─ gen vLLM (localhost:8000)
                    ↓
MateRolloutAdapter
  └─ AgentPipe
       ├─ ModelMonitor (单一 proxy URL)
       │    ├─ model=kernel_designer → routes to localhost:9000
       │    └─ model=kernel_gen      → routes to localhost:8000
       │
       └─ MASLauncher → akg_rl_entry.py
            └─ 设置环境变量:
                 AKG_AGENTS_DESIGNER_BASE_URL=monitor_url
                 AKG_AGENTS_DESIGNER_MODEL_NAME=kernel_designer
                 AKG_AGENTS_CODER_BASE_URL=monitor_url
                 AKG_AGENTS_CODER_MODEL_NAME=kernel_gen
            └─ AKG LangGraphTask
                 ├─ Designer → create_llm_client(model_level="designer")
                 │             → reads AKG_AGENTS_DESIGNER_*
                 │             → calls monitor_url with model=kernel_designer
                 └─ Coder    → create_llm_client(model_level="coder")
                               → reads AKG_AGENTS_CODER_*
                               → calls monitor_url with model=kernel_gen
```

### 关键机制

1. **统一 MonitorURL**：两个 agent 都调用同一个 ModelMonitor proxy
2. **model 字段区分**：通过 `AKG_AGENTS_{LEVEL}_MODEL_NAME` 设置不同的 model name
3. **ModelMonitor 路由**：根据请求中的 `model` 字段路由到对应的 vLLM backend
4. **独立 vLLM 实例**：designer 和 gen 各自有独立的 vLLM server 和模型权重

## 具体改动

### 1. `akg_rl_entry.py`

**删除**：
```python
def _inject_akg_env_vars(config: dict[str, Any]) -> None:
    # 只注入 kernel_gen 的配置
```

**新增**：
```python
def _inject_agent_env_vars(config: dict[str, Any]) -> None:
    """从 temp YAML 读取 monitor_url，为 designer 和 coder 分别设置环境变量。"""
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
```

**修改 `_build_task`**：
```python
def _build_task(...) -> Any:
    from akg_agents.op.config.config_validator import load_config

    config = load_config()
    config["log_dir"] = config.get("log_dir", "/tmp/akg_rl_logs")

    # 注入 agent_model_config，让 AKG 使用 per-agent 的 model_level
    config["agent_model_config"] = {
        "designer": "designer",  # 使用 AKG_AGENTS_DESIGNER_* 环境变量
        "coder": "coder",        # 使用 AKG_AGENTS_CODER_* 环境变量
    }

    return LangGraphTask(...)
```

**修改 `run`**：
```python
def run(config_path: str, task_json: str) -> int:
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 设置环境变量
        _inject_agent_env_vars(config)

        # 解析 task
        task_info = json.loads(task_json)
        ...

        # 构建并运行 task
        task = _build_task(...)
        ...
```

### 2. `akg_config_template.yaml`

```yaml
mas_command_template: >-
  python examples/akg_kernel_gen/mas_entry/akg_rl_entry.py
  --config {config_path}
  --task {prompt}

# 新增：明确列出两个 agent role
roles:
  - kernel_designer
  - kernel_gen

agents:
  kernel_designer:
    model: kernel_designer
    llm:
      base_url: ""  # 由 prepare_config() 注入 monitor_url
      api_key: "dummy"
  kernel_gen:
    model: kernel_gen
    llm:
      base_url: ""  # 由 prepare_config() 注入 monitor_url
      api_key: "dummy"

# 新增：AKG agent_model_config（在 akg_rl_entry.py 中注入到 AKG config）
agent_model_config:
  designer: "designer"
  coder: "coder"

task:
  framework: torch
  backend: cuda
  arch: a100
  dsl: triton_cuda
  max_iterations: 5

batch_size: 8
n_samples_per_prompt: 4
max_concurrent_episodes: 8
timeout: 300
```

### 3. `run_akg_rollout.py`

**新增参数**：
```python
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

**修改 `build_adapter`**：
```python
def build_adapter(...) -> MateRolloutAdapter:
    ...
    return MateRolloutAdapter(
        config=config_template,
        prompt_loader=loader,
        reward_provider=reward_provider,
        server_address_dict={
            args.policy_designer: args.designer_vllm_url,
            args.policy_gen: args.gen_vllm_url,
        },
        role_policy_mapping={
            "kernel_designer": args.policy_designer,
            "kernel_gen": args.policy_gen,
        },
        policy_server_name_mapping={
            args.policy_designer: args.designer_model,
            args.policy_gen: args.gen_model,
        },
    )
```

### 4. `akg_kernel_reward.py`

**修改 `compute` 方法**：
```python
def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]:
    kernel_gen_turns = trajectory.agent_trajectories.get("kernel_gen", [])

    if not kernel_gen_turns:
        logger.warning(...)
        final_reward = 0.0
        return {
            "agent_rewards": {
                role: final_reward
                for role in trajectory.agent_trajectories
            },
            "final_reward": final_reward,
        }

    # 提取生成的代码
    last_response = kernel_gen_turns[-1].response_text
    code = _extract_code(last_response)

    # 计算奖励
    r_correct = self._compute_correctness(code)
    r_perf = 0.0
    if r_correct and self.enable_profiling:
        r_perf = self._compute_performance(code)

    n_turns = len(kernel_gen_turns)
    r_iter = max(0.0, 1.0 - n_turns / self.max_turns)

    final_reward = (
        self.alpha * r_correct
        + self.beta * r_perf * r_correct
        + self.gamma * r_iter
    )

    # 给所有 agent 分配相同的 final_reward
    return {
        "agent_rewards": {
            role: final_reward for role in trajectory.agent_trajectories
        },
        "final_reward": final_reward,
    }
```

## 使用示例

```bash
# 启动两个 vLLM 实例
vllm serve Qwen/Qwen2.5-Coder-7B --port 9000  # designer
vllm serve Qwen/Qwen2.5-Coder-7B --port 8000  # gen

# 运行 rollout
python examples/akg_kernel_gen/run_akg_rollout.py \
    --kernelbench-dir /path/to/KernelBench \
    --designer-vllm-url http://localhost:9000/v1 \
    --gen-vllm-url http://localhost:8000/v1 \
    --policy-designer kernel_designer \
    --policy-gen kernel_gen \
    --steps 10
```

## 测试计划

1. **单元测试**：
   - `test_inject_agent_env_vars`：验证环境变量正确设置
   - `test_build_task_with_agent_model_config`：验证 agent_model_config 正确注入

2. **集成测试**：
   - 启动两个 mock vLLM server
   - 运行单个 episode，验证 ModelMonitor 正确路由
   - 检查 trajectory 中 designer 和 gen 的 turns 都被记录

3. **端到端测试**：
   - 运行完整的 rollout（10 steps）
   - 验证 reward 计算正确
   - 验证两个 agent 的 token_ids 和 logprobs 都被收集

## 风险和限制

1. **AKG 版本依赖**：依赖 AKG 支持 `agent_model_config` 和环境变量配置，需要确认 AKG 版本
2. **环境变量冲突**：如果用户已设置 `AKG_AGENTS_*` 环境变量，可能被覆盖
3. **奖励分配策略**：当前给两个 agent 分配相同的 final_reward，未来可能需要 per-agent 的奖励

## 未来扩展

1. **Phase 2**：添加 performance reward，启用 profiling
2. **Phase 3**：支持 Ascend backend
3. **Per-agent 奖励**：根据各自的贡献分配不同的奖励
