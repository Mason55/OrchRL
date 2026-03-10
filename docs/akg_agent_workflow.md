# OrchRL — AKG Kernel Gen Agent 工作流文档

## 1. OrchRL 设计理念与框架方案

### 1.1 核心设计目标

OrchRL 是一个**面向 Agent 的强化学习框架**，解决的核心问题是：

> **如何在不修改任何 Agent 内部代码的前提下，对复杂的多 Agent 工作流进行 RL 训练？**

传统 RL 框架要求训练系统深度集成到推理逻辑中，而 OrchRL 采用**黑盒子进程（MATE）** 模式——Agent 以独立子进程运行，框架只在"外部"捕获其 LLM 交互轨迹。

### 1.2 方案核心：MATE（Multi-Agent Trajectory Engine）

MATE 是 OrchRL 的轨迹采集引擎，基于以下三个关键机制：

#### 机制一：ModelMonitor — 透明 HTTP 代理

`ModelMonitor`（`orchrl/agent_trajectory_engine/gateway.py`）是整个方案的核心。它启动一个本地 HTTP 服务，**完全兼容 OpenAI `/v1/chat/completions` 接口**，充当 Agent 和真实 vLLM 推理服务之间的透明代理：

```
Agent subprocess
    │  POST /v1/chat/completions  model="kernel_gen"
    ▼
ModelMonitor (本地 HTTP 服务, 随机端口)
    │  1. 按 model 字段识别 agent_role
    │  2. 查 model_mapping → 获取真实 model name + backend_url
    │  3. 转发请求到真实 vLLM
    │  4. 记录完整 InteractionRecord 到 buffer
    │  5. 返回响应给 Agent（透明转发）
    ▼
vLLM Server (真实推理服务)
```

无侵入原理：Agent 只需将 `base_url` 改为 ModelMonitor 的地址，其他代码**零修改**。

#### 机制二：配置注入 — 运行时替换 base_url

`MASLauncher`（`orchrl/agent_trajectory_engine/launcher.py`）在每个 episode 启动前：
1. 深拷贝 `config_template`（YAML 配置文件）
2. 将 ModelMonitor 的监听地址注入 `agents.*.llm.base_url`
3. 写入临时文件 `/tmp/trajectory_mas_xxxxx.yaml`
4. 以子进程启动 Agent，传入临时配置路径

每个 episode 独立配置文件 → 支持大规模并发采集互不干扰。

#### 机制三：轨迹收集 — 结构化数据聚合

`TrajectoryCollector`（`orchrl/agent_trajectory_engine/collector.py`）将 ModelMonitor buffer 中的 `InteractionRecord` 按 `agent_role` 分组，构建 `EpisodeTrajectory`：

```python
# 核心数据结构
@dataclass
class EpisodeTrajectory:
    episode_id: str
    agent_trajectories: dict[str, list[TurnData]]  # 按角色分类的对话轮次
    metadata: dict[str, Any]

@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    messages: list[dict]      # 完整对话历史
    response_text: str        # 模型生成内容
    token_ids: list[int]      # 用于 RL 训练的 token 序列
    logprobs: list[float]     # 对数概率（可选）
    finish_reason: str
```

### 1.3 框架分层架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户集成层                             │
│  MateRolloutAdapter — 连接训练循环与 Agent 工作流          │
│  · prompt_loader (duck-type: get_step_batch)             │
│  · reward_provider (duck-type: compute)                  │
├─────────────────────────────────────────────────────────┤
│                   rollout 执行层                          │
│  parallel_rollout / tree_rollout                         │
│  · asyncio 并发控制 (Semaphore)                           │
│  · AgentPipe — 单 episode 完整执行流水线                  │
├─────────────────────────────────────────────────────────┤
│                 Agent Trajectory Engine                  │
│  ModelMonitor  — HTTP 代理，捕获 LLM 交互                 │
│  MASLauncher   — 子进程生命周期管理                        │
│  TrajectoryCollector — 轨迹结构化聚合                     │
│  VLLMBackend   — 真实推理服务适配器                        │
│  ReplayCache   — 树形 rollout 的分支复用缓存               │
├─────────────────────────────────────────────────────────┤
│                  训练框架层 (verl)                         │
│  GRPO / PPO — 利用 EpisodeResult 计算策略梯度更新模型       │
└─────────────────────────────────────────────────────────┘
```

### 1.4 两种 Rollout 模式

**Parallel Rollout**（默认）：
- 每个 prompt 独立启动 N 个 episode，完全并行
- 适合 GRPO：同一 prompt 的多条轨迹用于估计相对优势函数

**Tree Rollout**：
- 先运行一条 pilot episode，在指定 turn 处分叉出 K 条 branch
- Branch 通过 `ReplayCache` 复用 pilot 的前置轨迹，避免重复推理
- 适合需要探索的任务（如多步规划、代码调试）

### 1.5 接口协议（Duck Typing）

OrchRL 通过鸭子类型定义两个核心接口，用户无需继承任何基类：

```python
# prompt_loader 协议
class PromptLoader:
    def get_step_batch(self, step_idx: int, batch_size: int) -> list[dict]:
        # 返回: [{"prompt": str, "raw": dict, "expected": any}, ...]
        ...

# reward_provider 协议
class RewardProvider:
    def compute(self, trajectory: EpisodeTrajectory) -> dict:
        # 返回: {"agent_rewards": {role: float}, "final_reward": float}
        ...
```

只要实现这两个方法，任何 Agent 工作流都可以直接接入 OrchRL。

---

### 1.6 OrchRL 对 verl 的封装与扩展

OrchRL 以 verl 为基础训练框架，在其上做了三个层次的包装：

#### 层次一：`RayPPOTrainer` — 扩展推理引擎（`orchrl/verl/ray_trainer.py`）

继承 verl 原生 `RayPPOTrainer`，主要扩展：

| 扩展点 | 说明 |
|--------|------|
| `AsyncLLMServerManager` | 每个 policy 可挂多个 vLLM 服务实例，支持副本间**最少请求数负载均衡**（`ChatCompletionScheduler` heap 实现）|
| `sleep` / `wake_up` | rollout 阶段唤醒推理服务加载模型权重，training 阶段休眠释放显存，实现推理与训练的显存复用 |
| 扩展 `AdvantageEstimator` | 在 verl 的 GAE/PPO 基础上，新增 `GRPO`、`REINFORCE_PLUS_PLUS`、`REMAX`、`RLOO`、`LOOP`、`GRPO_PASSK` 等 Agent RL 常用估计器 |
| `apply_kl_penalty` | 支持将 KL 惩罚直接注入 reward（`use_kl_in_reward` 模式），而非作为 loss 项 |

#### 层次二：`MultiAgentsPPOTrainer` — 多 Agent 训练编排（`orchrl/trainer/multi_agents_ppo_trainer.py`）

OrchRL 最核心的训练编排器，管理整个多 Agent RL 训练循环：

**三种 Agent 特化模式（`specialization`）**：

```
full    → 每个 Agent 独立 RayPPOTrainer + 独立模型权重（参数完全分离）
prompt  → 共享单个 RayPPOTrainer，通过 system prompt 区分 Agent 行为
lora    → 共享基础模型，每个 Agent 持有独立 LoRA adapter（显存高效）
```

**完整训练步（`fit()` 主循环）**：

```
每个 step:
  [1] wake_up()          唤醒所有 policy 的 vLLM 推理服务
       │
  [2] _collect_mate_step_batches()
       │  MATE 轨迹采集（调用 MateRolloutAdapter）
       │  → episodes_to_policy_batches()   按 policy 分桶
       │  → DataProto（prompts / responses / response_mask / reward）
       │
  [3] compute_log_prob()     计算 old log probabilities（在 actor workers 上）
       │
  [4] compute_ref_log_prob() 计算参考模型 log prob（用于 KL 惩罚，可选）
       │
  [5] compute_values()       critic 估值（使用 Critic 时）
       │
  [6] apply_kl_penalty()     KL reward 注入（可选）
       │
  [7] compute_advantage()    优势估计（GRPO / GAE / RLOO 等）
       │
  [8] update_critic()        更新 critic（可选）
       │
  [9] update_actor()         更新 actor
       │  LoRA 模式：按 agent_name 分桶，分别更新各 agent 的 LoRA adapter
       │
  [10] sleep()               休眠 vLLM 推理服务，释放显存
       │
  [11] save_checkpoint()     保存检查点
```

**`agent_untrained` 机制**：配置中声明的角色（如冻结的 `kernel_designer`）会在数据过滤阶段被排除，其轨迹不参与梯度更新，但仍然参与 rollout 采集。

#### 层次三：`mate_dataproto_adapter` — 轨迹 → 训练数据（`orchrl/trainer/mate_dataproto_adapter.py`）

连接 MATE 轨迹引擎与 verl 训练框架的**格式转换层**，是多 Agent 训练的关键胶水：

```
EpisodeResult (多 Agent 多轮对话轨迹)
      │
      ▼  episodes_to_policy_batches()
      │
      ├── 按 agent_role → policy_name 分桶
      │
      ├── 对每个 turn：tokenize messages → prompt_ids，截断至 max_prompt_length
      │
      ├── 信用分配（credit_assignment）：
      │     all_turns → 将 final_reward 复制到该 role 的每一轮
      │     last_turn → reward 只放在最后一轮，其余轮为 0
      │
      └── 打包为 verl DataProto：
            tensors:     prompts, responses, response_mask
            non_tensors: agent_name, agent_idx, turn_idx, env_idx,
                         episode_id, prompt_group_id, sample_idx,
                         reward, uid
```

树形 rollout 额外处理：`tree_episodes_to_policy_batches` 同时处理 pilot episode 和 branch episodes，按全局时间戳重建执行顺序，branch 只包含分叉点之后的 turn。

**整体封装关系**：

```
verl (基础 RL 训练框架)
  └── orchrl/verl/ray_trainer.py
        RayPPOTrainer (扩展推理引擎 + 更多估计器)
            └── orchrl/trainer/multi_agents_ppo_trainer.py
                  MultiAgentsPPOTrainer (多 Agent 编排 + MATE 集成)
                       ├── MateRolloutAdapter  ← MATE 轨迹采集
                       ├── mate_dataproto_adapter  ← 轨迹→DataProto 转换
                       └── 多 RayPPOTrainer 实例 (full/prompt/lora 模式)
```

---

## 2. AKG Kernel Gen — 任务目标

训练一个 LLM（`kernel_gen` 角色），使其能够为给定的 PyTorch 算子自动生成**正确且高效的 CUDA/Triton kernel**。

训练数据来自 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 数据集，每个任务包含：
- `op_name`：算子名称，如 `1_relu`、`5_matrix_multiply`
- `task_desc`：PyTorch 参考实现代码

---

## 3. AKG Kernel Gen — 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        OrchRL 训练循环                           │
│                                                                  │
│  KernelBenchLoader                                               │
│       │  get_step_batch()                                        │
│       ▼                                                          │
│  MateRolloutAdapter ──► collect_step_rollouts()                  │
│       │                                                          │
│       ├── AgentPipeConfig  (角色-模型-URL 映射)                   │
│       │                                                          │
│       └── parallel_rollout / tree_rollout                        │
│                │                                                 │
│          ┌─────▼──────────────────────────────────┐             │
│          │         Agent Trajectory Engine         │             │
│          │                                         │             │
│          │  ModelMonitor (LLM 流量代理)             │             │
│          │       │  拦截并记录 LLM 请求/响应         │             │
│          │       ▼                                 │             │
│          │  MASLauncher ──► 子进程                 │             │
│          │                   akg_rl_entry.py       │             │
│          │                       │                 │             │
│          │                  LangGraphTask          │             │
│          │                  ┌──────────────────┐  │             │
│          │                  │ kernel_designer   │  │             │
│          │                  │ (冻结, 分析算子)   │  │             │
│          │                  │ kernel_gen        │  │             │
│          │                  │ (训练目标)         │  │             │
│          │                  └──────────────────┘  │             │
│          │                                         │             │
│          │  EpisodeTrajectory (按角色存储轨迹)       │             │
│          └─────────────────────────────────────────┘             │
│                │                                                  │
│       AKGKernelRewardProvider                                    │
│                │  compute(trajectory)                            │
│                ▼                                                  │
│          KernelVerifier (执行并验证生成的 kernel)                  │
│                │                                                  │
│          最终奖励 → GRPO 训练更新                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件

### 4.1 数据加载：`KernelBenchLoader`

**文件**：`orchrl_glue/kernelbench_loader.py`

负责从 KernelBench 数据集加载任务，提供滚动批次供训练使用。

```python
loader = KernelBenchLoader(
    dataset_dir="/path/to/KernelBench",
    level="level1",   # 任务难度级别
    shuffle=True,
    seed=42,
)
batch = loader.get_step_batch(step_idx=0, batch_size=8)
```

每条数据的格式：
```python
{
    "prompt": '{"op_name": "1_relu", "task_desc": "..."}',  # JSON 字符串
    "raw": {"op_name": "1_relu", "task_desc": "...", "level": "level1"},
}
```

**批次滚动策略**：`offset = (step_idx × batch_size) % dataset_size`，循环遍历数据集。

---

### 4.2 核心适配器：`MateRolloutAdapter`

**文件**：`orchrl/trainer/mate_rollout_adapter.py`

OrchRL 的训练-推理接口层，将 Agent 工作流适配到 RL 训练循环。

**初始化关键参数**：
```python
adapter = MateRolloutAdapter(
    config={
        "mas_command_template": "python akg_rl_entry.py --config {config_path} --task {prompt}",
        "roles": ["kernel_designer", "kernel_gen"],
        "batch_size": 8,
        "n_samples_per_prompt": 4,       # 每个 prompt 采样 4 次（用于 GRPO）
        "max_concurrent_episodes": 8,
        "timeout": 300,
    },
    prompt_loader=loader,
    reward_provider=reward_provider,
    server_address_dict={
        "kernel_designer": "http://localhost:9000/v1",
        "kernel_gen": "http://localhost:8000/v1",  # 被训练的模型
    },
    role_policy_mapping={
        "kernel_designer": "kernel_designer",
        "kernel_gen": "kernel_gen",
    },
)
```

**每步收集流程** (`collect_step_rollouts`)：
1. 从 `KernelBenchLoader` 获取一批任务
2. 展开为 `batch_size × n_samples_per_prompt` 个并发 job
3. 通过 `asyncio.Semaphore` 控制最大并发数
4. 异步等待所有 episode 完成，返回带奖励的轨迹列表

---

### 4.3 配置注入：`MASLauncher`

**文件**：`orchrl/agent_trajectory_engine/launcher.py`

负责为每个 episode 准备配置文件并启动 Agent 子进程。

**关键步骤**：
```
1. prepare_config()
   ├── 深拷贝 config_template（akg_config_template.yaml）
   ├── 将 ModelMonitor 的 base_url 注入到 agents.*.llm.base_url
   └── 写入临时文件 /tmp/trajectory_mas_xxxxx.yaml

2. launch(command)
   └── subprocess.Popen(shell=True, start_new_session=True)
       执行: python akg_rl_entry.py --config /tmp/... --task {...}
```

**配置模板** (`configs/akg_config_template.yaml`)：
```yaml
agents:
  kernel_gen:
    model: kernel_gen
    llm:
      base_url: ""      # ← 运行时注入 ModelMonitor URL
      api_key: "dummy"
  kernel_designer:
    model: kernel_designer
    llm:
      base_url: ""      # ← 运行时注入 ModelMonitor URL
      api_key: "dummy"
task:
  framework: torch
  backend: cuda
  arch: a100
  dsl: triton_cuda
  max_iterations: 5
```

---

### 4.4 Agent 子进程：`akg_rl_entry.py`

**文件**：`mas_entry/akg_rl_entry.py`

子进程是 Agent 工作流的实际执行者，使用 `akg_agents`（LangGraph）运行 `coder_only` 工作流，单模型模式通过 `AKG_AGENTS_BASE_URL` 路由到 ModelMonitor。

**`akg_rl_entry.py` 执行逻辑**：
```python
def run(config_path, task_json):
    # 1. 加载 YAML 配置（含 ModelMonitor URL）
    config = yaml.safe_load(open(config_path))

    # 2. 设置环境变量，将 LLM 调用路由到 ModelMonitor
    os.environ["AKG_AGENTS_DESIGNER_BASE_URL"] = designer_base_url
    os.environ["AKG_AGENTS_CODER_BASE_URL"] = gen_base_url

    # 3. 构建 LangGraphTask
    task = LangGraphTask(op_name, task_desc, backend="cuda", dsl="triton_cuda", ...)

    # 4. 运行多 Agent 工作流（最多 max_iterations 轮）
    op_name_out, success, final_state = asyncio.run(task.run())

    return 0  # 始终返回 0！
```

> **重要设计**：子进程**始终 exit 0**。OrchRL 会丢弃 exit 非 0 的 episode，而错误的 kernel 作为负样本对 GRPO 训练同样重要，不能丢弃。

---

### 4.5 奖励计算：`AKGKernelRewardProvider`

**文件**：`orchrl_glue/akg_kernel_reward.py`

实现 OrchRL 的 `RewardProvider` 协议，计算多目标奖励。

**奖励公式**：

$$reward = \alpha \cdot r_{correct} + \beta \cdot r_{perf} \cdot r_{correct} + \gamma \cdot r_{iter}$$

| 分量 | 计算方式 | 默认权重 |
|------|----------|---------|
| $r_{correct}$ | `KernelVerifier` 验证通过 → 1.0，否则 → 0.0 | α = 1.0 |
| $r_{perf}$ | 相对 PyTorch 基准的加速比（Phase 2，默认关闭） | β = 0.3 |
| $r_{iter}$ | `max(0, 1 - n_turns / max_turns)`，鼓励少轮完成 | γ = 0.1 |

**计算过程**：
```python
def compute(self, trajectory: EpisodeTrajectory) -> dict:
    # 1. 从 kernel_gen 最后一轮响应中提取代码块
    last_response = trajectory.agent_trajectories["kernel_gen"][-1].response_text
    code = extract_code(last_response)  # 提取 ```python ... ``` 内容

    # 2. 运行 KernelVerifier 验证正确性
    r_correct = self._compute_correctness(code)

    # 3. 计算效率奖励
    n_turns = len(trajectory.agent_trajectories["kernel_gen"])
    r_iter = max(0.0, 1.0 - n_turns / self.max_turns)

    # 4. 合并奖励
    return {"final_reward": α*r_correct + β*r_perf*r_correct + γ*r_iter}
```

---

## 5. 完整执行流程

```
训练步骤 step_idx
      │
      ▼
[1] KernelBenchLoader.get_step_batch(step_idx, batch_size=8)
      │  返回 8 个 PyTorch 算子任务
      │
      ▼
[2] 展开为 8×4=32 个并发 jobs（n_samples_per_prompt=4，GRPO 需要多样本）
      │
      ▼
[3] asyncio.Semaphore(8) 控制并发，对每个 job：
      │
      ▼
[4] MASLauncher.prepare_config()
      │  将 ModelMonitor URL 注入配置模板 → 写入临时 YAML
      │
      ▼
[5] MASLauncher.launch(command)
      │  启动子进程: akg_rl_entry.py --config /tmp/... --task {...}
      │
      ▼
[6] 子进程内部（LangGraphTask）：
      │
      ├── kernel_designer 分析算子，生成设计方案
      │       └── LLM 调用 → ModelMonitor 拦截 → 转发到 designer vLLM
      │
      └── kernel_gen 根据方案生成 Triton kernel（最多 5 轮迭代）
              └── LLM 调用 → ModelMonitor 拦截 → 转发到 gen vLLM（被训练模型）
      │
      ▼
[7] ModelMonitor 收集完整对话轨迹 → EpisodeTrajectory
      │  按角色分类：{"kernel_designer": [...turns], "kernel_gen": [...turns]}
      │
      ▼
[8] AKGKernelRewardProvider.compute(trajectory)
      │
      ├── 提取 kernel_gen 最后响应中的 Python 代码块
      ├── KernelVerifier 实际运行验证（在 CUDA GPU 上执行）
      └── 计算: reward = r_correct + 0.3×r_perf×r_correct + 0.1×r_iter
      │
      ▼
[9] 返回 32 个 EpisodeResult（含 final_reward）
      │
      ▼
[10] GRPO 算法利用多样本奖励计算策略梯度 → 更新 kernel_gen 模型权重
```

---

## 6. 快速启动

### 前置条件

1. 启动 vLLM 推理服务（被训练模型）：
   ```bash
   vllm serve Qwen/Qwen2.5-Coder-7B --port 8000
   ```

2. 启动 designer 模型服务（冻结）：
   ```bash
   vllm serve Qwen/Qwen2.5-Coder-7B --port 9000
   ```

3. 下载 KernelBench 数据集

### 运行 rollout 收集

```bash
python examples/akg_kernel_gen/run_akg_rollout.py \
    --kernelbench-dir /path/to/KernelBench \
    --gen-vllm-url http://localhost:8000/v1 \
    --gen-model Qwen/Qwen2.5-Coder-7B \
    --designer-vllm-url http://localhost:9000/v1 \
    --designer-model Qwen/Qwen2.5-Coder-7B \
    --steps 10
```

---

## 7. 关键设计决策

| 设计 | 原因 |
|------|------|
| **子进程始终 exit 0** | GRPO 需要负样本；非零退出码会导致 episode 被丢弃，损失训练信号 |
| **ModelMonitor 透明代理** | 通过替换 `base_url` 实现无侵入式轨迹捕获，无需修改 `akg_agents` 代码 |
| **奖励在主进程计算** | 防止子进程的 verifier 结果被 Agent 内部逻辑干扰，保证奖励计算的独立性 |
| **配置模板 + 运行时注入** | 每个 episode 独立配置文件，支持并发执行互不干扰 |
| **`n_samples_per_prompt=4`** | GRPO 算法需要同一 prompt 的多个采样来估计相对优势 |
