# AKG Kernel Gen via OrchRL Agent Trajectory Engine

这个示例的目标是说明如何在 OrchRL 中通过 `agent_trajectory_engine` 接入 `akg_agent`，把外部 agent 的执行过程转换成 OrchRL 可消费的轨迹与奖励，并接到 `verl_base` 训练链路里。

这里默认只保留一条推荐路径：

- 外部 agent 执行由 `akg_agents` 完成。
- 轨迹采集由 `orchrl.agent_trajectory_engine` 完成。
- 训练入口固定为 `examples/akg_kernel_gen/train_verl_base.py`。
- 启动脚本固定为 `examples/akg_kernel_gen/train.sh`。

## 1. 接入思路

这个示例演示的是一条标准的“外部 agent 接入 OrchRL”路径：

1. OrchRL 通过 `MateRolloutAdapter` 发起一次 rollout。
2. `MateRolloutAdapter` 内部调用 `agent_trajectory_engine`。
3. `agent_trajectory_engine` 通过 `MASLauncher` 启动外部入口 [akg_rl_entry.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/mas_entry/akg_rl_entry.py)。
4. `akg_rl_entry.py` 加载 `akg_agents`，执行 AKG 的 agent workflow。
5. 外部 agent 的 LLM 请求经 `ModelMonitor` 转发并记录，交互被整理为 `EpisodeTrajectory`。
6. Reward provider 基于轨迹内容计算 reward。
7. `episodes_to_policy_batches` 将轨迹转换成训练 batch，交给 verl trainer 更新策略。

核心点在于：OrchRL 不需要重写 AKG agent 本身，只需要把“外部 agent 的一次运行”封装成可追踪、可计分、可训练的 episode。

## 2. 关键组件

- 轨迹引擎：`orchrl/agent_trajectory_engine/`
  负责外部 agent 启动、交互采集、轨迹组织。
- AKG 外部入口：[mas_entry/akg_rl_entry.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/mas_entry/akg_rl_entry.py)
  负责把 AKG 的运行方式适配到 OrchRL 的外部 rollout 协议。
- Reward 适配：[orchrl_glue/akg_kernel_reward.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py)
  负责从 `EpisodeTrajectory` 中提取 `kernel_gen` 输出并计算 reward。
- 训练入口：[train_verl_base.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/train_verl_base.py)
  使用 verl `RayPPOTrainer`，rollout 数据来自 `agent_trajectory_engine`。
- 启动脚本：[train.sh](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/train.sh)
  这是本示例唯一保留的训练启动方式。

## 3. 数据流

```text
Prompt / Task
  -> MateRolloutAdapter
  -> orchrl.agent_trajectory_engine
  -> MASLauncher
  -> akg_rl_entry.py
  -> akg_agents
  -> EpisodeTrajectory
  -> AKGKernelRewardProvider
  -> episodes_to_policy_batches
  -> verl RayPPOTrainer
```

其中 `EpisodeTrajectory` 是 OrchRL 接外部 agent 时最关键的中间表示。只要外部系统能被封装成一次 episode，并能产出按 role 组织的交互轨迹，就可以复用同样的训练链路。

## 4. 环境准备

1. 安装 AKG Agents：

```bash
pip install -e /path/to/akg/akg_agents --no-build-isolation
```

2. 安装 OrchRL：

```bash
pip install -e /path/to/OrchRL
```

3. 安装本示例依赖：

```bash
pip install -r examples/akg_kernel_gen/requirements.txt
```

4. 准备 KernelBench 数据：

```bash
cd /path/to/akg/akg_agents
git submodule update --init "thirdparty/KernelBench"
```

数据目录通常为：
`akg_agents/thirdparty/KernelBench/KernelBench/level1/`

5. 按 AKG 要求注册本地 CUDA worker：

```python
from akg_agents.core.worker.manager import register_local_worker
import asyncio

asyncio.run(register_local_worker([0], backend="cuda", arch="a100"))
```

6. 可选：如果你只是单独调试 [run_akg_rollout.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/run_akg_rollout.py)，需要预先准备一个 OpenAI-compatible 模型服务，例如：

```bash
vllm serve Qwen/Qwen2.5-Coder-7B --port 8000
```

默认训练入口 [train.sh](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/train.sh) 走的是 `verl_base` 路径，训练时会由 trainer 初始化自己的 rollout 服务，通常不需要你手动额外启动一个外部 `vllm serve`。

## 5. 如何启动训练

推荐直接使用示例脚本 [train.sh](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/train.sh)：

```bash
cd /path/to/OrchRL
bash examples/akg_kernel_gen/train.sh
```

如果需要显式指定配置：

```bash
cd /path/to/OrchRL
CONFIG_NAME=kernel_gen_level1_verl_base \
bash examples/akg_kernel_gen/train.sh
```

对应的 Python 入口等价于：

```bash
python3 -m examples.akg_kernel_gen.train_verl_base \
  --config-path examples/akg_kernel_gen/configs \
  --config-name kernel_gen_level1_verl_base \
  "hydra.searchpath=[file://$(pwd)/orchrl/config]"
```

## 6. `agent_trajectory_engine` 在这里做什么

相对于把 rollout 逻辑写死在 trainer 里，`agent_trajectory_engine` 提供了一层统一的外部 agent 适配：

- 用统一方式启动外部 agent 进程。
- 记录各个 role 的 prompt / response / token 信息。
- 将一次外部执行整理为 `EpisodeTrajectory`。
- 把 reward 计算与 agent 执行过程解耦。
- 让上层 trainer 不需要感知 `akg_agents` 的内部实现细节。

因此，这个示例的重点不是“如何训练一个普通 LLM”，而是“如何把 AKG 这样的外部 agent 系统接到 OrchRL 的轨迹采集与训练接口上”。

## 7. 相关文件

- 示例配置：[configs/kernel_gen_level1_verl_base.yaml](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/configs/kernel_gen_level1_verl_base.yaml)
- AKG 配置模板：[configs/akg_config_template.yaml](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/configs/akg_config_template.yaml)
- 训练入口：[train_verl_base.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/train_verl_base.py)
- 外部 agent 入口：[mas_entry/akg_rl_entry.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/mas_entry/akg_rl_entry.py)
- Reward 适配：[orchrl_glue/akg_kernel_reward.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py)
- 可选调试脚本：[run_akg_rollout.py](/data1/lmy/agentic-rl/OrchRL/examples/akg_kernel_gen/run_akg_rollout.py)

## 8. 验证建议

如果你是在扩展新的外部 agent 接入，建议先单独验证：

- 外部 agent 进程能正常拉起。
- `ModelMonitor` 能正确拦截 LLM 调用。
- `EpisodeTrajectory` 中包含预期的 role 和 turn。
- reward provider 能从轨迹中读到正确字段。
- 再进入完整训练。
