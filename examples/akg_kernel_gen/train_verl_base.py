"""Single-model base PPO training entry point for AKG kernel generation.

This script runs the maintained AKG example training path using verl RayPPOTrainer
and orchrl.agent_trajectory_engine for external rollout collection.

Usage:
    python -m examples.akg_kernel_gen.train_verl_base \
        --config-path configs --config-name kernel_gen_level1_verl_base \
        "hydra.searchpath=[file:///path/to/OrchRL/orchrl/config]"
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, DictConfig, open_dict
from tqdm import tqdm

import hydra

os.environ["PYTHONUNBUFFERED"] = "1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch construction helpers for the verl training loop
# ---------------------------------------------------------------------------

def _build_tensor_batch(batch, tokenizer, max_prompt_length: int, max_response_length: int):
    """Construct input_ids / attention_mask / position_ids / token_level_scores on batch."""
    from verl.utils.torch_functional import pad_sequence_to_length

    pad_id = tokenizer.pad_token_id or 0

    # Prompts: left-padded
    prompts_batch = torch.nn.utils.rnn.pad_sequence(
        [torch.flip(t, dims=[0]) for t in batch.batch["prompts"]],
        batch_first=True, padding_value=pad_id,
    ).flip(dims=[1])

    # Responses: right-padded
    responses_batch = torch.nn.utils.rnn.pad_sequence(
        list(batch.batch["responses"]),
        batch_first=True, padding_value=pad_id,
    )

    if "response_mask" in batch.batch.keys():
        response_mask_batch = torch.nn.utils.rnn.pad_sequence(
            list(batch.batch["response_mask"]),
            batch_first=True, padding_value=0,
        )
    else:
        response_mask_batch = None

    # Pad to fixed lengths
    prompts_batch = pad_sequence_to_length(
        prompts_batch, max_prompt_length, pad_id, left_pad=True
    )
    responses_batch = pad_sequence_to_length(
        responses_batch, max_response_length, pad_id, left_pad=False
    )
    if response_mask_batch is not None:
        response_mask_batch = pad_sequence_to_length(
            response_mask_batch, max_response_length, 0, left_pad=False
        )

    input_ids_batch = torch.cat([prompts_batch, responses_batch], dim=1)
    attention_mask_batch = (input_ids_batch != pad_id).long()
    position_ids = (torch.cumsum(attention_mask_batch, dim=1) - 1) * attention_mask_batch

    batch.batch["prompts"] = prompts_batch
    batch.batch["responses"] = responses_batch
    batch.batch["input_ids"] = input_ids_batch
    batch.batch["attention_mask"] = attention_mask_batch
    batch.batch["position_ids"] = position_ids

    if response_mask_batch is None:
        response_mask_batch = (responses_batch != pad_id).long()
    batch.batch["response_mask"] = response_mask_batch

    batch.meta_info["global_token_num"] = (
        torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    )

    # Token-level rewards: place scalar reward at the last valid response token
    reward_tensor = torch.zeros_like(responses_batch, dtype=torch.float32)
    valid_counts = response_mask_batch.sum(dim=-1)
    valid_mask = valid_counts > 0
    if valid_mask.any():
        valid_indices = torch.where(valid_mask)[0]
        last_positions = valid_counts[valid_indices] - 1
        rewards = torch.tensor(
            [float(batch.non_tensor_batch["reward"][i]) for i in valid_indices.tolist()],
            dtype=torch.float32,
        )
        reward_tensor[valid_indices, last_positions] = rewards

    batch.batch["token_level_scores"] = reward_tensor
    batch.batch["token_level_rewards"] = reward_tensor
    return batch


def _pad_to_world_size(batch, world_size: int):
    from verl.protocol import pad_dataproto_to_divisor
    if world_size > 1:
        batch, _ = pad_dataproto_to_divisor(batch, world_size)
    return batch


# ---------------------------------------------------------------------------
# Core training function (runs inside Ray remote task)
# ---------------------------------------------------------------------------

def _merge_verl_defaults(ppo_config: DictConfig) -> DictConfig:
    """Merge our sparse ppo_trainer_config with verl's full generated defaults.

    This ensures every key that verl internals access (use_kl_loss, profiler,
    checkpoint_engine, etc.) exists, while our overrides take precedence.
    """
    import importlib.resources
    import pathlib
    import re

    # Use installed verl defaults (works regardless of source tree layout)
    try:
        import verl.trainer.config as _verl_cfg_pkg
        verl_defaults_path = pathlib.Path(_verl_cfg_pkg.__file__).parent / "_generated_ppo_trainer.yaml"
    except Exception:
        verl_defaults_path = pathlib.Path(
            "/usr/local/lib/python3.12/dist-packages/verl/trainer/config/_generated_ppo_trainer.yaml"
        )

    base = OmegaConf.load(verl_defaults_path)
    base_plain = OmegaConf.to_container(base, resolve=False)
    user_plain = OmegaConf.to_container(ppo_config, resolve=False)

    # Pattern: ${oc.select:some.path, default_value}
    _OC_SELECT_RE = re.compile(r"^\$\{oc\.select:[^,}]+,([^}]+)\}$")

    def _resolve_interpolation(s: str):
        """Extract default value from ${oc.select:..., default} or return None."""
        m = _OC_SELECT_RE.match(s.strip())
        if not m:
            return None
        default_str = m.group(1).strip()
        if default_str == "null":
            return None
        if default_str.lower() == "true":
            return True
        if default_str.lower() == "false":
            return False
        if re.match(r"^-?\d+$", default_str):
            return int(default_str)
        try:
            return float(default_str)
        except ValueError:
            pass
        if default_str.startswith("[") and default_str.endswith("]"):
            return []
        return default_str  # string default

    def _strip_interpolations(d):
        """Replace ${...} strings with their default values where available."""
        if isinstance(d, dict):
            return {k: _strip_interpolations(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_strip_interpolations(v) for v in d]
        if isinstance(d, str) and "${" in d:
            return _resolve_interpolation(d)
        return d

    def _deep_merge(a: dict, b: dict) -> dict:
        """Merge b into a (b wins on conflicts)."""
        merged = dict(a)
        for k, v in b.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = _deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    base_clean = _strip_interpolations(base_plain)
    merged = _deep_merge(base_clean, user_plain)
    return OmegaConf.create(merged)


def train_single_model(config: DictConfig) -> None:
    """Initialize and run single-model GRPO/PPO training loop."""
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.trainer.ppo.ray_trainer import (
        RayPPOTrainer, ResourcePoolManager, Role,
        compute_advantage, compute_data_metrics, compute_timing_metrics, reduce_metrics,
    )
    from verl.trainer.ppo import core_algos
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    from verl.protocol import pad_dataproto_to_divisor

    from orchrl.trainer.mate_rollout_adapter import MateRolloutAdapter
    from orchrl.trainer.mate_dataproto_adapter import episodes_to_policy_batches
    from orchrl.trainer.mate_prompt_loader import MatePromptLoader
    from orchrl.trainer.mate_reward_bridge import build_reward_provider
    from orchrl.trainer.mate_config import validate_mate_config
    from orchrl.utils.served_model_name import resolve_policy_server_name
    from orchrl.utils.performance import simple_timer
    from verl.utils.tracking import Tracking

    OmegaConf.resolve(config)

    # ------------------------------------------------------------------
    # Model + tokenizer
    # ------------------------------------------------------------------
    model_cfg = config.model
    model_path = copy_local_path_from_hdfs(model_cfg.path)
    model_name = model_cfg.name
    trust_remote_code = config.resource.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    n_gpus_per_node = config.resource.get("n_gpus_per_node", 8)
    nnodes = config.resource.get("nnodes", 1)

    # ------------------------------------------------------------------
    # Resource pool — single model uses all GPUs
    # ------------------------------------------------------------------
    global_pool_id = "global_pool_0"
    resource_pool_spec = {global_pool_id: [n_gpus_per_node] * nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )
    resource_pool_manager.create_resource_pool()

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
    }

    # ------------------------------------------------------------------
    # Build RayPPOTrainer
    # ------------------------------------------------------------------
    ppo_config = _merge_verl_defaults(config.ppo_trainer_config)
    with open_dict(ppo_config):
        ppo_config.data["train_batch_size"] = config.training.train_batch_size

    ppo_trainer = RayPPOTrainer(
        config=ppo_config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
    )
    ppo_trainer.global_steps = 0
    ppo_trainer.init_workers()

    # ------------------------------------------------------------------
    # Collect server addresses exposed by vLLM rollout
    # ------------------------------------------------------------------
    rollout_engine = ppo_trainer.async_rollout_manager
    server_addresses = getattr(rollout_engine, "server_addresses", [])
    server_address_dict = {model_name: server_addresses}
    policy_server_name = resolve_policy_server_name(model_name, ppo_config)
    policy_server_name_mapping = {model_name: policy_server_name}

    # ------------------------------------------------------------------
    # MateRolloutAdapter — uses agent_trajectory_engine under the hood
    # ------------------------------------------------------------------
    mate_cfg_raw = OmegaConf.to_container(config.training.mate, resolve=True)
    mate_config = validate_mate_config(config.training.mate, {
        agent: model_name for agent in mate_cfg_raw.get("roles", [model_name])
    })

    prompt_loader_cfg = mate_config.get("prompt_loader") or mate_config.get("prompt_source", {})
    reward_cfg = mate_config.get("reward", {})

    prompt_loader = MatePromptLoader(
        source_type=prompt_loader_cfg.get("source_type", prompt_loader_cfg.get("type")),
        path=prompt_loader_cfg["path"],
        prompt_keys=list(prompt_loader_cfg["prompt_keys"]),
        expected_keys=list(prompt_loader_cfg.get("expected_keys", [])),
    )
    reward_provider = build_reward_provider(reward_cfg)

    mate_adapter = MateRolloutAdapter(
        config=mate_config,
        prompt_loader=prompt_loader,
        reward_provider=reward_provider,
        server_address_dict=server_address_dict,
        role_policy_mapping=mate_config["role_policy_mapping"],
        policy_server_name_mapping=policy_server_name_mapping,
        tokenizer_dict={model_name: tokenizer},
    )

    max_prompt_length = config.training.get("max_prompt_length", ppo_config.data.max_prompt_length)
    max_response_length = config.training.get("max_response_length", ppo_config.data.max_response_length)
    credit_assignment = mate_config.get("credit_assignment", "all_turns")
    role_names = list(mate_config["role_policy_mapping"].keys())

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    resolved_config = OmegaConf.to_container(config, resolve=True)
    resolved_config.setdefault("trainer", resolved_config.get("training", {}))
    tracker = Tracking(
        project_name=config.training.project_name,
        experiment_name=config.training.experiment_name,
        default_backend=config.training.logger,
        config=resolved_config,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_steps = config.training.total_training_steps
    global_steps = 0

    # Resume from checkpoint
    loaded_step = ppo_trainer._load_checkpoint()
    if loaded_step > 0:
        global_steps = loaded_step
        ppo_trainer.global_steps = global_steps
        logger.info("Resumed from step %d", global_steps)

    progress_bar = tqdm(range(total_steps), desc="Training", position=0, leave=True)
    progress_bar.update(global_steps)

    while global_steps < total_steps:
        progress_bar.set_description(f"Step {global_steps}")
        metrics: dict = {}
        timing_raw: dict = {}

        with simple_timer("step", timing_raw):
            # --- Rollout collection ---
            with simple_timer("collect_trajectory", timing_raw):
                for _, checkpoint_manager in [("model", ppo_trainer.checkpoint_manager)]:
                    if checkpoint_manager is not None:
                        checkpoint_manager.update_weights()
                try:
                    episodes = asyncio.run(
                        mate_adapter.collect_step_rollouts(step_idx=global_steps)
                    )
                finally:
                    if ppo_trainer.checkpoint_manager is not None:
                        ppo_trainer.checkpoint_manager.sleep_replicas()

            if not episodes:
                logger.warning("Step %d: no episodes collected, skipping.", global_steps)
                global_steps += 1
                ppo_trainer.global_steps = global_steps
                progress_bar.update(1)
                continue

            # --- Convert episodes → DataProto ---
            policy_batches = episodes_to_policy_batches(
                episodes=episodes,
                tokenizer_dict={model_name: tokenizer},
                role_policy_mapping=mate_config["role_policy_mapping"],
                role_index_mapping={role: idx for idx, role in enumerate(role_names)},
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                credit_assignment=credit_assignment,
            )

            batch = policy_batches.get(model_name)
            if batch is None:
                logger.warning("Step %d: no batch for model %s, skipping.", global_steps, model_name)
                global_steps += 1
                ppo_trainer.global_steps = global_steps
                progress_bar.update(1)
                continue

            # --- Pad to DP world size ---
            dp_world_size = ppo_trainer.actor_rollout_wg.world_size
            batch = _pad_to_world_size(batch, dp_world_size)

            # --- Build tensor fields ---
            batch.meta_info.setdefault("metrics", {})
            batch = _build_tensor_batch(batch, tokenizer, max_prompt_length, max_response_length)

            # --- PPO update ---
            with simple_timer("update_parameters", timing_raw):
                # old log probs
                if dp_world_size > 1:
                    batch, _ = pad_dataproto_to_divisor(batch, dp_world_size)
                old_log_prob = ppo_trainer.actor_rollout_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)

                # ref log probs (needed for KL)
                if ppo_trainer.use_reference_policy or ppo_config.algorithm.use_kl_in_reward:
                    if not ppo_trainer.ref_in_actor:
                        ref_log_prob = ppo_trainer.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = ppo_trainer.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

                # critic values
                if ppo_trainer.use_critic:
                    values = ppo_trainer.critic_wg.compute_values(batch)
                    batch = batch.union(values)

                # advantages
                batch = compute_advantage(
                    batch,
                    adv_estimator=ppo_config.algorithm.adv_estimator,
                    gamma=ppo_config.algorithm.gamma,
                    lam=ppo_config.algorithm.lam,
                    num_repeat=ppo_config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=ppo_config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=ppo_config.algorithm,
                )

                # critic update
                if ppo_trainer.use_critic:
                    critic_output = ppo_trainer.critic_wg.update_critic(batch)
                    batch.meta_info["metrics"].update(
                        reduce_metrics(critic_output.meta_info["metrics"])
                    )

                # actor update
                batch.meta_info["multi_turn"] = ppo_config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = ppo_trainer.actor_rollout_wg.update_actor(batch)
                batch.meta_info["metrics"].update(
                    reduce_metrics(actor_output.meta_info["metrics"])
                )

        # --- Metrics ---
        metrics.update(batch.meta_info.get("metrics", {}))
        metrics.update(compute_data_metrics(batch=batch, use_critic=ppo_trainer.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        metrics["training/global_step"] = global_steps

        global_steps += 1
        ppo_trainer.global_steps = global_steps
        progress_bar.update(1)

        try:
            tracker.log(data=metrics, step=global_steps)
        except Exception as exc:
            logger.warning("Failed to log metrics: %s", exc)

    progress_bar.close()
    logger.info("Training complete after %d steps.", global_steps)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="kernel_gen_level1_verl_base", version_base=None)
def main(config: DictConfig) -> None:
    from orchrl.utils.ray_utils import init_ray_with_temp_dirs
    from orchrl.utils.clean_up import cleanup_ray_runtime, install_cleanup_hooks

    install_cleanup_hooks()

    try:
        init_ray_with_temp_dirs(config)

        num_cpus = max(8, int(ray.cluster_resources()["CPU"] * 0.1))
        remote_train = ray.remote(num_cpus=num_cpus)(train_single_model)
        ray.get(remote_train.remote(config))
    finally:
        cleanup_ray_runtime()


if __name__ == "__main__":
    main()
