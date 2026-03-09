from __future__ import annotations

import asyncio
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import InferenceBackend
from .collector import TrajectoryCollector
from .datatypes import EpisodeResult, ModelMappingEntry
from .launcher import MASLauncher
from .monitor import ModelMonitor
from .reward import RewardProvider, RewardWorker


@dataclass
class AgentPipeConfig:
    mas_command_template: str
    config_template: dict[str, Any]
    model_mapping: dict[str, ModelMappingEntry]
    timeout: float = 300.0
    monitor_host: str = "127.0.0.1"
    monitor_port: int = 0
    mas_work_dir: str | Path | None = None


class AgentPipe:
    def __init__(self, config: AgentPipeConfig, backend: InferenceBackend) -> None:
        self._config = config
        self._backend = backend
        self._collector = TrajectoryCollector()
        self._reward_worker = RewardWorker()

    async def run(
        self,
        prompt: str,
        reward_provider: RewardProvider,
    ) -> EpisodeResult:
        episode_id = uuid.uuid4().hex
        monitor = ModelMonitor(
            backend=self._backend,
            model_mapping=self._config.model_mapping,
            episode_id=episode_id,
        )
        launcher = MASLauncher(work_dir=self._config.mas_work_dir)
        primary_error: BaseException | None = None

        try:
            port = await monitor.start(
                host=self._config.monitor_host,
                port=self._config.monitor_port,
            )
            monitor_url = f"http://{self._config.monitor_host}:{port}/v1"
            config_path = await asyncio.to_thread(
                launcher.prepare_config,
                config_template=self._config.config_template,
                monitor_url=monitor_url,
                agent_roles=list(self._config.model_mapping.keys()),
            )
            command = self._config.mas_command_template.format(
                config_path=shlex.quote(str(config_path)),
                prompt=shlex.quote(prompt),
            )
            process = await asyncio.to_thread(launcher.launch, command=command)
            exit_code = await asyncio.to_thread(
                launcher.wait,
                process,
                self._config.timeout,
            )
            if exit_code != 0:
                raise RuntimeError(f"MAS process exited with non-zero exit code {exit_code}")

            trajectory = self._collector.build(buffer=monitor.get_buffer(), episode_id=episode_id)
            result = await asyncio.to_thread(
                self._reward_worker.compute,
                trajectory,
                reward_provider,
            )
            result.metadata["exit_code"] = exit_code
            return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            stop_error: Exception | None = None
            cleanup_error: Exception | None = None

            try:
                await monitor.stop()
            except Exception as exc:  # pragma: no cover - exercised via tests
                stop_error = exc

            try:
                launcher.cleanup()
            except Exception as exc:  # pragma: no cover - exercised via tests
                cleanup_error = exc

            if primary_error is None:
                if stop_error is not None:
                    if cleanup_error is not None:
                        stop_error.add_note(f"launcher.cleanup() also failed: {cleanup_error}")
                    raise stop_error
                if cleanup_error is not None:
                    raise cleanup_error
