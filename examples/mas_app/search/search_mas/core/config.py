from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _expand_env_in_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, "")

    return ENV_PATTERN.sub(replace, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return _expand_env_in_string(value)
    return value


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load yaml config with `${ENV_VAR}` / `$ENV_VAR` expansion."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dict in file: {path}")
    return _expand_env(config)
