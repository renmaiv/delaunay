"""Configuration loader: YAML with ${VAR} / ${VAR:default} env substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Load and query config.yaml."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return self._substitute_env_vars(config)

    def _substitute_env_vars(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_string(obj)
        return obj

    def _substitute_env_var_string(self, value: str) -> str:
        # ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

        def replacer(match):
            env_value = os.environ.get(match.group(1))
            if env_value is not None:
                return env_value
            if match.group(2) is not None:
                return match.group(2)
            return match.group(0)

        return re.sub(pattern, replacer, value)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a value by dot-separated path, e.g. "judge.provider"."""
        value = self.config
        for key in key_path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_judge_config(self) -> Dict[str, Any]:
        return self.config.get("judge", {}) or {}

    def get_encoders_config(self) -> Dict[str, Any]:
        return self.config.get("encoders", {}) or {}

    def get_analysis_config(self) -> Dict[str, Any]:
        return self.config.get("analysis", {}) or {}

    def get_api_config(self) -> Dict[str, Any]:
        return self.config.get("api", {}) or {}

    def reload(self):
        self.config = self._load_config()


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    return ConfigLoader(config_path)
