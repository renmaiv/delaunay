"""
Configuration Loader

Loads configuration from YAML with environment variable substitution
"""

import os
import re
import yaml
from typing import Any, Dict
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Substitute environment variables
        config = self._substitute_env_vars(config)

        return config

    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in config

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_string(obj)
        else:
            return obj

    def _substitute_env_var_string(self, value: str) -> str:
        """Substitute environment variables in a string"""
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                # Variable not found and no default
                return match.group(0)  # Return original

        return re.sub(pattern, replacer, value)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path

        Args:
            key_path: Dot-separated path (e.g., "llm.provider")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get('llm', {})

    def get_bert_config(self) -> Dict[str, Any]:
        """Get BERT configuration"""
        return self.config.get('bert', {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.config.get('api', {})

    def get_judge_mode(self) -> str:
        """Get judge mode"""
        return self.config.get('judge_mode', 'bert')

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()


# Global config instance
_config = None


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Load configuration

    Args:
        config_path: Path to config file

    Returns:
        ConfigLoader instance
    """
    global _config
    _config = ConfigLoader(config_path)
    return _config


def get_config() -> ConfigLoader:
    """
    Get global config instance

    Returns:
        ConfigLoader instance
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    # Demo
    print("=" * 70)
    print("CONFIG LOADER DEMO")
    print("=" * 70)

    # Load config
    config = load_config()

    print(f"\nJudge Mode: {config.get_judge_mode()}")
    print(f"LLM Provider: {config.get('llm.provider')}")
    print(f"BERT Model: {config.get('bert.model_name')}")
    print(f"API Port: {config.get('api.port')}")
    print(f"Logging Level: {config.get('logging.level')}")

    print("\n[BERT Config]")
    bert_config = config.get_bert_config()
    for key, value in bert_config.items():
        print(f"  {key}: {value}")

    print("\n[API Config]")
    api_config = config.get_api_config()
    for key, value in api_config.items():
        print(f"  {key}: {value}")
