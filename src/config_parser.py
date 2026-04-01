import argparse
import logging
from typing import Any, Dict, List, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} must parse to a mapping.")
    return config


def _parse_bool(raw_value: str) -> bool:
    lowered = raw_value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from '{raw_value}'.")


def _coerce_override(raw_value: str, template_value: Any) -> Any:
    if isinstance(template_value, bool):
        return _parse_bool(raw_value)
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(raw_value)
    if isinstance(template_value, float):
        return float(raw_value)
    if isinstance(template_value, list):
        parsed = yaml.safe_load(raw_value)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a list override, got: {raw_value}")
        return parsed
    if isinstance(template_value, dict):
        parsed = yaml.safe_load(raw_value)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a mapping override, got: {raw_value}")
        return parsed
    if template_value is None:
        return yaml.safe_load(raw_value)
    return raw_value


def parse_args_with_config(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse ``--config`` and apply CLI overrides safely.

    Example:
        python train.py --config cfg.yaml --learning_rate 0.0001 --use_flash_attention false
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, remaining_argv = parser.parse_known_args(argv)

    config = load_config(args.config)

    parser_override = argparse.ArgumentParser()
    for key in config:
        parser_override.add_argument(f"--{key}", dest=key, default=argparse.SUPPRESS)

    overrides = vars(parser_override.parse_args(remaining_argv))
    final_config = dict(config)

    for key, raw_value in overrides.items():
        final_config[key] = _coerce_override(raw_value, config.get(key))

    logging.info("Loaded Configuration: %s", final_config)
    return final_config
