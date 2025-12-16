import yaml
import argparse
import logging
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args_with_config() -> Dict[str, Any]:
    """
    Parses CLI arguments.
    Allows specifying a --config YAML file and overriding keys via CLI.
    Example: python train.py --config cfg.yaml --learning_rate 0.0001
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    
    # Parse known args first to get the config path
    args, remaining_argv = parser.parse_known_args()
    
    config = load_config(args.config)
    
    # Add config keys as CLI arguments for overriding
    parser_override = argparse.ArgumentParser()
    for key, value in config.items():
        arg_type = type(value) if value is not None else str
        parser_override.add_argument(f"--{key}", type=arg_type, default=value)
        
    # Re-parse to allow overrides
    args_final = parser_override.parse_args(remaining_argv)
    
    # Convert to dict
    final_config = vars(args_final)
    logging.info(f"Loaded Configuration: {final_config}")
    return final_config