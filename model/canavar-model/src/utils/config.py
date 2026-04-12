"""
YAML configuration loader with inheritance support.
Configs can specify `inherits: "path/to/parent.yaml"` to inherit defaults.
"""
import yaml
import os


def load_config(config_path: str) -> dict:
    """
    Load a YAML config file. If it has an 'inherits' key, recursively load
    the parent config and merge (child overrides parent).

    The 'inherits' path is resolved in this order:
    1. As-is (absolute path or relative to CWD)
    2. Relative to the config file's own directory
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'inherits' in config:
        parent_path = config.pop('inherits')
        if not os.path.isabs(parent_path):
            # Try CWD-relative first (most intuitive for "configs/base.yaml")
            if not os.path.exists(parent_path):
                # Fall back to config-file-directory-relative
                parent_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), parent_path)
        parent_config = load_config(parent_path)
        config = deep_merge(parent_config, config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_device(config: dict):
    """Get the torch.device from config. 'auto' picks GPU if available."""
    import torch
    device_str = config.get('project', {}).get('device', 'auto')
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)
