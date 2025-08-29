# src/utils/config.py
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = 'configs/main_config.yaml') -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration loaded successfully.")
    return config