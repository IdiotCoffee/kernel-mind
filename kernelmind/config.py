import os
import yaml

CONFIG_PATH = os.path.expanduser("~/.kernelmind/config.yaml")

DEFAULT_CONFIG = {
    "inference": {
        "mode": "local",        # "local" or "cloud"
        "api_key": None,
        "model": "gpt-4o-mini"
    }
}

def ensure_config_dir():
    path = os.path.dirname(CONFIG_PATH)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_config():
    """Load config from ~/.kernelmind/config.yaml or create one."""
    ensure_config_dir()

    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_CONFIG

    # Merge missing defaults (so future updates don't crash older installs)
    merged = DEFAULT_CONFIG.copy()
    for key, val in DEFAULT_CONFIG.items():
        if key not in data:
            merged[key] = val
        else:
            # nested merge
            inner = val.copy()
            inner.update(data.get(key, {}))
            merged[key] = inner

    return merged

def save_config(config):
    ensure_config_dir()
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)
