from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


def load_config(config_path):
    """Load CONFIG dictionary from a Python config file."""
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = spec_from_file_location("r2mer_config", str(config_path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config file: {config_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise AttributeError(
            f"Config file does not define `CONFIG`: {config_path}"
        )

    config = module.CONFIG

    if not isinstance(config, dict):
        raise TypeError(
            f"`CONFIG` must be a dict, got {type(config).__name__}"
        )

    return config