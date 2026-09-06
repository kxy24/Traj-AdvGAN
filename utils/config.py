from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_split(data_cfg: dict, split: str, data_root: str | None = None) -> str:
    """Resolve a dataset split directory from a public YAML plus an optional local root override."""
    if split not in data_cfg:
        raise KeyError(f"Split '{split}' is not defined in the dataset config")
    root = Path(data_root if data_root is not None else data_cfg.get("path", ".")).expanduser()
    split_value = Path(str(data_cfg[split])).expanduser()
    if split_value.is_absolute():
        return str(split_value)
    return str(root / split_value)
