from .tt100k import TT100KDataset
from .cctsdb import CCTSDBDataset


def build_dataset(name: str, image_root: str, image_size: int = 640):
    key = name.lower().replace("-", "").replace("_", "")
    if key in {"tt100k", "tt100k2016"}:
        return TT100KDataset(image_root, image_size=image_size)
    if key in {"cctsdb", "cctsdb2021"}:
        return CCTSDBDataset(image_root, image_size=image_size)
    raise ValueError(f"Unsupported dataset: {name}")


__all__ = ["TT100KDataset", "CCTSDBDataset", "build_dataset"]
