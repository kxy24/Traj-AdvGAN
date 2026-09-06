"""Dataset utilities shared by TT100K and CCTSDB.

Training of Traj-AdvGAN is clean-box/detector driven and only needs images. Standard detector
metrics can still use the original dataset annotations through the detector repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RecursiveImageDataset(Dataset):
    def __init__(self, image_root: str, image_size: int = 640):
        self.root = Path(image_root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Image directory not found: {self.root}")
        self.image_paths: Sequence[Path] = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found under: {self.root}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, str(path)
