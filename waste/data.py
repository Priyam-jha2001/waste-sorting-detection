"""Reading the dataset and the box geometry every other module builds on."""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config


@dataclass
class Sample:
    """One image with its ground truth: boxes as normalised (cx, cy, w, h), one class id per box."""
    image: Path
    boxes: np.ndarray
    classes: np.ndarray
    invalid: int        # malformed or zero-size boxes in the label file
    source: str         # class named in the file name, e.g. "glass" for glass123_jpg.rf....jpg


def load_samples(cfg: Config) -> list[Sample]:
    """Read every image and its YOLO label file (one 'class cx cy w h' line per box)."""
    samples = []
    for img in sorted(cfg.images_dir.glob("*.jp*g")):
        boxes, classes, invalid = [], [], 0
        label = cfg.labels_dir / f"{img.stem}.txt"
        for line in label.read_text().splitlines() if label.exists() else []:
            parts = line.split()
            if not parts:
                continue
            values = [float(v) for v in parts[1:]]
            if len(values) != 4 or not (0 <= values[0] <= 1 and 0 <= values[1] <= 1
                                        and 0 < values[2] <= 1 and 0 < values[3] <= 1):
                invalid += 1
                continue
            boxes.append(values)
            classes.append(int(parts[0]))
        source = re.match(r"([a-z]+)\d+_", img.name)
        samples.append(Sample(img, np.array(boxes).reshape(-1, 4), np.array(classes, dtype=int),
                              invalid, source.group(1) if source else "unknown"))
    return samples


def split_names(cfg: Config, part: str) -> list[str]:
    """Image file names listed in splits/<part>.txt (read line by line: paths contain spaces)."""
    lines = (cfg.split_dir / f"{part}.txt").read_text(encoding="utf-8").splitlines()
    names = [Path(line.strip().replace("\\", "/")).name for line in lines]
    return list(dict.fromkeys(n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))))


def to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised (cx, cy, w, h) -> pixel corners (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.reshape(-1, 4).T
    return np.stack([(cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height], axis=1)


def box_area(xyxy: np.ndarray) -> np.ndarray:
    return (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in a with every box in b."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter + 1e-9)


def size_bucket(area_px: float) -> str:
    """COCO size classes, in pixels of the original 416 px image."""
    return "small" if area_px < 32 ** 2 else "medium" if area_px < 96 ** 2 else "large"


GROUP_ORDER = {k: i for i, k in enumerate(("small", "medium", "large", "1", "2-5", "6-20", ">20"))}


def density_bin(n_objects: int) -> str:
    """How crowded an image is."""
    return "1" if n_objects <= 1 else "2-5" if n_objects <= 5 else "6-20" if n_objects <= 20 else ">20"


@lru_cache(maxsize=None)
def image_domain(path: Path) -> str:
    """'clean background' if >= 80 % of a 5 % border strip is near-white (product photo), else 'scene'."""
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    b = max(2, int(0.05 * min(gray.shape)))
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return "clean background" if (border > 225).mean() >= 0.8 else "scene"


def near_duplicate_groups(samples: list[Sample], max_bits: int) -> np.ndarray:
    """Group id per image; images whose difference hash (dHash) differs in <= max_bits bits share a group."""
    def dhash(path: Path) -> np.ndarray:
        gray = np.asarray(Image.open(path).convert("L").resize((17, 16)), dtype=np.int16)
        return (gray[:, 1:] > gray[:, :-1]).flatten()

    bits = np.stack([dhash(s.image) for s in samples]).astype(np.int16)
    parent = list(range(len(samples)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(samples), 512):                 # compare in blocks to save memory
        dist = (bits[start:start + 512, None, :] != bits[None, :, :]).sum(axis=2)
        for a, b in zip(*np.where(dist <= max_bits)):
            if a + start < b:
                parent[root(a + start)] = root(b)
    return np.array([root(i) for i in range(len(samples))])


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
