"""Demo: run the trained detector on any image (or folder of images) and save the picture with its boxes."""
import os
import sys
from collections import Counter
from pathlib import Path

from .config import CLASSES, Config
from .evaluation import threshold
from .training import head_options


def predict_image(cfg: Config) -> dict:
    """Detect waste in cfg.image; annotated copies go to results/demo/ and a single image opens automatically."""
    from ultralytics import YOLO
    source = Path(cfg.image)
    if not source.exists():
        raise FileNotFoundError(f"No image or folder at {source}")
    results = YOLO(str(cfg.weights)).predict(str(source), imgsz=cfg.imgsz, conf=threshold(cfg), max_det=cfg.max_det,
                                             save=True, project=str(cfg.results_dir), name="demo", exist_ok=True,
                                             verbose=False, **head_options(cfg.model))
    summary = {}
    for r in results:
        found = Counter(CLASSES[c] for c in r.boxes.cls.int().tolist())
        summary[Path(r.path).name] = {"objects": sum(found.values()), **dict(found.most_common())}
    saved = Path(results[0].save_dir)
    if len(results) == 1 and sys.platform == "win32":               # open the picture in the Windows viewer
        os.startfile(saved / Path(results[0].path).name)
    return {"saved_in": str(saved), "detections": summary}
