"""Task 2 - model selection (pilot runs compared on validation) and the final training."""
import logging
from functools import cache

from .config import CLASSES, Config
from .data import save_json
from .split import localize_paths


def head_options(model_name: str) -> dict:
    """
    YOLO26 has an NMS-free head that is capped at 300 detections. Our densest image has 358 objects,
    so for YOLO26 we use its second head with NMS (end2end=False). Other YOLO versions use NMS anyway.
    """
    if "yolo26" not in model_name.lower():
        return {}
    _hide_deprecation_note()
    return {"end2end": False}


@cache
def _hide_deprecation_note() -> None:
    """Ultralytics repeats a harmless note about end2end=False on every call; hide it."""
    logging.getLogger("ultralytics").addFilter(lambda record: "end2end=False' is deprecated" not in record.getMessage())


def train(cfg: Config) -> None:
    """Train on the train split; Ultralytics keeps the epoch with the best validation mAP as best.pt."""
    from ultralytics import YOLO
    localize_paths(cfg)
    YOLO(cfg.model).train(data=str(cfg.data_yaml), imgsz=cfg.imgsz, epochs=cfg.epochs, patience=cfg.patience,
                          batch=cfg.batch, workers=cfg.workers, seed=cfg.seed, deterministic=True, cos_lr=True, max_det=cfg.max_det,
                          project=str(cfg.results_dir / "train"), name=cfg.run, exist_ok=True, **cfg.augment)


def compare(cfg: Config) -> list[dict]:
    """Evaluate every trained run on the VALIDATION split with identical settings (model selection)."""
    from ultralytics import YOLO
    localize_paths(cfg)
    rows = []
    for weights in sorted((cfg.results_dir / "train").glob("*/weights/best.pt")):
        run = weights.parent.parent.name
        box = YOLO(str(weights)).val(data=str(cfg.data_yaml), split="val", imgsz=cfg.imgsz, max_det=cfg.max_det,
                                     plots=False, verbose=False, project=str(cfg.results_dir / "compare"),
                                     name=run, exist_ok=True, **head_options(run)).box
        rows.append({"run": run, "val_mAP50-95": round(float(box.map), 4), "val_mAP50": round(float(box.map50), 4),
                     "AP50-95_per_class": {CLASSES[c]: round(float(box.ap[i]), 4)
                                           for i, c in enumerate(box.ap_class_index)}})
    rows.sort(key=lambda r: -r["val_mAP50-95"])
    save_json(cfg.results_dir / "model_comparison.json", rows)
    return rows
