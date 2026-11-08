"""All settings of the pipeline in one place, so every run is reproducible."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("biodegradable", "cardboard", "glass", "metal", "paper", "plastic")


@dataclass
class Config:
    # data and output folders
    data_dir: Path = ROOT / "GarbageDataset" / "GarbageDataset"
    analysis_dir: Path = ROOT / "analysis"
    split_dir: Path = ROOT / "splits"
    results_dir: Path = ROOT / "results"

    # split (Task 1)
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    dup_bits: int = 6           # images whose 256-bit dHash differs in <= 6 bits are near-duplicates

    # model and training (Task 1 design, Task 2 training)
    model: str = "yolo26s.pt"   # won the pilot against yolo11s.pt (results/model_comparison.json)
    imgsz: int = 640            # images are 416 px; upsampling enlarges the many small objects
    epochs: int = 60
    patience: int = 15          # early stopping on validation mAP
    batch: int = 16             # fits the 8 GB GPU
    workers: int = 4            # data-loading processes
    max_det: int = 500          # densest image has 358 objects; YOLO's default limit is 300
    augment: dict = field(default_factory=lambda: dict(
        fliplr=0.5, flipud=0.5, degrees=10.0,          # waste has no fixed orientation
        mosaic=1.0, close_mosaic=10,                   # more objects and context per batch
        scale=0.5, translate=0.1,                      # many object sizes, breaks the centre bias
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4))            # lighting / colour of web images

    # evaluation and error analysis (Task 2, Task 3)
    iou_match: float = 0.5      # a detection is correct at IoU >= 0.5
    iou_poor: float = 0.1       # right class but 0.1 <= IoU < 0.5 = imprecise box
    examples: int = 3           # example images per case
    run_name: str = ""          # default: <model>_<imgsz>_<epochs>ep
    image: str = ""             # image or folder for the demo stage (`predict`)

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def data_yaml(self) -> Path:
        return self.split_dir / "garbage.yaml"

    @property
    def run(self) -> str:
        return self.run_name or f"{Path(self.model).stem}_{self.imgsz}_{self.epochs}ep"

    @property
    def weights(self) -> Path:
        return self.results_dir / "train" / self.run / "weights" / "best.pt"
