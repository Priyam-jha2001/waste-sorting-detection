"""
CV Portfolio 1 - Object detection for automated waste sorting.
Team: Priyam Jha, Dhrumil Desai, Abin Skaria

Usage (one stage per command, in this order):
    python main.py analyze                                  Task 1  dataset analysis      -> analysis/
    python main.py split                                    Task 1  train/val/test split  -> splits/
    python main.py train --model yolo11s.pt --epochs 15     Task 2  pilot run (model selection)
    python main.py train --model yolo26s.pt --epochs 15     Task 2  pilot run (model selection)
    python main.py compare                                  Task 2  pilots compared on validation
    python main.py train                                    Task 2  final training (YOLO26s, 60 epochs)
    python main.py evaluate                                 Task 2  test-set metrics      -> results/test_eval/
    python main.py errors                                   Task 3  examples + statistics -> results/error_analysis/
    python main.py predict --image "path/to/photo.jpg"      Demo    boxes drawn on any image  -> results/demo/
"""
import argparse
import json

from waste.analysis import analyze
from waste.config import Config
from waste.demo import predict_image
from waste.errors import error_analysis
from waste.evaluation import evaluate
from waste.split import make_split
from waste.training import compare, train

STAGES = {"analyze": analyze, "split": make_split, "train": train, "compare": compare,
          "evaluate": evaluate, "errors": error_analysis, "predict": predict_image}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--model", help="e.g. yolo11s.pt (default: yolo26s.pt)")
    parser.add_argument("--epochs", type=int, help="e.g. 15 for a pilot run (default: 60)")
    parser.add_argument("--batch", type=int, help="lower it if the GPU runs out of memory (default: 16)")
    parser.add_argument("--image", help="image or folder for the predict stage")
    args = parser.parse_args()
    if args.stage == "predict" and not args.image:
        parser.error("predict needs --image, e.g. --image my_photo.jpg")

    cfg = Config()
    cfg.model = args.model or cfg.model
    cfg.epochs = args.epochs or cfg.epochs
    cfg.batch = args.batch or cfg.batch
    cfg.image = args.image or ""
    result = STAGES[args.stage](cfg)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":   # needed on Windows: data-loader workers re-import this file
    main()
