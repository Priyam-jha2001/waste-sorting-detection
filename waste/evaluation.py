"""Task 2 - evaluation on the test set: standard metrics plus breakdowns by object type."""
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLASSES, Config
from .data import (GROUP_ORDER, Sample, box_area, density_bin, image_domain, iou_matrix, load_samples,
                   save_json, size_bucket, split_names, to_xyxy)
from .plots import bars, plt, save
from .split import localize_paths
from .training import head_options


def predict(cfg: Config, part: str) -> dict:
    """Run the model once over a split and cache every prediction (conf >= 0.01) in results/predictions/."""
    cache = cfg.results_dir / "predictions" / f"{cfg.run}_{part}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO
    localize_paths(cfg)
    model, paths, preds = YOLO(str(cfg.weights)), [str(cfg.images_dir / n) for n in split_names(cfg, part)], {}
    for i in range(0, len(paths), 32):
        print(f"\rpredicting {part}: {min(i + 32, len(paths))}/{len(paths)}", end="", flush=True)
        for r in model.predict(paths[i:i + 32], imgsz=cfg.imgsz, conf=0.01, max_det=cfg.max_det,
                               verbose=False, **head_options(cfg.model)):
            preds[Path(r.path).name] = {"size": [int(r.orig_shape[1]), int(r.orig_shape[0])],
                                        "xyxy": np.round(r.boxes.xyxy.cpu().numpy(), 1).tolist(),
                                        "cls": r.boxes.cls.cpu().numpy().astype(int).tolist(),
                                        "conf": np.round(r.boxes.conf.cpu().numpy(), 4).tolist()}
    print()
    save_json(cache, preds)
    return preds


@dataclass
class Matched:
    """One image after matching: the outcome of every ground-truth box and every prediction."""
    sample: Sample
    gt_xyxy: np.ndarray
    pr_xyxy: np.ndarray
    pr_cls: np.ndarray
    pr_conf: np.ndarray
    gt_outcome: list[str]      # correct | misclassified | poorly_localised | missed
    gt_wrong_cls: list[int]    # class predicted for a misclassified object (-1 otherwise)
    pr_outcome: list[str]      # tp | wrong_class | poor_box | duplicate | background


def match_image(gt_xyxy, gt_cls, pr_xyxy, pr_cls, pr_conf, iou_ok: float, iou_poor: float):
    """
    Predictions are handled from highest to lowest confidence (as in mAP).
    Pass 1: a prediction is correct (tp) if it overlaps a still-free object of the same class with IoU >= iou_ok.
    Pass 2: every other prediction is explained by the object it overlaps most:
            same class, already taken -> duplicate; other class -> wrong_class (object: misclassified);
            same class but iou_poor <= IoU < iou_ok -> poor_box (object: poorly_localised); else -> background.
    Objects no prediction explains stay 'missed'.
    """
    ious = iou_matrix(pr_xyxy, gt_xyxy)
    gt_out, gt_wrong, pr_out = ["missed"] * len(gt_cls), [-1] * len(gt_cls), [""] * len(pr_cls)
    order = np.argsort(-pr_conf)
    for p in order:                                                          # pass 1
        free = [g for g in range(len(gt_cls)) if gt_cls[g] == pr_cls[p] and gt_out[g] != "correct"]
        if free:
            g = max(free, key=lambda j: ious[p, j])
            if ious[p, g] >= iou_ok:
                pr_out[p], gt_out[g] = "tp", "correct"
    for p in order:                                                          # pass 2
        if pr_out[p]:
            continue
        g = int(np.argmax(ious[p])) if len(gt_cls) else -1
        iou, same = (ious[p, g], gt_cls[g] == pr_cls[p]) if g >= 0 else (0.0, False)
        if iou >= iou_ok and same:
            pr_out[p] = "duplicate"
        elif iou >= iou_ok:
            pr_out[p] = "wrong_class"
            if gt_out[g] in ("missed", "poorly_localised"):
                gt_out[g], gt_wrong[g] = "misclassified", int(pr_cls[p])
        elif iou >= iou_poor and same:
            pr_out[p] = "poor_box"
            gt_out[g] = "poorly_localised" if gt_out[g] == "missed" else gt_out[g]
        else:
            pr_out[p] = "background"
    return gt_out, gt_wrong, pr_out


def match_split(cfg: Config, part: str, conf: float) -> list[Matched]:
    """Match the cached predictions (>= conf) of a split with its ground truth."""
    preds, names = predict(cfg, part), set(split_names(cfg, part))
    out = []
    for s in (s for s in load_samples(cfg) if s.image.name in names):
        p = preds[s.image.name]
        keep = np.array(p["conf"]) >= conf
        pr_xyxy, pr_cls, pr_conf = (np.array(p["xyxy"]).reshape(-1, 4)[keep],
                                    np.array(p["cls"], dtype=int)[keep], np.array(p["conf"])[keep])
        gt_xyxy = to_xyxy(s.boxes, *p["size"])
        out.append(Matched(s, gt_xyxy, pr_xyxy, pr_cls, pr_conf,
                           *match_image(gt_xyxy, s.classes, pr_xyxy, pr_cls, pr_conf, cfg.iou_match, cfg.iou_poor)))
    return out


def precision_recall(tp: int, n_pred: int, n_gt: int) -> dict:
    p, r = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "objects": n_gt, "detections": n_pred}


def choose_threshold(cfg: Config) -> float:
    """Confidence threshold with the best F1 on the VALIDATION split (keeps the test set unbiased)."""
    curve = {}
    for conf in np.round(np.arange(0.05, 0.91, 0.05), 2):
        m = match_split(cfg, "val", conf)
        curve[float(conf)] = precision_recall(sum(o == "tp" for x in m for o in x.pr_outcome),
                                              sum(len(x.pr_outcome) for x in m), sum(len(x.gt_outcome) for x in m))
    best = max(curve, key=lambda c: curve[c]["f1"])
    save_json(cfg.results_dir / "test_eval" / "threshold_selection_val.json", {"chosen_conf": best, "val_curve": curve})
    return best


def threshold(cfg: Config) -> float:
    """The threshold chosen by `evaluate`, or choose it now."""
    f = cfg.results_dir / "test_eval" / "threshold_selection_val.json"
    return json.loads(f.read_text())["chosen_conf"] if f.exists() else choose_threshold(cfg)


def groups_of(m: Matched, xyxy: np.ndarray, cls: int) -> dict:
    """The four groupings used in the breakdowns: class, object size, objects per image, image domain."""
    return {"class": CLASSES[cls], "size": size_bucket(box_area(xyxy[None])[0]),
            "density": density_bin(len(m.sample.classes)), "domain": image_domain(m.sample.image)}


def breakdown(matched: list[Matched]) -> dict:
    """Precision and recall per class, object size, objects per image and image domain."""
    counts = defaultdict(lambda: [0, 0, 0])                                  # (kind, group) -> tp, preds, objects
    for m in matched:
        for g in range(len(m.gt_outcome)):
            for kind, key in groups_of(m, m.gt_xyxy[g], m.sample.classes[g]).items():
                counts[kind, key][2] += 1
        for p, outcome in enumerate(m.pr_outcome):
            for kind, key in groups_of(m, m.pr_xyxy[p], m.pr_cls[p]).items():
                counts[kind, key][0] += outcome == "tp"
                counts[kind, key][1] += 1
    result = defaultdict(dict)
    for (kind, key), c in sorted(counts.items(), key=lambda item: (item[0][0], GROUP_ORDER.get(item[0][1], 9), item[0][1])):
        result[kind][key] = precision_recall(*c)
    return dict(result)


def evaluate(cfg: Config) -> dict:
    """Task 2 entry point: Ultralytics test metrics (mAP, per class, confusion matrix) + breakdowns."""
    from ultralytics import YOLO
    localize_paths(cfg)
    box = YOLO(str(cfg.weights)).val(data=str(cfg.data_yaml), split="test", imgsz=cfg.imgsz, max_det=cfg.max_det,
                                     project=str(cfg.results_dir), name="test_eval", exist_ok=True,
                                     **head_options(cfg.model)).box
    per_class = {CLASSES[c]: {"AP50": round(float(box.ap50[i]), 4), "AP50-95": round(float(box.ap[i]), 4),
                              "precision": round(float(box.p[i]), 4), "recall": round(float(box.r[i]), 4)}
                 for i, c in enumerate(box.ap_class_index)}
    conf = choose_threshold(cfg)
    result = {"mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4),
              "per_class": per_class, "threshold": conf, "breakdown": breakdown(match_split(cfg, "test", conf))}
    save_json(cfg.results_dir / "test_eval" / "test_metrics.json", result)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, kind in zip(ax, ("size", "density", "domain")):
        groups = result["breakdown"][kind]
        bars(a, [f"{k}\n(n={v['objects']})" for k, v in groups.items()],
             {"precision": [v["precision"] for v in groups.values()],
              "recall": [v["recall"] for v in groups.values()]}, f"Test precision / recall by {kind}")
    save(fig, cfg.results_dir / "test_eval" / "breakdowns.png")
    return result
