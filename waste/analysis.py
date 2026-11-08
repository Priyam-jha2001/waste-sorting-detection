"""Task 1.1 - dataset analysis: the characteristics that shape the detection pipeline."""
import random
from collections import Counter

import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image, ImageDraw

from .config import CLASSES, Config
from .data import (GROUP_ORDER, density_bin, image_domain, load_samples, near_duplicate_groups, save_json,
                   size_bucket)
from .plots import bars, plt, save

COLOURS = ["#2e7d32", "#8d6e63", "#00acc1", "#757575", "#1e88e5", "#e53935"]


def ordered(groups: dict) -> dict:
    """Small -> large and 1 -> >20 objects, instead of random order."""
    return dict(sorted(groups.items(), key=lambda item: GROUP_ORDER[item[0]]))


def summarise(samples, cfg: Config) -> dict:
    """Class balance, scene density, object size, image domain and label quality in one dictionary."""
    side = 416                                                     # all images are 416 x 416 px
    counts = np.array([len(s.classes) for s in samples])
    boxes_of = {c: np.concatenate([s.boxes[s.classes == c] for s in samples]) for c in range(len(CLASSES))}
    areas = {c: b[:, 2] * b[:, 3] * side * side for c, b in boxes_of.items()}
    all_areas = np.concatenate(list(areas.values()))
    domains = Counter(image_domain(s.image) for s in samples)
    groups = Counter(near_duplicate_groups(samples, cfg.dup_bits))
    return {
        "images": len(samples), "objects": int(counts.sum()),
        "per_class": {name: {"objects": len(areas[c]),
                             "images": sum(c in s.classes for s in samples),
                             "median_box_side_px": round(float(np.sqrt(np.median(areas[c]))), 1),
                             "small_pct": round(100 * float(np.mean(areas[c] < 32 ** 2)), 1)}
                      for c, name in enumerate(CLASSES)},
        "objects_per_image": {"median": float(np.median(counts)), "max": int(counts.max()),
                              **ordered({b: int(n) for b, n in Counter(map(density_bin, counts)).items()})},
        "object_size_pct": ordered({b: round(100 * n / len(all_areas), 1)
                                    for b, n in Counter(map(size_bucket, all_areas)).items()}),
        "image_domain_pct": {d: round(100 * n / len(samples), 1) for d, n in domains.items()},
        "images_with_invalid_boxes": sum(s.invalid > 0 for s in samples),
        "images_in_near_duplicate_groups": sum(n for n in groups.values() if n > 1),
    }


def plot_figures(samples, stats: dict, cfg: Config) -> None:
    """Figures for the presentation (analysis/*.png)."""
    out, pc = cfg.analysis_dir, stats["per_class"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))                   # 1 class balance
    bars(ax[0], list(CLASSES), {"objects": [pc[c]["objects"] for c in CLASSES]}, "Objects per class", "{:.0f}")
    bars(ax[1], list(CLASSES), {"images": [pc[c]["images"] for c in CLASSES]}, "Images containing the class", "{:.0f}")
    save(fig, out / "1_class_distribution.png")

    counts = [len(s.classes) for s in samples]                      # 2 scene density
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(counts, 1, 51), bins=np.arange(1, 53) - 0.5, log=True)
    ax.set(xlabel="objects per image (last bar: > 50)", ylabel="images (log)", title="Objects per image")
    save(fig, out / "2_objects_per_image.png")

    sides = [[np.sqrt(w * h) * 416 for s in samples for (_, _, w, h) in s.boxes[s.classes == c]]
             for c in range(len(CLASSES))]                          # 3 object size per class
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.violinplot(sides, showmedians=True)
    ax.axhline(32, ls="--", c="k", lw=0.8)
    ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES)
    ax.set(ylabel="box side in px (416 px image)", title="Object size per class (dashed: COCO small < 32 px)")
    save(fig, out / "3_object_size_per_class.png")

    centres = np.concatenate([s.boxes[:, :2] for s in samples])     # 4 where objects are
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.hist2d(centres[:, 0], centres[:, 1], bins=40, norm=LogNorm())
    ax.invert_yaxis()
    ax.set_title("Box centres (log colour scale)")
    save(fig, out / "4_box_centres.png")


def sample_grid(samples, cfg: Config, tile: int = 260) -> None:
    """Two labelled random images per class plus the most crowded image (analysis/5_ground_truth_samples.jpg)."""
    rng = random.Random(cfg.seed)
    picks = [s for c in CLASSES for s in rng.sample([s for s in samples if s.source == c], 2)]
    picks.append(max(samples, key=lambda s: len(s.classes)))
    sheet = Image.new("RGB", (5 * tile, 3 * tile), "white")
    for k, s in enumerate(picks):
        im = Image.open(s.image).convert("RGB").resize((tile, tile))
        draw = ImageDraw.Draw(im)
        for c, (cx, cy, w, h) in zip(s.classes, s.boxes * tile):
            draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=COLOURS[c], width=2)
        draw.text((3, 2), f"{s.source}: {len(s.classes)} boxes", fill="black")
        sheet.paste(im, ((k % 5) * tile, (k // 5) * tile))
    sheet.save(cfg.analysis_dir / "5_ground_truth_samples.jpg", quality=90)


def analyze(cfg: Config) -> dict:
    """Task 1.1 entry point: summary JSON + figures in analysis/."""
    samples = load_samples(cfg)
    stats = summarise(samples, cfg)
    save_json(cfg.analysis_dir / "dataset_summary.json", stats)
    plot_figures(samples, stats, cfg)
    sample_grid(samples, cfg)
    return stats
