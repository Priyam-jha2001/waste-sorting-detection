"""Task 3 - example images of correct, missed and incorrect detections, and error statistics."""
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

from .config import CLASSES, Config
from .data import GROUP_ORDER, box_area, density_bin, save_json, size_bucket
from .evaluation import Matched, match_split, threshold
from .plots import plt, save

GREEN, ORANGE, BLUE, RED, PURPLE = (46, 125, 50), (255, 143, 0), (30, 136, 229), (229, 57, 53), (142, 36, 170)
PRED_COLOUR = {"tp": BLUE, "wrong_class": RED, "background": RED, "poor_box": PURPLE, "duplicate": PURPLE}


def score(m: Matched, case: str) -> float:
    """How good an example image is for a case (higher = better, <= 0 = not suitable)."""
    if case == "correct":                               # all predictions right, all (or >= 80 %) objects found
        found = m.gt_outcome.count("correct") / max(len(m.gt_outcome), 1)
        ok = m.pr_outcome and all(o == "tp" for o in m.pr_outcome)
        return (100 + len(m.gt_outcome) if found == 1 else found if found >= 0.8 else -1) if ok else -1
    if case == "missed":                                # largest share of objects nobody detected
        missed = m.gt_outcome.count("missed")
        return missed / len(m.gt_outcome) + 0.01 * missed if missed else -1
    wrong = sum(o != "tp" for o in m.pr_outcome)        # incorrect: most wrong predictions
    return wrong + 0.5 * ("wrong_class" in m.pr_outcome) if wrong else -1


def pick_examples(matched: list[Matched], case: str, k: int) -> list[Matched]:
    """Best k images for a case, from different classes, readable on a slide (<= 40 objects)."""
    ranked = sorted((m for m in matched if 0 < len(m.sample.classes) <= 40 and score(m, case) > 0),
                    key=lambda m: -score(m, case))
    chosen, seen = [], set()
    for m in ranked:
        if m.sample.source not in seen and len(chosen) < k:
            chosen.append(m)
            seen.add(m.sample.source)
    return chosen


def draw(m: Matched, title: str, path, size: int = 640) -> None:
    """Ground truth (green found / orange missed) and predictions (coloured by outcome) on one image."""
    im = Image.open(m.sample.image).convert("RGB")
    scale = size / max(im.size)
    canvas = Image.new("RGB", (round(im.width * scale), round(im.height * scale) + 24), "white")
    canvas.paste(im.resize((canvas.width, canvas.height - 24)), (0, 24))
    d, shift = ImageDraw.Draw(canvas), np.array([0, 24, 0, 24])

    def rect(box, colour, width=2):
        x1, y1, x2, y2 = box * scale + shift
        d.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], outline=colour, width=width)
        return x1, y1, y2

    for g, outcome in enumerate(m.gt_outcome):
        x1, _, y2 = rect(m.gt_xyxy[g], GREEN if outcome == "correct" else ORANGE, 2 if outcome == "correct" else 3)
        if outcome != "correct":
            d.text((x1 + 2, y2 - 12), f"GT {CLASSES[m.sample.classes[g]]}: {outcome}", fill=ORANGE)
    for p, outcome in enumerate(m.pr_outcome):
        x1, y1, _ = rect(m.pr_xyxy[p], PRED_COLOUR[outcome])
        d.text((x1 + 2, y1 + 2), f"{CLASSES[m.pr_cls[p]]} {m.pr_conf[p]:.2f}", fill=PRED_COLOUR[outcome])
    d.rectangle([0, 0, canvas.width, 22], fill="black")
    d.text((4, 5), title, fill="white")
    canvas.save(path, quality=92)


def statistics(matched: list[Matched]) -> dict:
    """What happens to each test object, per class / object size / objects per image, plus class confusions."""
    outcomes, confusions = defaultdict(lambda: defaultdict(Counter)), Counter()
    for m in matched:
        for g, outcome in enumerate(m.gt_outcome):
            groups = {"class": CLASSES[m.sample.classes[g]], "size": size_bucket(box_area(m.gt_xyxy[g:g + 1])[0]),
                      "objects_per_image": density_bin(len(m.sample.classes))}
            for kind, key in groups.items():
                outcomes[kind][key][outcome] += 1
            if outcome == "misclassified":
                confusions[f"{CLASSES[m.sample.classes[g]]} -> {CLASSES[m.gt_wrong_cls[g]]}"] += 1
    kinds = ("correct", "misclassified", "poorly_localised", "missed")
    rank = {**GROUP_ORDER, **{c: i for i, c in enumerate(CLASSES)}}
    shares = {kind: {key: {o: round(100 * c[o] / sum(c.values()), 1) for o in kinds}
                     for key, c in sorted(groups.items(), key=lambda item: rank[item[0]])}
              for kind, groups in outcomes.items()}
    return {"object_outcomes_pct": shares,
            "prediction_outcomes": dict(Counter(o for m in matched for o in m.pr_outcome)),
            "top_confusions": dict(confusions.most_common(10))}


def plot_outcomes(stats: dict, path) -> None:
    """Stacked bars: share of objects found / misclassified / imprecise / missed per class."""
    shares = stats["object_outcomes_pct"]["class"]
    colours = {"correct": "#2e7d32", "misclassified": "#fb8c00", "poorly_localised": "#8e24aa", "missed": "#e53935"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(CLASSES))
    for outcome, colour in colours.items():
        values = np.array([shares[c][outcome] for c in CLASSES])
        ax.bar(CLASSES, values, bottom=bottom, color=colour, label=outcome)
        bottom += values
    ax.set(ylabel="% of test objects", title="Outcome of each test object per class")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, path)


def error_analysis(cfg: Config) -> dict:
    """Task 3 entry point: 3 examples per case + statistics in results/error_analysis/."""
    out = cfg.results_dir / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    matched = match_split(cfg, "test", threshold(cfg))
    for case in ("correct", "missed", "incorrect"):
        for i, m in enumerate(pick_examples(matched, case, cfg.examples), start=1):
            o = Counter(m.gt_outcome)
            draw(m, f"{case.upper()} {i} | {m.sample.image.name[:28]} | {len(m.gt_outcome)} objects: "
                    f"found {o['correct']}, missed {o['missed']}, wrong class {o['misclassified']}",
                 out / f"{case}_{i}.jpg")
    stats = statistics(matched)
    save_json(out / "error_summary.json", stats)
    plot_outcomes(stats, out / "error_breakdown.png")
    return stats
