"""Task 1.2 - stratified, duplicate-aware train / validation / test split (fixed seed = same split every time)."""
import random
from collections import Counter, defaultdict

from .config import CLASSES, Config
from .data import density_bin, load_samples, near_duplicate_groups, save_json, split_names


def make_split(cfg: Config) -> dict:
    """
    70 / 15 / 15 split.
    - images with broken (zero-size) boxes are left out
    - strata = class in the file name x objects-per-image bin, so every split has the same mix
    - near-duplicate images stay together, so the test set holds no copy of a training image
    """
    samples = [s for s in load_samples(cfg) if s.invalid == 0 and len(s.classes) > 0]
    members = defaultdict(list)                                    # duplicate group -> image indices
    for i, g in enumerate(near_duplicate_groups(samples, cfg.dup_bits)):
        members[g].append(i)
    strata = defaultdict(list)                                     # (class, density) -> groups
    for g, idx in members.items():
        first = samples[idx[0]]
        strata[(first.source, density_bin(len(first.classes)))].append(g)

    split, rng = {"train": [], "val": [], "test": []}, random.Random(cfg.seed)
    for key in sorted(strata):
        groups = strata[key]
        rng.shuffle(groups)
        total = sum(len(members[g]) for g in groups)
        filled = {"test": 0, "val": 0}
        for g in groups:                                           # fill test, then val, rest to train
            n = len(members[g])
            part = ("test" if filled["test"] + n <= round(total * cfg.test_frac) else
                    "val" if filled["val"] + n <= round(total * cfg.val_frac) else "train")
            filled[part] = filled.get(part, 0) + n
            split[part] += [samples[i] for i in members[g]]

    cfg.split_dir.mkdir(parents=True, exist_ok=True)
    for part, items in split.items():
        (cfg.split_dir / f"{part}.txt").write_text("\n".join(str(s.image) for s in items) + "\n", encoding="utf-8")
    write_yaml(cfg)
    report = {part: {"images": len(items), **Counter(CLASSES[c] for s in items for c in s.classes)}
              for part, items in split.items()}
    save_json(cfg.split_dir / "split_report.json", report)
    return report


def write_yaml(cfg: Config) -> None:
    """Dataset file that tells Ultralytics where the three image lists are."""
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    cfg.data_yaml.write_text(f"path: {cfg.split_dir.as_posix()}\ntrain: train.txt\nval: val.txt\n"
                             f"test: test.txt\nnames:\n{names}")


def localize_paths(cfg: Config) -> None:
    """Rewrite the image lists with this computer's paths; the split itself never changes."""
    for part in ("train", "val", "test"):
        names = split_names(cfg, part)
        (cfg.split_dir / f"{part}.txt").write_text(
            "\n".join(str(cfg.images_dir / n) for n in names) + "\n", encoding="utf-8")
    write_yaml(cfg)
