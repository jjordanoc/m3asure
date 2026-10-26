"""
Build adversarial Toys4K dataset: sample, render, match, poison.

Usage:
    uv run build_adversarial_dataset.py --classes whale --workers 1 --front +z --up +y
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


N_VIEWS = 16

ALL_AZIMUTHS = np.linspace(0, 360, num=N_VIEWS, endpoint=False).astype(int).tolist()
SWAPPABLE_AZIMUTHS = ALL_AZIMUTHS[1:]

BLENDER_SCRIPT = Path(__file__).parent / "create_adv_dataset.py"
DEFAULT_SOURCE = "/workspace/m3asure/data/assets/adv_toys4k/blender/toys4k_blend_files"
DEFAULT_RENDER_DIR = "/workspace/m3asure/data/assets/adv_toys4k/rendered_16_views_latest"
# DEFAULT_OUTPUT = "/workspace/m3asure/data/assets/adv_toys4k_processed"
DEFAULT_OUTPUT = "/workspace/m3asure/data/assets/adv_toys4k_processed_16_views_latest"

DEFAULT_BLENDER = "/root/.m3asure/blender/blender-3.2.2-linux-x64/blender"

PREVIEW_SAMPLES = 32   # low quality for front-view histogram matching
FULL_SAMPLES = 128     # production quality for final dataset views


def call_blender(
    blender_path: str,
    blend_path: str,
    render_dir: str,
    class_name: str,
    object_uid: str,
    azimuths: list[int],
    samples: int = FULL_SAMPLES,
    front: str = "+y",
    up: str = "+z",
) -> tuple[str, bool]:
    """Invoke Blender to render views. Returns (uid, success)."""
    cmd = [
        blender_path, "-b",
        "-P", str(BLENDER_SCRIPT),
        "--",
        "--blend_path", str(blend_path),
        "--output_dir", render_dir,
        "--class_name", class_name,
        "--object_uid", object_uid,
        "--azimuths", ",".join(str(a) for a in azimuths),
        "--samples", str(samples),
        "--front", front,
        "--up", up,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = (result.stdout + "\n" + result.stderr).lower()
    if result.returncode != 0 or "error: python:" in combined or "traceback" in combined:
        detail = result.stderr[-1000:] if result.stderr else result.stdout[-1000:]
        raise RuntimeError(
            f"Blender render failed for {class_name}/{object_uid}:\n{detail}"
        )


def has_renders(render_dir: Path, class_name: str, uid: str, azimuths: list[int]) -> bool:
    rgb_dir = render_dir / class_name / uid / "rgb"
    if not rgb_dir.is_dir():
        return False
    return all((rgb_dir / f"rgb_{a}.png").exists() for a in azimuths)


def render_batch(
    tasks: list[tuple[str, str, str]],
    blender_path: str,
    render_dir: str,
    azimuths: list[int],
    samples: int,
    workers: int,
    desc: str,
    front: str = "+y",
    up: str = "+z",
) -> int:
    """Render a batch of (class_name, uid, blend_path) tuples in parallel.

    Raises on any Blender failure. Returns number of successful renders.
    """
    if not tasks:
        return 0

    rendered = 0

    if workers <= 1:
        for cls, uid, bp in tqdm(tasks, desc=desc):
            call_blender(blender_path, bp, render_dir, cls, uid, azimuths, samples, front, up)
            rendered += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(call_blender, blender_path, bp, render_dir, cls, uid, azimuths, samples, front, up): uid
                for cls, uid, bp in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                future.result()  # raises on Blender failure
                rendered += 1

    return rendered


def discover_blend_files(source_dir: Path, class_filter: set[str] | None = None) -> dict[str, dict[str, Path]]:
    """Returns {class_name: {object_uid: blend_path}}.

    If class_filter is set, only include those classes.
    """
    classes: dict[str, dict[str, Path]] = {}
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_filter and class_dir.name not in class_filter:
            continue
        objects: dict[str, Path] = {}
        for item in sorted(class_dir.iterdir()):
            if item.is_dir():
                blends = list(item.glob("*.blend"))
                if blends:
                    objects[item.name] = blends[0]
            elif item.suffix == ".blend":
                objects[item.stem] = item
        if objects:
            classes[class_dir.name] = objects
    return classes


def compute_hsv_histogram(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def find_best_match(
    target_hist: np.ndarray,
    candidate_hists: dict[str, np.ndarray],
) -> tuple[str, float]:
    best_uid = ""
    best_score = -2.0
    for uid, hist in candidate_hists.items():
        score = cv2.compareHist(target_hist, hist, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            best_uid = uid
    return best_uid, best_score


def copy_views(render_dir: Path, out_dir: Path, class_name: str, uid: str) -> None:
    for modality in ("rgb", "normal"):
        src = render_dir / class_name / uid / modality
        dst = out_dir / modality
        dst.mkdir(parents=True, exist_ok=True)
        for png in src.glob("*.png"):
            shutil.copy2(png, dst / png.name)


def build_poisoned_tuple(
    render_dir: Path,
    dst_dir: Path,
    class_name: str,
    clean_uid: str,
    adversarial_uid: str,
    swap_azimuth: int,
) -> None:
    for modality in ("rgb", "normal"):
        dst_mod = dst_dir / modality
        dst_mod.mkdir(parents=True, exist_ok=True)
        for az in ALL_AZIMUTHS:
            src_uid = adversarial_uid if az == swap_azimuth else clean_uid
            src = render_dir / class_name / src_uid / modality / f"{modality}_{az}.png"
            shutil.copy2(src, dst_mod / f"{modality}_{az}.png")


def build_dataset(
    source_dir: str,
    render_dir: str,
    output_dir: str,
    blender_path: str,
    n_total: int,
    seed: int,
    workers: int,
    class_filter: set[str] | None = None,
    front: str = "+y",
    up: str = "+z",
) -> None:
    source = Path(source_dir)
    renders = Path(render_dir)
    output = Path(output_dir)
    rng = random.Random(seed)

    # Phase 0: discover
    all_classes = discover_blend_files(source, class_filter)
    n_classes = len(all_classes)
    if n_classes == 0:
        raise RuntimeError("No classes found. Check --source_dir and --classes.")
    per_class = math.ceil(n_total / n_classes)
    print(f"{n_classes} classes, sampling {per_class} per class (~{per_class * n_classes} total)")

    # Phase 1: sample per class
    sampled: dict[str, list[str]] = {}
    remaining: dict[str, list[str]] = {}
    for cls, objects in all_classes.items():
        uids = list(objects.keys())
        if len(uids) < 2:
            raise RuntimeError(f"Class '{cls}' has only {len(uids)} object(s), need ≥2 for adversarial matching")
        k = min(per_class, len(uids) - 1)
        chosen = rng.sample(uids, k)
        sampled[cls] = chosen
        remaining[cls] = [u for u in uids if u not in set(chosen)]

    n_sampled = sum(len(v) for v in sampled.values())
    n_candidates = sum(len(v) for v in remaining.values())
    print(f"Sampled {n_sampled} objects, {n_candidates} matching candidates")

    # Phase 2: render front view (0°) for candidates (low-quality, just for histograms)
    print(f"Phase 2: Front views for candidates ({PREVIEW_SAMPLES} spp, {workers} workers)...")
    front_tasks = [
        (cls, uid, str(all_classes[cls][uid]))
        for cls, uids in remaining.items()
        for uid in uids
        if not has_renders(renders, cls, uid, [0])
    ]
    n_rendered = render_batch(front_tasks, blender_path, render_dir, [0], PREVIEW_SAMPLES, workers,
                             "Front views (candidates)", front, up)
    print(f"  → {n_rendered} rendered, {n_candidates - len(front_tasks)} skipped (cached)")

    # Phase 3: render full 4 views for sampled objects (production quality)
    print(f"Phase 3: Full views for sampled objects ({FULL_SAMPLES} spp, {workers} workers)...")
    sampled_tasks = [
        (cls, uid, str(all_classes[cls][uid]))
        for cls, uids in sampled.items()
        for uid in uids
        if not has_renders(renders, cls, uid, ALL_AZIMUTHS)
    ]
    n_rendered = render_batch(sampled_tasks, blender_path, render_dir, ALL_AZIMUTHS, FULL_SAMPLES, workers,
                             "Full views (sampled)", front, up)
    print(f"  → {n_rendered} rendered, {n_sampled - len(sampled_tasks)} skipped (cached)")

    # Phase 4: histogram matching
    print("Phase 4: HSV histogram matching...")
    adversarial_pairs: dict[str, dict[str, tuple[str, float, int]]] = {}

    for cls in tqdm(sampled, desc="Histogram matching"):
        candidate_hists: dict[str, np.ndarray] = {}
        for uid in remaining[cls]:
            front_path = renders / cls / uid / "rgb" / "rgb_0.png"
            img = cv2.imread(str(front_path))
            if img is None:
                raise FileNotFoundError(f"Cannot read rendered front view: {front_path}")
            candidate_hists[uid] = compute_hsv_histogram(img)

        if not candidate_hists:
            raise RuntimeError(f"No valid candidates for class '{cls}' — all front views missing")

        adversarial_pairs[cls] = {}
        for uid in sampled[cls]:
            front_path = renders / cls / uid / "rgb" / "rgb_0.png"
            target_img = cv2.imread(str(front_path))
            if target_img is None:
                raise FileNotFoundError(f"Cannot read rendered front view: {front_path}")
            target_hist = compute_hsv_histogram(target_img)
            adv_uid, score = find_best_match(target_hist, candidate_hists)
            swap_az = rng.choice(SWAPPABLE_AZIMUTHS)
            adversarial_pairs[cls][uid] = (adv_uid, score, swap_az)

    # Phase 5: render full 4 views for adversarial matches
    adv_tasks_set: set[tuple[str, str]] = set()
    for cls, pairs in adversarial_pairs.items():
        for _, (adv_uid, _, _) in pairs.items():
            if not has_renders(renders, cls, adv_uid, ALL_AZIMUTHS):
                adv_tasks_set.add((cls, adv_uid))

    adv_tasks = [(cls, uid, str(all_classes[cls][uid])) for cls, uid in adv_tasks_set]
    print(f"Phase 5: Full views for {len(adv_tasks)} adversarial matches ({FULL_SAMPLES} spp)...")
    n_rendered = render_batch(adv_tasks, blender_path, render_dir, ALL_AZIMUTHS, FULL_SAMPLES, workers,
                             "Full views (adversarial)", front, up)
    print(f"  → {n_rendered} rendered")

    # Phase 6: assemble dataset
    print("Phase 6: Assembling clean + poisoned tuples...")
    manifest: dict[str, dict] = {}
    total_clean = 0
    total_poisoned = 0

    for cls, pairs in adversarial_pairs.items():
        for uid, (adv_uid, score, swap_az) in pairs.items():
            clean_dst = output / cls / uid / "clean"
            copy_views(renders, clean_dst, cls, uid)
            total_clean += 1

            poisoned_dst = output / cls / uid / "poisoned"
            build_poisoned_tuple(renders, poisoned_dst, cls, uid, adv_uid, swap_az)
            total_poisoned += 1

            manifest[f"{cls}/{uid}"] = {
                "class": cls,
                "object_uid": uid,
                "adversarial_uid": adv_uid,
                "histogram_score": round(float(score), 4),
                "swapped_azimuth": swap_az,
                "azimuths": ALL_AZIMUTHS,
            }

    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {total_clean} clean + {total_poisoned} poisoned tuples")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build adversarial Toys4K dataset.")
    parser.add_argument("--source_dir", type=str, default=DEFAULT_SOURCE)
    parser.add_argument("--render_dir", type=str, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--blender_path", type=str, default=DEFAULT_BLENDER)
    parser.add_argument("--n_total", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel Blender processes (4-6 recommended for A40 48GB)")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class filter for testing (e.g. 'whale,airplane')")
    parser.add_argument("--front", type=str, default="+z",
                        help="Axis indicating the front of the object (default: +y)")
    parser.add_argument("--up", type=str, default="+y",
                        help="Axis indicating the top of the object (default: +z)")
    args = parser.parse_args()

    class_filter = set(args.classes.split(",")) if args.classes else None

    build_dataset(
        source_dir=args.source_dir,
        render_dir=args.render_dir,
        output_dir=args.output_dir,
        blender_path=args.blender_path,
        n_total=args.n_total,
        seed=args.seed,
        workers=args.workers,
        class_filter=class_filter,
        front=args.front,
        up=args.up,
    )


if __name__ == "__main__":
    main()
