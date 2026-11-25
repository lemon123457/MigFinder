#!/usr/bin/env python3
"""Figure 6 RF-DETR inference for mouse liver, plasma, or human neutrophils.

Writes one JSON file per image with boxes in original-image XYXY coordinates,
plus a CSV manifest. Plasma 2048x2048 images are inferred as four 1024x1024
tiles, matching the selected plasma checkpoint's training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


DATA_ROOT = Path("/path/of/images")
CHECKPOINT_ROOT = Path("/path/of/checkpoints")
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Settings:
    input_root: Path
    weights: Path
    resolution: int
    threshold: float
    batch_size: int


SETTINGS = {
    "mouse_liver": Settings(
        DATA_ROOT / "mouse_liver",
        CHECKPOINT_ROOT / "mouse_liver_detect.pth",
        1344, 0.32, 2,
    ),
    "plasma": Settings(
        DATA_ROOT / "plasma",
        CHECKPOINT_ROOT / "plasma_detect.pth",
        1008, 0.43, 2,
    ),
    "neutrophil": Settings(
        DATA_ROOT / "neutrophil",
        CHECKPOINT_ROOT / "neutrophil_detect.pth",
        1120, 0.30, 2,
    ),
}


def find_images(dataset: str, root: Path, custom_root: bool) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if custom_root:
        candidates = root.rglob("*")
    elif dataset == "mouse_liver":
        candidates = (p for p in root.rglob("*") if p.parent.name.endswith("-rgb"))
    elif dataset == "plasma":
        candidates = (
            p for group in ("Ctrl-new", "6-new", "7-new", "8-new", "9-new")
            for p in (root / group).iterdir()
        )
    else:
        candidates = (p for group in ("Control", "LPS") for p in (root / group).rglob("*"))
    images = sorted(
        (p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p).lower(),
    )
    if not images:
        raise ValueError(f"No images found under {root}")
    return images


def predict_batch(model, images: list[Image.Image], dataset: str, threshold: float):
    inputs = []
    locations = []
    for index, image in enumerate(images):
        if dataset == "plasma":
            if image.size != (2048, 2048):
                raise ValueError(f"Plasma expects 2048x2048 images; got {image.size}")
            for top in (0, 1024):
                for left in (0, 1024):
                    inputs.append(image.crop((left, top, left + 1024, top + 1024)))
                    locations.append((index, left, top))
        else:
            inputs.append(image)
            locations.append((index, 0, 0))

    detections = model.predict(inputs, threshold=threshold)
    if not isinstance(detections, list):
        detections = [detections]
    if len(detections) != len(locations):
        raise RuntimeError(f"Model returned {len(detections)} results for {len(locations)} inputs")

    grouped: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in images]
    for detection, (index, left, top) in zip(detections, locations):
        boxes = np.asarray(detection.xyxy, dtype=float).reshape(-1, 4).copy()
        scores = np.asarray(detection.confidence, dtype=float).reshape(-1)
        if len(boxes) != len(scores):
            raise RuntimeError("Detection boxes and confidence scores have different lengths")
        boxes[:, [0, 2]] += left
        boxes[:, [1, 3]] += top
        grouped[index].append((boxes, scores))

    results = []
    for image, parts in zip(images, grouped):
        boxes = np.concatenate([part[0] for part in parts], axis=0)
        scores = np.concatenate([part[1] for part in parts], axis=0)
        width, height = image.size
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid = (
            np.isfinite(boxes).all(axis=1)
            & np.isfinite(scores)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        results.append((boxes[valid], scores[valid]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(SETTINGS))
    parser.add_argument("--input-dir", type=Path, help="Infer every image recursively in this directory")
    parser.add_argument("--weights", type=Path, help="Override the selected checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "release_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    settings = SETTINGS[args.dataset]
    root = settings.input_root if args.input_dir is None else args.input_dir
    weights = settings.weights if args.weights is None else args.weights
    threshold = settings.threshold if args.threshold is None else args.threshold
    batch_size = settings.batch_size if args.batch_size is None else args.batch_size
    if batch_size < 1 or not 0 <= threshold <= 1:
        parser.error("batch size must be positive and threshold must be in [0, 1]")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    sources = find_images(args.dataset, root, args.input_dir is not None)

    from rfdetr import RFDETRLarge

    model = RFDETRLarge(
        pretrain_weights=str(weights), num_classes=1, class_names=["migrasome"],
        resolution=settings.resolution, device=args.device,
    )
    result_root = args.output_dir / args.dataset
    rows = []
    for start in range(0, len(sources), batch_size):
        batch = sources[start : start + batch_size]
        with_images = []
        for path in batch:
            with Image.open(path) as opened:
                with_images.append(opened.convert("RGB"))
        predictions = predict_batch(model, with_images, args.dataset, threshold)
        for source, image, (boxes, scores) in zip(batch, with_images, predictions):
            relative = source.relative_to(root)
            output = result_root / relative.parent / f"{relative.name}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset": args.dataset,
                "source": str(source),
                "image_width": image.width,
                "image_height": image.height,
                "weights": str(weights),
                "resolution": settings.resolution,
                "score_threshold": threshold,
                "boxes_xyxy": boxes.tolist(),
                "scores": scores.tolist(),
            }
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            rows.append({"source": str(source), "prediction": str(output), "detections": len(boxes)})
        print(f"{args.dataset}: {min(start + batch_size, len(sources))}/{len(sources)} images", flush=True)

    manifest = result_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "prediction", "detections"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} predictions to {result_root}")


if __name__ == "__main__":
    main()
