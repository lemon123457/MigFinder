#!/usr/bin/env python3
"""Leakage-free RF-DETR fine-tuning from the user-supplied checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from rfdetr import RFDETRLarge


ROOT = Path("/path/of/results/finetune")
ORIGINAL_WEIGHTS = Path(
    "/path/of/checkpoints/detect.pth"
)
SETTINGS = {
    "mouse_liver": {
        "dataset_dir": ROOT / "mouse_liver" / "coco",
        "output_dir": ROOT / "mouse_liver" / "model_noleak_fixed1344_from_original",
        "resolution": 1344,
        "batch_size": 2,
        "grad_accum_steps": 4,
    },
    "plasma": {
        "dataset_dir": ROOT / "plasma" / "coco_tiles_noleak",
        "output_dir": ROOT / "plasma" / "model_tiles_noleak_from_original",
        "resolution": 1008,
        "batch_size": 4,
        "grad_accum_steps": 2,
    },
}


def load_coco(dataset_dir: Path, split: str) -> dict:
    with (dataset_dir / split / "_annotations.coco.json").open(
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


def verify_dataset(dataset: str, dataset_dir: Path) -> dict:
    split_names: dict[str, set[str]] = {}
    summary = {}
    for split in ("train", "valid", "test"):
        coco = load_coco(dataset_dir, split)
        category_ids = sorted(category["id"] for category in coco["categories"])
        annotation_categories = {item["category_id"] for item in coco["annotations"]}
        if category_ids != [0] or not annotation_categories.issubset({0}):
            raise ValueError(f"Expected zero-based one-class COCO in {split}")
        split_names[split] = {image["file_name"] for image in coco["images"]}
        if len(split_names[split]) != len(coco["images"]):
            raise ValueError(f"Duplicate filenames in {split}")
        summary[split] = {
            "images": len(coco["images"]),
            "boxes": len(coco["annotations"]),
        }

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = split_names[left] & split_names[right]
        if overlap:
            raise ValueError(f"Filename leakage between {left}/{right}: {len(overlap)}")

    if dataset == "plasma":
        manifest_path = dataset_dir / "noleak_manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("validation_is_leaked_into_train") is not False:
            raise ValueError(f"Invalid no-leak declaration: {manifest_path}")
        if manifest.get("test_is_independent") is not True:
            raise ValueError(f"Invalid test declaration: {manifest_path}")
    return summary


def train_one(
    dataset: str,
    epochs: int,
    lr: float,
    num_workers: int,
    seed: int,
) -> None:
    settings = SETTINGS[dataset]
    dataset_dir = settings["dataset_dir"]
    output_dir = settings["output_dir"]
    summary = verify_dataset(dataset, dataset_dir)
    if not ORIGINAL_WEIGHTS.is_file():
        raise FileNotFoundError(ORIGINAL_WEIGHTS)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix training outputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": dataset,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "initial_weights": str(ORIGINAL_WEIGHTS),
        "validation_is_leaked_into_train": False,
        "validation_is_independent": True,
        "test_is_independent": True,
        "split_summary": summary,
        "resolution": settings["resolution"],
        "batch_size": settings["batch_size"],
        "grad_accum_steps": settings["grad_accum_steps"],
        "effective_batch_size": (
            settings["batch_size"] * settings["grad_accum_steps"]
        ),
        "epochs": epochs,
        "lr": lr,
        "lr_drop": 40,
        "weight_decay": 1e-5,
        "multi_scale": False,
        "expanded_scales": False,
        "augmentation": "fixed-size resize plus horizontal flip; no random crop",
        "weak_train_augmentation": True,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_use_ema": True,
        "num_workers": num_workers,
        "seed": seed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    model = RFDETRLarge(
        pretrain_weights=str(ORIGINAL_WEIGHTS),
        num_classes=1,
        class_names=["migrasome"],
        resolution=settings["resolution"],
        device="cuda",
    )
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=settings["batch_size"],
        grad_accum_steps=settings["grad_accum_steps"],
        lr=lr,
        lr_drop=40,
        resolution=settings["resolution"],
        multi_scale=False,
        expanded_scales=False,
        disable_train_augmentation=False,
        weak_train_augmentation=True,
        weight_decay=1e-5,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0005,
        early_stopping_use_ema=True,
        num_workers=num_workers,
        run_test=True,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
    train_one(args.dataset, args.epochs, args.lr, args.num_workers, args.seed)


if __name__ == "__main__":
    main()
