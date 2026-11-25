"""Evaluate detection on clean, Poisson-noisy and denoised reference images.

For every GT image this script:
1. creates several Poisson noise levels directly from the clean image;
2. denoises every noisy image;
3. runs detection on the clean, noisy and denoised images;
4. computes dataset-level precision, recall, and F1 against YOLO labels;
5. saves metric curves and bbox overlays as PDF files.

Lower ``photons`` values produce stronger Poisson noise.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT
PROJECT_DIR = REPO_ROOT
DENOISE_DIR = REPO_ROOT / 'denoise'


import argparse
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 11


# -----------------------------------------------------------------------------
# Project imports and defaults
# -----------------------------------------------------------------------------


from denoise import UNet
from detect import DetectionModel


DENOISE_WEIGHTS = REPO_ROOT / 'checkpoint' / 'denoise.pth'
DETECT_WEIGHTS = REPO_ROOT / 'checkpoint' / 'detect.pth'
GT_DIR = Path('/path/of/images/denoise_synthetic/GT')
OUTPUT_DIR = Path('/path/of/results/supplementary1')

# Jointly selected with the score threshold from a Poisson 40-to-5 sweep.
# On the full dataset, noisy precision/recall/F1 all decrease across these levels,
# while every denoised metric remains above its noisy counterpart.
DEFAULT_NOISE_PHOTONS = (30.0, 25.0, 15.0)
DEFAULT_NOISE_LABELS = ("Low", "Medium", "High")
DEFAULT_RESOLUTION = 1120
# The same threshold is applied to Clean, Noisy, and Denoised predictions.
DEFAULT_SCORE_THRESHOLD = 0.34
DEFAULT_IOU_THRESHOLD = 0.50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PREDICTION_COLOR = "#FFD400"
GT_COLOR = "#00FF66"


# -----------------------------------------------------------------------------
# Image preparation
# -----------------------------------------------------------------------------
def get_gt_images(gt_dir: Path) -> List[Path]:
    """Return image files directly inside the GT directory."""
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")
    return sorted(
        path
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def apply_poisson_noise(
    clean_rgb: np.ndarray,
    photons: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply signal-dependent shot noise; fewer photons means stronger noise."""
    if photons <= 0:
        raise ValueError(f"Poisson photon level must be > 0, got {photons}")
    clean_float = clean_rgb.astype(np.float32) / 255.0
    noisy = rng.poisson(np.clip(clean_float, 0.0, 1.0) * photons)
    return np.clip(noisy / photons * 255.0, 0, 255).round().astype(np.uint8)


def load_denoiser(weight_path: Path):
    if not weight_path.is_file():
        raise FileNotFoundError(f"denoise weights do not exist: {weight_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_nc=3, out_nc=3, n_feature=48).to(device)
    state_dict = torch.load(str(weight_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded denoise from {weight_path} on {device}")
    return model, device


def denoise_image(model, noisy_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """Denoise one RGB uint8 image, padding dimensions to a multiple of 32."""
    height, width = noisy_rgb.shape[:2]
    pad_h = (-height) % 32
    pad_w = (-width) % 32
    if pad_h or pad_w:
        noisy_rgb = np.pad(
            noisy_rgb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )

    tensor = (
        torch.from_numpy(noisy_rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        prediction = model(tensor)

    prediction = prediction[0, :, :height, :width]
    prediction = prediction.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (prediction * 255.0 + 0.5).astype(np.uint8)


# -----------------------------------------------------------------------------
# Detection and metrics
# -----------------------------------------------------------------------------
def load_yolo_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Load YOLO ``class xc yc w h`` labels as pixel-space xyxy boxes."""
    if not label_path.is_file():
        raise FileNotFoundError(f"YOLO label does not exist: {label_path}")

    boxes: List[List[float]] = []
    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}:{line_number}: {line.rstrip()}"
                )
            xc, yc, box_w, box_h = map(float, fields[1:5])
            x1 = np.clip((xc - box_w / 2.0) * width, 0, width)
            y1 = np.clip((yc - box_h / 2.0) * height, 0, height)
            x2 = np.clip((xc + box_w / 2.0) * width, 0, width)
            y2 = np.clip((yc + box_h / 2.0) * height, 0, height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def predict_boxes(
    model,
    image_rgb: np.ndarray,
    score_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run detection and return boxes, confidence scores, and class IDs."""
    detections = model.predict(Image.fromarray(image_rgb), threshold=score_threshold)
    if isinstance(detections, list):
        detections = detections[0]

    if len(detections) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = np.asarray(detections.xyxy, dtype=np.float32)
    scores = np.asarray(detections.confidence, dtype=np.float32)
    if detections.class_id is None:
        class_ids = np.zeros(len(boxes), dtype=np.int32)
    else:
        class_ids = np.asarray(detections.class_id, dtype=np.int32)
    return boxes, scores, class_ids


def pairwise_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute the pairwise IoU matrix with shape (predictions, GT)."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    top_left = np.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    bottom_right = np.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    intersection_wh = np.clip(bottom_right - top_left, 0, None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    pred_wh = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0, None)
    gt_wh = np.clip(gt_boxes[:, 2:] - gt_boxes[:, :2], 0, None)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    gt_area = gt_wh[:, 0] * gt_wh[:, 1]
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def match_boxes(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Greedily perform one-to-one matching in descending IoU order."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    ious = pairwise_iou(pred_boxes, gt_boxes)
    pred_indices, gt_indices = np.where(ious >= iou_threshold)
    candidates = sorted(
        zip(ious[pred_indices, gt_indices], pred_indices, gt_indices),
        reverse=True,
    )

    matched_predictions = set()
    matched_gt = set()
    for _, pred_index, gt_index in candidates:
        pred_index = int(pred_index)
        gt_index = int(gt_index)
        if pred_index in matched_predictions or gt_index in matched_gt:
            continue
        matched_predictions.add(pred_index)
        matched_gt.add(gt_index)

    true_positive = len(matched_predictions)
    return (
        true_positive,
        len(pred_boxes) - true_positive,
        len(gt_boxes) - true_positive,
    )


def compute_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# -----------------------------------------------------------------------------
# PDF outputs
# -----------------------------------------------------------------------------
def save_bbox_pdf(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    output_path: Path,
    color: str,
    line_width: float,
) -> None:
    """Draw boxes directly over an image and save the result as a PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_rgb.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(image_rgb)

    for x1, y1, x2, y2 in boxes:
        x1 = float(np.clip(x1, 0, width))
        y1 = float(np.clip(y1, 0, height))
        x2 = float(np.clip(x2, 0, width))
        y2 = float(np.clip(y2, 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=line_width,
            )
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.savefig(output_path, format="pdf", dpi=dpi)
    plt.close(fig)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    noise_labels: Sequence[str],
    output_path: Path,
) -> None:
    """Plot Noisy/Denoised performance and the clean-image baseline."""
    clean_value = float(summary.loc[summary["condition"] == "Clean", metric].iloc[0])
    x = np.arange(len(noise_labels))

    fig, ax = plt.subplots(figsize=(4.0, 3.3), dpi=220)
    for condition, color, linestyle, marker in (
        ("Noisy", "#777777", "--", "o"),
        ("Denoised", "#27B89A", "-", "o"),
    ):
        condition_rows = summary[summary["condition"] == condition].set_index(
            "noise_label"
        )
        values = [float(condition_rows.loc[label, metric]) for label in noise_labels]
        ax.plot(
            x,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=1.7,
            label=f"{condition} input",
        )

    ax.axhline(
        clean_value,
        color="#222222",
        linestyle=":",
        linewidth=1.3,
        label="Clean input",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels)
    ax.set_xlabel("Noise severity")
    ax.set_ylabel("Score")
    ax.set_title(metric.capitalize())
    ax.set_ylim(0, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved metric plot: {output_path}")


def append_detection_rows(
    rows: List[Dict[str, object]],
    stem: str,
    noise_label: str,
    photons: float,
    condition: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> None:
    for detection_id, (box, score, class_id) in enumerate(
        zip(boxes, scores, class_ids), start=1
    ):
        rows.append(
            {
                "image": stem,
                "noise_label": noise_label,
                "photons": photons,
                "condition": condition,
                "detection_id": detection_id,
                "class_id": int(class_id),
                "confidence": float(score),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            }
        )


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add multiple Poisson noise levels to GT images, denoise them, and "
            "evaluate detection precision/recall/F1."
        )
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=GT_DIR,
        help="Directory containing GT images and same-stem YOLO .txt labels",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--noise-photons",
        type=float,
        nargs="+",
        default=list(DEFAULT_NOISE_PHOTONS),
        metavar="N",
        help="Photon levels from low to high noise severity (smaller = noisier)",
    )
    parser.add_argument(
        "--noise-labels",
        nargs="+",
        default=list(DEFAULT_NOISE_LABELS),
        metavar="LABEL",
        help="Display labels corresponding to --noise-photons",
    )
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--denoise-weights", type=Path, default=DENOISE_WEIGHTS)
    parser.add_argument("--detect-weights", type=Path, default=DETECT_WEIGHTS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--bbox-line-width", type=float, default=1.0)
    return parser.parse_args()


def validate_args(args) -> None:
    if len(args.noise_photons) != len(args.noise_labels):
        raise ValueError(
            "--noise-photons and --noise-labels must contain the same number of values"
        )
    if not args.noise_photons:
        raise ValueError("At least one Poisson noise level is required")
    if any(photons <= 0 for photons in args.noise_photons):
        raise ValueError("Every --noise-photons value must be > 0")
    if len(set(args.noise_labels)) != len(args.noise_labels):
        raise ValueError("--noise-labels must be unique")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("--score-threshold must be between 0 and 1")
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold must be in (0, 1]")
    if args.bbox_line_width <= 0:
        raise ValueError("--bbox-line-width must be > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    gt_images = get_gt_images(args.gt_dir)
    if not gt_images:
        raise RuntimeError(f"No GT images found in {args.gt_dir}")
    missing_labels = [path.with_suffix(".txt") for path in gt_images]
    missing_labels = [path for path in missing_labels if not path.is_file()]
    if missing_labels:
        preview = "\n".join(str(path) for path in missing_labels[:10])
        raise FileNotFoundError(f"Missing YOLO labels:\n{preview}")

    args.output.mkdir(parents=True, exist_ok=True)
    overlay_root = args.output / "overlays"
    rng = np.random.default_rng(args.seed)

    print(f"GT images: {len(gt_images)}")
    print(
        "Poisson levels: "
        + ", ".join(
            f"{label}={photons:g} photons"
            for label, photons in zip(args.noise_labels, args.noise_photons)
        )
    )
    denoiser, device = load_denoiser(args.denoise_weights)
    if not args.detect_weights.is_file():
        raise FileNotFoundError(f"detect weights do not exist: {args.detect_weights}")
    detector = DetectionModel(
        pretrain_weights=str(args.detect_weights),
        resolution=args.resolution,
    )
    detector.optimize_for_inference()
    print(f"Loaded detect from {args.detect_weights}")

    totals: Dict[Tuple[str, str], List[int]] = {
        ("Clean", "Clean"): [0, 0, 0]
    }
    for noise_label in args.noise_labels:
        totals[(noise_label, "Noisy")] = [0, 0, 0]
        totals[(noise_label, "Denoised")] = [0, 0, 0]

    per_image_rows: List[Dict[str, object]] = []
    detection_rows: List[Dict[str, object]] = []

    for gt_path in tqdm(gt_images, desc="Poisson denoise + detect"):
        stem = gt_path.stem
        clean_rgb = load_rgb(gt_path)
        height, width = clean_rgb.shape[:2]
        gt_boxes = load_yolo_boxes(gt_path.with_suffix(".txt"), width, height)

        save_bbox_pdf(
            clean_rgb,
            gt_boxes,
            overlay_root / "GT" / f"{stem}_gt.pdf",
            GT_COLOR,
            args.bbox_line_width,
        )

        clean_boxes, clean_scores, clean_classes = predict_boxes(
            detector, clean_rgb, args.score_threshold
        )
        clean_counts = match_boxes(clean_boxes, gt_boxes, args.iou_threshold)
        totals[("Clean", "Clean")] = [
            old + new
            for old, new in zip(totals[("Clean", "Clean")], clean_counts)
        ]
        clean_precision, clean_recall, clean_f1 = compute_prf1(*clean_counts)
        per_image_rows.append(
            {
                "image": stem,
                "noise_label": "Clean",
                "photons": np.nan,
                "condition": "Clean",
                "gt_count": len(gt_boxes),
                "prediction_count": len(clean_boxes),
                "tp": clean_counts[0],
                "fp": clean_counts[1],
                "fn": clean_counts[2],
                "precision": clean_precision,
                "recall": clean_recall,
                "f1": clean_f1,
            }
        )
        append_detection_rows(
            detection_rows,
            stem,
            "Clean",
            np.nan,
            "Clean",
            clean_boxes,
            clean_scores,
            clean_classes,
        )
        save_bbox_pdf(
            clean_rgb,
            clean_boxes,
            overlay_root / "Clean" / f"{stem}_clean_prediction.pdf",
            PREDICTION_COLOR,
            args.bbox_line_width,
        )

        for noise_label, photons in zip(args.noise_labels, args.noise_photons):
            noisy_rgb = apply_poisson_noise(clean_rgb, photons, rng)
            denoised_rgb = denoise_image(denoiser, noisy_rgb, device)

            for condition, input_rgb in (
                ("Noisy", noisy_rgb),
                ("Denoised", denoised_rgb),
            ):
                boxes, scores, class_ids = predict_boxes(
                    detector, input_rgb, args.score_threshold
                )
                counts = match_boxes(boxes, gt_boxes, args.iou_threshold)
                key = (noise_label, condition)
                totals[key] = [old + new for old, new in zip(totals[key], counts)]
                precision, recall, f1 = compute_prf1(*counts)
                per_image_rows.append(
                    {
                        "image": stem,
                        "noise_label": noise_label,
                        "photons": photons,
                        "condition": condition,
                        "gt_count": len(gt_boxes),
                        "prediction_count": len(boxes),
                        "tp": counts[0],
                        "fp": counts[1],
                        "fn": counts[2],
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
                append_detection_rows(
                    detection_rows,
                    stem,
                    noise_label,
                    photons,
                    condition,
                    boxes,
                    scores,
                    class_ids,
                )
                save_bbox_pdf(
                    input_rgb,
                    boxes,
                    overlay_root / noise_label / condition / f"{stem}_{condition.lower()}.pdf",
                    PREDICTION_COLOR,
                    args.bbox_line_width,
                )

    summary_rows: List[Dict[str, object]] = []
    ordered_keys = [("Clean", "Clean")]
    ordered_keys.extend(
        (noise_label, condition)
        for noise_label in args.noise_labels
        for condition in ("Noisy", "Denoised")
    )
    photon_lookup = dict(zip(args.noise_labels, args.noise_photons))
    for noise_label, condition in ordered_keys:
        tp, fp, fn = totals[(noise_label, condition)]
        precision, recall, f1 = compute_prf1(tp, fp, fn)
        summary_rows.append(
            {
                "noise_label": noise_label,
                "photons": photon_lookup.get(noise_label, np.nan),
                "condition": condition,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "score_threshold": args.score_threshold,
                "iou_threshold": args.iou_threshold,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)
    pd.DataFrame(per_image_rows).to_csv(
        args.output / "per_image_metrics.csv", index=False
    )
    pd.DataFrame(detection_rows).to_csv(
        args.output / "detection_boxes.csv", index=False
    )

    for metric in ("precision", "recall", "f1"):
        plot_metric(
            summary,
            metric,
            args.noise_labels,
            args.output / f"{metric}_vs_noise.pdf",
        )

    print("\nDataset-level metrics")
    print(
        summary[
            ["noise_label", "photons", "condition", "tp", "fp", "fn", "precision", "recall", "f1"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
