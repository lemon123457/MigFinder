"""
denoise denoise + detect evaluation for paired High/Low real images.

Workflow:
1. Pair Low images with High reference images and LabelMe JSON labels.
2. Denoise Low images and save High, Low, and Denoised images.
3. Run detect detection on High, Low, and Denoised images.
4. Save bbox-only PDF overlays for all three conditions.
5. Use High LabelMe annotations as GT for counts/F1, and compute PSNR plus
   pixel Pearson correlation.
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
import json
import re
import shutil
import time as _time
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.size": 11})

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is expected in this repo env.
    stats = None



from detect import DetectionModel




from denoise import UNet


# ============================================================
# Default configuration
# ============================================================
DENOISE_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'denoise.pth')
DETECT_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'detect.pth')

BASE_DIR = '/path/of/images/denoise_paired'
HIGH_DIR = f"{BASE_DIR}/High"
LOW_DIR = f"{BASE_DIR}/Low"
OUTPUT_DIR = '/path/of/results/denoise_paired'

DEFAULT_IMG_SIZE = 1120
DEFAULT_SCORE_THR = 0.25
DEFAULT_LOW_TOKEN = "Low-1"
DEFAULT_HIGH_TOKEN = "High"
DEFAULT_DENOISED_SUFFIX = "_denoised"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CONDITION_COLORS = {
    "High": "#67A9CF",
    "Low": "#8F8F8F",
    "Denoised": "#2A83B9",
}
OVERLAY_COLORS = {
    "annotation": "white",
    "correct": "#00A651",
    "missing": "#E41A1C",
    "extra": "#377EB8",
}


# ============================================================
# denoise denoise
# ============================================================
def load_denoise_model(weight_path: str):
    """Load the denoise model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(in_nc=3, out_nc=3, n_feature=48).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    print(f"denoise model loaded from {weight_path}")
    return net, device


def denoise_image(net, img_rgb: np.ndarray, device) -> np.ndarray:
    """Denoise a single RGB uint8 image, padding to UNet-compatible size."""
    original_h, original_w = img_rgb.shape[:2]
    divisor = 32
    pad_h = (divisor - original_h % divisor) % divisor
    pad_w = (divisor - original_w % divisor) % divisor

    if pad_h or pad_w:
        img_rgb = np.pad(
            img_rgb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )

    img_in = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = net(tensor)

    pred = pred.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    pred = pred[:original_h, :original_w, :]
    return (pred * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# IO and pairing
# ============================================================
def get_image_files(
    folder: str,
    extensions: Tuple[str, ...] = IMAGE_EXTENSIONS,
) -> List[Path]:
    """Collect image files in a folder, sorted by file name."""
    folder_path = Path(folder)
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    return sorted(set(image_files), key=lambda x: x.name)


def low_stem_to_high_stem(low_stem: str, low_token: str, high_token: str) -> str:
    if low_token and low_stem.endswith(low_token):
        return low_stem[: -len(low_token)] + high_token
    if low_token:
        match = re.search(rf"{re.escape(low_token)}(\d+)$", low_stem)
        if match:
            return low_stem[: match.start()] + high_token + match.group(1)
    if low_stem.endswith("0000"):
        return low_stem[:-4] + "0001"
    return low_stem


def find_image_by_stem(folder: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        for candidate_ext in (ext, ext.upper()):
            candidate = folder / f"{stem}{candidate_ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No image found for stem={stem} under {folder}")


def build_pairs(high_dir: str, low_dir: str, low_token: str, high_token: str) -> List[Dict[str, Path]]:
    """Pair Low images with High reference images and LabelMe JSON labels."""
    high_root = Path(high_dir)
    low_files = get_image_files(low_dir)
    pairs: List[Dict[str, Path]] = []

    for low_path in low_files:
        high_stem = low_stem_to_high_stem(low_path.stem, low_token, high_token)
        high_path = find_image_by_stem(high_root, high_stem)
        label_path = high_root / f"{high_stem}.json"
        if not label_path.exists():
            raise FileNotFoundError(f"No LabelMe JSON found for stem={high_stem}: {label_path}")

        pairs.append(
            {
                "stem": Path(high_stem),
                "high_image": high_path,
                "low_image": low_path,
                "label": label_path,
            }
        )

    return pairs


def load_rgb_array(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def save_rgb_image(image_rgb: np.ndarray, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def clean_generated_output_dirs(output_dir: Path) -> None:
    """Remove generated per-image outputs so stale files do not survive filtering."""
    for dirname in ("images", "bbox_pdfs", "labelme_json"):
        path = output_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def resize_to_shape(image_rgb: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image_rgb.shape[:2] == (target_h, target_w):
        return image_rgb
    return cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


# ============================================================
# Metrics and statistics
# ============================================================
def compute_psnr(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Compute PSNR against the High/reference image."""
    test_rgb = resize_to_shape(test_rgb, reference_rgb.shape[:2])
    ref = reference_rgb.astype(np.float32)
    test = test_rgb.astype(np.float32)
    mse = float(np.mean((ref - test) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def rgb_to_gray_float(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def pixel_pearson(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Pixel-wise Pearson correlation between grayscale intensities."""
    test_rgb = resize_to_shape(test_rgb, reference_rgb.shape[:2])
    ref = rgb_to_gray_float(reference_rgb).reshape(-1)
    test = rgb_to_gray_float(test_rgb).reshape(-1)
    ref_std = float(np.std(ref))
    test_std = float(np.std(test))
    if ref_std == 0.0 or test_std == 0.0:
        return float("nan")
    return float(np.corrcoef(ref, test)[0, 1])


def paired_p_value(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Paired Wilcoxon p-value with a paired t-test fallback."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return float("nan")
    if np.allclose(a, b):
        return 1.0

    if stats is not None:
        try:
            return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except Exception:
            try:
                return float(stats.ttest_rel(a, b).pvalue)
            except Exception:
                pass
    return float("nan")


def p_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def add_sig_bar(ax, x1: float, x2: float, y: float, text: str, h: float) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=10)


def plot_paired_stat(
    data: Dict[str, Sequence[float]],
    comparisons: Sequence[Tuple[str, str]],
    ylabel: str,
    title: str,
    output_path: Path,
    colors: Sequence[str],
) -> None:
    """Bar plot with error bars and significance annotations."""
    labels = list(data.keys())
    x = np.arange(len(labels))
    values = [np.asarray(data[label], dtype=np.float64) for label in labels]

    fig, ax = plt.subplots(figsize=(4.2 + len(labels) * 0.6, 4.2), dpi=200)
    means = [float(np.nanmean(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else np.nan for v in values]
    errors = [sem(v) for v in values]

    ax.bar(
        x,
        means,
        yerr=errors,
        width=0.62,
        color=colors[: len(labels)],
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
        alpha=0.72,
        zorder=1,
    )

    finite_parts = [v[np.isfinite(v)] for v in values if np.any(np.isfinite(v))]
    finite_all = np.concatenate(finite_parts) if finite_parts else np.asarray([], dtype=np.float64)
    if finite_all.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = float(np.min(finite_all))
        y_max = float(np.max(finite_all))
        if y_min == y_max:
            y_max = y_min + 1.0

    y_range = y_max - y_min
    base_y = y_max + y_range * 0.03
    step_y = y_range * 0.09
    bar_h = y_range * 0.025
    label_to_x = {label: idx for idx, label in enumerate(labels)}

    for comp_idx, (left, right) in enumerate(comparisons):
        p_value = paired_p_value(data[left], data[right])
        text = f"{p_to_stars(p_value)}\np={p_value:.2g}" if np.isfinite(p_value) else "n/a"
        add_sig_bar(
            ax,
            label_to_x[left],
            label_to_x[right],
            base_y + comp_idx * step_y,
            text,
            bar_h,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(bottom=min(0.0, y_min - y_range * 0.05), top=base_y + len(comparisons) * step_y + y_range * 0.06)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_correlation(correlation_df: pd.DataFrame, output_path: Path) -> None:
    """Boxplot + paired-line plot for pixel Pearson correlations."""
    low = correlation_df["low_ref_corr"].to_numpy(dtype=np.float64)
    den = correlation_df["denoised_ref_corr"].to_numpy(dtype=np.float64)
    mask = np.isfinite(low) & np.isfinite(den)
    low = low[mask]
    den = den[mask]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.0), dpi=220, sharey=True)
    low_color = "#9A9A9A"
    den_color = "#5EA6CC"
    paired_line_color = "#B8CAE8"
    paired_point_color = "#78A3D8"
    labels = ["Low SNR", "Denoised"]

    box = axes[0].boxplot(
        [low, den],
        tick_labels=labels,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.25", "linewidth": 1.1},
        boxprops={"edgecolor": "0.55", "linewidth": 1.0},
        whiskerprops={"color": "0.45", "linewidth": 1.0},
        capprops={"color": "0.45", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], [low_color, den_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    for raw_value, den_value in zip(low, den):
        axes[1].plot([0, 1], [raw_value, den_value], color=paired_line_color, alpha=0.95, linewidth=1.0)
    axes[1].scatter(np.zeros_like(low), low, color=paired_point_color, alpha=0.55, s=10, edgecolors="none")
    axes[1].scatter(np.ones_like(den), den, color=paired_point_color, alpha=0.9, s=10, edgecolors="none")

    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].text(0.82, 0.985, f"100%\n({len(low)})", ha="center", va="top", fontsize=9.5)

    p_value = paired_p_value(low, den)
    y_max = max(float(np.nanmax(low)), float(np.nanmax(den))) if len(low) else 1.0
    y_min = min(float(np.nanmin(low)), float(np.nanmin(den))) if len(low) else 0.0
    y_range = max(0.05, y_max - y_min)
    add_sig_bar(axes[0], 1, 2, min(0.965, y_max + y_range * 0.04), p_to_stars(p_value), y_range * 0.025)

    for ax in axes:
        ax.set_ylim(0.3, 1.0)
        ax.set_yticks(np.arange(0.4, 1.01, 0.2))
        ax.tick_params(axis="both", width=1.0, length=3, labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)

    axes[0].set_ylabel("Pearson correlation")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ============================================================
# Detection and label metrics
# ============================================================
def predict_boxes(model, image: Image.Image, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    detections = model.predict(image.convert("RGB"), threshold=threshold)
    if isinstance(detections, list):
        detections = detections[0]

    if len(detections) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    boxes = detections.xyxy.astype(np.float32)
    scores = detections.confidence.astype(np.float32)
    if detections.class_id is None:
        class_ids = np.zeros((len(detections),), dtype=np.int32)
    else:
        class_ids = detections.class_id.astype(np.int32)
    return boxes, scores, class_ids


def load_labelme_boxes(label_path: Path, image_width: int, image_height: int) -> np.ndarray:
    """Load LabelMe rectangle/polygon JSON labels as xyxy boxes."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        pts = np.asarray(points, dtype=np.float32)
        x1 = float(np.clip(np.min(pts[:, 0]), 0, image_width))
        y1 = float(np.clip(np.min(pts[:, 1]), 0, image_height))
        x2 = float(np.clip(np.max(pts[:, 0]), 0, image_width))
        y2 = float(np.clip(np.max(pts[:, 1]), 0, image_height))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def scale_boxes_to_shape(boxes: np.ndarray, from_hw: Tuple[int, int], to_hw: Tuple[int, int]) -> np.ndarray:
    if len(boxes) == 0 or from_hw == to_hw:
        return boxes.astype(np.float32)
    from_h, from_w = from_hw
    to_h, to_w = to_hw
    scaled = boxes.astype(np.float32).copy()
    scaled[:, [0, 2]] *= float(to_w) / float(from_w)
    scaled[:, [1, 3]] *= float(to_h) / float(from_h)
    return scaled


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> Tuple[int, int, int]:
    """One-to-one greedy matching by highest IoU."""
    if len(pred_boxes) == 0:
        return 0, 0, int(len(gt_boxes))
    if len(gt_boxes) == 0:
        return 0, int(len(pred_boxes)), 0

    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = iou_xyxy(pred_box, gt_box)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_pred = set()
    matched_gt = set()

    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

    tp = len(matched_pred)
    fp = int(len(pred_boxes) - tp)
    fn = int(len(gt_boxes) - tp)
    return tp, fp, fn


def match_box_indices(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> Tuple[set, set]:
    """Return matched prediction and GT indices using one-to-one greedy IoU matching."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return set(), set()

    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = iou_xyxy(pred_box, gt_box)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_pred = set()
    matched_gt = set()
    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
    return matched_pred, matched_gt


def compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_f1_curve(
    predictions: Dict[str, List[np.ndarray]],
    gt_boxes: List[np.ndarray],
    iou_thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for condition, pred_list in predictions.items():
        for iou_thr in iou_thresholds:
            total_tp = total_fp = total_fn = 0
            for pred_boxes, boxes_gt in zip(pred_list, gt_boxes):
                tp, fp, fn = match_boxes(pred_boxes, boxes_gt, float(iou_thr))
                total_tp += tp
                total_fp += fp
                total_fn += fn

            precision, recall, f1 = compute_prf(total_tp, total_fp, total_fn)
            rows.append(
                {
                    "condition": condition,
                    "iou_threshold": float(iou_thr),
                    "tp": int(total_tp),
                    "fp": int(total_fp),
                    "fn": int(total_fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )
    return pd.DataFrame(rows)


def plot_f1_iou_curve(curve_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=220)
    style = {
        "Low": {"color": "0.45", "label": "Raw data", "linewidth": 1.6},
        "Denoised": {"color": "#2A83B9", "label": "denoise enhanced", "linewidth": 1.8},
    }

    for condition in ["Denoised", "Low"]:
        sub = curve_df[curve_df["condition"] == condition].sort_values("iou_threshold")
        if sub.empty:
            continue
        s = style[condition]
        ax.plot(
            sub["iou_threshold"],
            sub["f1"],
            color=s["color"],
            linewidth=s["linewidth"],
            label=s["label"],
        )

    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Accuracy / F1-score")
    ax.set_xlim(0.3, 0.9)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(np.arange(0.3, 0.91, 0.1))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def save_bbox_pdf(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    output_path: Path,
    line_width: int = 1,
    color: str = "yellow",
) -> None:
    """Save image with bbox-only overlay as PDF."""
    save_colored_bbox_pdf(image_rgb, [(boxes_xyxy, color)], output_path, line_width=line_width)


def save_colored_bbox_pdf(
    image_rgb: np.ndarray,
    box_groups: Sequence[Tuple[np.ndarray, str]],
    output_path: Path,
    line_width: int = 1,
) -> None:
    """Save image with colored bbox-only overlays as PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    for boxes_xyxy, color in box_groups:
        for box in boxes_xyxy:
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = np.clip(x1, 0, w)
            y1 = np.clip(y1, 0, h)
            x2 = np.clip(x2, 0, w)
            y2 = np.clip(y2, 0, h)
            if x2 <= x1 or y2 <= y1:
                continue
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=line_width,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

    fig.savefig(output_path, format="pdf", dpi=dpi)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="denoise denoise + detect High/Low real-image evaluation")

    parser.add_argument("--high-dir", type=str, default=HIGH_DIR, help="Folder containing High images and LabelMe JSON labels")
    parser.add_argument("--low-dir", type=str, default=LOW_DIR, help="Folder containing paired Low images")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output folder")
    parser.add_argument("--low-token", type=str, default=DEFAULT_LOW_TOKEN, help="Low stem token replaced by high-token")
    parser.add_argument("--high-token", type=str, default=DEFAULT_HIGH_TOKEN, help="High stem token used for reference images")
    parser.add_argument("--denoised-suffix", type=str, default=DEFAULT_DENOISED_SUFFIX, help="Suffix for saved denoised image stems")

    parser.add_argument("--denoise-weights", type=str, default=DENOISE_WEIGHTS, help="Path to denoise denoising weights")
    parser.add_argument("--detect-weights", type=str, default=DETECT_WEIGHTS, help="Path to detect weights")
    parser.add_argument("--resolution", type=int, default=DEFAULT_IMG_SIZE, help="detect input resolution")
    parser.add_argument("--det-thresh", type=float, default=DEFAULT_SCORE_THR, help="Detection confidence threshold")

    parser.add_argument("--iou-min", type=float, default=0.30, help="Minimum IoU threshold for F1 curve")
    parser.add_argument("--iou-max", type=float, default=0.90, help="Maximum IoU threshold for F1 curve")
    parser.add_argument("--iou-step", type=float, default=0.05, help="IoU threshold step for F1 curve")
    parser.add_argument("--overlay-iou-threshold", type=float, default=0.50, help="IoU threshold for correct/missing/extra overlay matching")
    parser.add_argument("--bbox-line-width", type=int, default=1, help="BBox line width in overlay PDFs")

    parser.add_argument("--skip-detect", action="store_true", help="Only denoise and compute image-quality metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_output_dirs(output_dir)

    image_dirs = {
        "High": output_dir / "images" / "High",
        "Low": output_dir / "images" / "Low",
        "Denoised": output_dir / "images" / "Denoised",
    }
    overlay_dirs = {
        "High": output_dir / "bbox_pdfs" / "High",
        "Low": output_dir / "bbox_pdfs" / "Low",
        "Denoised": output_dir / "bbox_pdfs" / "Denoised",
    }
    for directory in [*image_dirs.values(), *overlay_dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.high_dir, args.low_dir, args.low_token, args.high_token)
    if not pairs:
        print(f"No paired images found under Low dir: {args.low_dir}")
        return

    print("=" * 80)
    print(f"Found {len(pairs)} paired High/Low images")
    print(f"High dir: {args.high_dir}")
    print(f"Low dir: {args.low_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    print("\nStep 1: Loading denoise model")
    denoise_net, device = load_denoise_model(args.denoise_weights)

    model = None
    if not args.skip_detect:
        print("\nStep 2: Loading detect model")
        model = DetectionModel(pretrain_weights=args.detect_weights, resolution=args.resolution)
        model.optimize_for_inference()

    records = []
    detection_records = []
    gt_boxes_all: List[np.ndarray] = []
    pred_boxes_for_curve = {"Low": [], "Denoised": []}
    counts = {"High": [], "Low": [], "Denoised": []}
    corr_records = []

    print("\nStep 3: Processing pairs")
    total_t0 = _time.time()
    for image_idx, pair in enumerate(tqdm(pairs, desc="Evaluate"), start=1):
        frame_t0 = _time.time()
        stem = pair["stem"].name
        high_path = pair["high_image"]
        low_path = pair["low_image"]
        label_path = pair["label"]

        high_rgb = load_rgb_array(high_path)
        low_rgb = load_rgb_array(low_path)
        denoised_rgb = denoise_image(denoise_net, low_rgb, device)

        high_save_path = image_dirs["High"] / high_path.name
        low_save_path = image_dirs["Low"] / low_path.name
        denoised_name = f"{low_path.stem}{args.denoised_suffix}{low_path.suffix}"
        denoised_path = image_dirs["Denoised"] / denoised_name
        save_rgb_image(high_rgb, high_save_path)
        save_rgb_image(low_rgb, low_save_path)
        save_rgb_image(denoised_rgb, denoised_path)

        psnr_low = compute_psnr(high_rgb, low_rgb)
        psnr_denoised = compute_psnr(high_rgb, denoised_rgb)
        corr_low = pixel_pearson(high_rgb, low_rgb)
        corr_denoised = pixel_pearson(high_rgb, denoised_rgb)
        corr_records.append(
            {
                "image_index": image_idx,
                "stem": stem,
                "low_ref_corr": corr_low,
                "denoised_ref_corr": corr_denoised,
            }
        )

        ref_h, ref_w = high_rgb.shape[:2]
        label_boxes = load_labelme_boxes(label_path, image_width=ref_w, image_height=ref_h)

        row = {
            "image_index": image_idx,
            "stem": stem,
            "high_image": str(high_path),
            "low_image": str(low_path),
            "denoised_image": str(denoised_path),
            "label_file": str(label_path),
            "label_annotation_count": int(len(label_boxes)),
            "psnr_low": float(psnr_low),
            "psnr_denoised": float(psnr_denoised),
            "low_ref_corr": float(corr_low),
            "denoised_ref_corr": float(corr_denoised),
            "denoise_passes": 1,
        }

        if model is not None:
            condition_images = {
                "High": high_rgb,
                "Low": low_rgb,
                "Denoised": denoised_rgb,
            }
            condition_predictions = {}

            for condition, image_rgb in condition_images.items():
                pred_boxes, scores, class_ids = predict_boxes(model, Image.fromarray(image_rgb), threshold=args.det_thresh)
                image_hw = image_rgb.shape[:2]
                condition_predictions[condition] = {
                    "boxes": pred_boxes,
                    "scores": scores,
                    "class_ids": class_ids,
                    "image_hw": image_hw,
                }
                row[f"{condition.lower()}_pred_count"] = int(len(pred_boxes))

            high_gt_boxes = label_boxes
            gt_boxes_all.append(high_gt_boxes)
            counts["High"].append(int(len(high_gt_boxes)))
            row["gt_source"] = "High LabelMe JSON annotations"
            row["gt_count"] = int(len(high_gt_boxes))

            for condition, image_rgb in condition_images.items():
                pred_info = condition_predictions[condition]
                pred_boxes = pred_info["boxes"]
                scores = pred_info["scores"]
                class_ids = pred_info["class_ids"]
                image_hw = pred_info["image_hw"]
                if condition != "High":
                    counts[condition].append(int(len(pred_boxes)))

                if condition in pred_boxes_for_curve:
                    pred_boxes_for_curve[condition].append(scale_boxes_to_shape(pred_boxes, image_hw, high_rgb.shape[:2]))

                pdf_path = overlay_dirs[condition] / f"{stem}_{condition}.pdf"
                if condition == "High":
                    save_bbox_pdf(
                        image_rgb,
                        high_gt_boxes,
                        pdf_path,
                        line_width=args.bbox_line_width,
                        color=OVERLAY_COLORS["annotation"],
                    )
                else:
                    pred_boxes_ref = scale_boxes_to_shape(pred_boxes, image_hw, high_rgb.shape[:2])
                    gt_boxes_image = scale_boxes_to_shape(high_gt_boxes, high_rgb.shape[:2], image_hw)
                    matched_pred, matched_gt = match_box_indices(
                        pred_boxes_ref,
                        high_gt_boxes,
                        iou_threshold=args.overlay_iou_threshold,
                    )
                    correct_boxes = np.asarray(
                        [pred_boxes[idx] for idx in sorted(matched_pred)],
                        dtype=np.float32,
                    ).reshape(-1, 4)
                    missing_boxes = np.asarray(
                        [gt_boxes_image[idx] for idx in range(len(gt_boxes_image)) if idx not in matched_gt],
                        dtype=np.float32,
                    ).reshape(-1, 4)
                    extra_boxes = np.asarray(
                        [pred_boxes[idx] for idx in range(len(pred_boxes)) if idx not in matched_pred],
                        dtype=np.float32,
                    ).reshape(-1, 4)
                    save_colored_bbox_pdf(
                        image_rgb,
                        [
                            (correct_boxes, OVERLAY_COLORS["correct"]),
                            (missing_boxes, OVERLAY_COLORS["missing"]),
                            (extra_boxes, OVERLAY_COLORS["extra"]),
                        ],
                        pdf_path,
                        line_width=args.bbox_line_width,
                    )

                for det_id, (bbox, score, class_id) in enumerate(zip(pred_boxes, scores, class_ids), start=1):
                    detection_records.append(
                        {
                            "image_index": image_idx,
                            "stem": stem,
                            "condition": condition,
                            "detection_id": det_id,
                            "class_id": int(class_id),
                            "score": float(score),
                            "x1": float(bbox[0]),
                            "y1": float(bbox[1]),
                            "x2": float(bbox[2]),
                            "y2": float(bbox[3]),
                            "image_width": int(image_rgb.shape[1]),
                            "image_height": int(image_rgb.shape[0]),
                        }
                    )

        records.append(row)
        print(
            f"  {image_idx:>3d}/{len(pairs)} {stem} | "
            f"PSNR low={psnr_low:.2f}, denoised={psnr_denoised:.2f} | "
            f"corr low={corr_low:.3f}, denoised={corr_denoised:.3f} | "
            f"time={_time.time() - frame_t0:.2f}s"
        )

    print(f"\nAll pairs processed in {_time.time() - total_t0:.1f}s")

    metrics_df = pd.DataFrame(records)
    metrics_csv = output_dir / "per_image_metrics.csv"
    metrics_xlsx = output_dir / "per_image_metrics.xlsx"
    metrics_df.to_csv(metrics_csv, index=False)
    metrics_df.to_excel(metrics_xlsx, index=False)
    print(f"Saved table: {metrics_csv}")
    print(f"Saved table: {metrics_xlsx}")

    corr_df = pd.DataFrame(corr_records)
    corr_csv = output_dir / "pixel_correlation.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"Saved table: {corr_csv}")

    plot_paired_stat(
        data={
            "Low": metrics_df["psnr_low"].to_numpy(),
            "Denoised": metrics_df["psnr_denoised"].to_numpy(),
        },
        comparisons=[("Low", "Denoised")],
        ylabel="PSNR (dB)",
        title="Denoising PSNR",
        output_path=output_dir / "psnr_low_vs_denoised.pdf",
        colors=[CONDITION_COLORS["Low"], CONDITION_COLORS["Denoised"]],
    )

    plot_correlation(corr_df, output_dir / "pixel_pearson_correlation.pdf")

    if model is not None:
        detections_csv = output_dir / "detection_results.csv"
        pd.DataFrame(detection_records).to_csv(detections_csv, index=False)
        print(f"Saved table: {detections_csv}")

        count_summary = pd.DataFrame(
            [
                {
                    "condition": condition,
                    "total_count": int(np.sum(values)),
                    "mean_count": float(np.mean(values)),
                    "sem_count": sem(values),
                    "source": "High LabelMe JSON annotations" if condition == "High" else "detect predictions",
                }
                for condition, values in counts.items()
            ]
        )
        count_summary_path = output_dir / "count_summary.csv"
        count_summary.to_csv(count_summary_path, index=False)
        print(f"Saved table: {count_summary_path}")

        plot_paired_stat(
            data=counts,
            comparisons=[("High", "Low"), ("High", "Denoised"), ("Low", "Denoised")],
            ylabel="Detected migrasome count / image",
            title="detect Migrasome Counts",
            output_path=output_dir / "detection_count_high_low_denoised.pdf",
            colors=[CONDITION_COLORS["High"], CONDITION_COLORS["Low"], CONDITION_COLORS["Denoised"]],
        )

        iou_thresholds = np.round(np.arange(args.iou_min, args.iou_max + args.iou_step / 2.0, args.iou_step), 4)
        curve_df = evaluate_f1_curve(pred_boxes_for_curve, gt_boxes_all, iou_thresholds)
        curve_csv = output_dir / "f1_iou_curve.csv"
        curve_df.to_csv(curve_csv, index=False)
        print(f"Saved table: {curve_csv}")

        plot_f1_iou_curve(curve_df, output_dir / "f1_score_iou_threshold.pdf")

    print("\nDone.")
    print(f"Saved condition images under: {output_dir / 'images'}")
    print(f"Saved bbox PDFs under: {output_dir / 'bbox_pdfs'}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
