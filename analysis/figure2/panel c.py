"""
Denoise + Detect evaluation for paired GT/Noisy images.

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
import time as _time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
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

BASE_DIR = '/path/of/images/denoise_synthetic'
GT_DIR = f"{BASE_DIR}/GT"
NOISY_DIR = f"{BASE_DIR}/Noisy"
OUTPUT_DIR = f"{BASE_DIR}/evaluation_result"

DEFAULT_IMG_SIZE = 1120
DEFAULT_SCORE_THR = 0.25
DEFAULT_NOISY_SUFFIX = "_poisson20"
DEFAULT_DENOISED_SUFFIX = "_denoised"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


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
    """
    Denoise a single RGB uint8 image.

    Args:
        net: denoise UNet model.
        img_rgb: (H, W, 3) uint8 RGB image.
        device: torch device.

    Returns:
        Denoised (H, W, 3) uint8 RGB image.
    """
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


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip a float image to uint8 range."""
    return np.clip(image, 0, 255).round().astype(np.uint8)


def apply_noisy_degradation(
    img_rgb: np.ndarray,
    rng: np.random.Generator,
    poisson_photons: float,
    gaussian_sigma: float,
    shading_strength: float,
    shading_grid: int,
) -> np.ndarray:
    """
    Optionally make the input Noisy image harder for detection.

    All knobs are disabled when their value is <= 0. The operations are
    intentionally simple and microscope-like: extra shot noise, read noise,
    and smooth illumination variation.
    """
    arr = img_rgb.astype(np.float32)

    if poisson_photons > 0:
        normalized = np.clip(arr / 255.0, 0.0, 1.0)
        arr = rng.poisson(normalized * poisson_photons).astype(np.float32)
        arr = arr / poisson_photons * 255.0

    if gaussian_sigma > 0:
        arr = arr + rng.normal(0.0, gaussian_sigma, size=arr.shape).astype(np.float32)

    if shading_strength > 0:
        height, width = arr.shape[:2]
        grid = max(2, int(shading_grid))
        coarse = rng.normal(0.0, 1.0, size=(grid, grid)).astype(np.float32)
        field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(height, width) / 12.0)
        field = field - float(field.mean())
        max_abs = float(np.max(np.abs(field)))
        if max_abs > 0:
            field = 1.0 + shading_strength * field / max_abs
            arr = arr * field[:, :, None]

    return to_uint8(arr)


def apply_poisson_noise(img_rgb: np.ndarray, photons: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a Poisson-noisy uint8 image from a clean RGB image."""
    if photons <= 0:
        raise ValueError("Poisson photons must be > 0")
    normalized = np.clip(img_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    noisy = rng.poisson(normalized * photons).astype(np.float32)
    return to_uint8(noisy / photons * 255.0)


def percentile_normalize_rgb(img_rgb: np.ndarray, low: float, high: float) -> np.ndarray:
    """Contrast stretch each channel using percentile limits."""
    arr = img_rgb.astype(np.float32)
    output = np.empty_like(arr)
    for channel_idx in range(arr.shape[2]):
        channel = arr[:, :, channel_idx]
        lo, hi = np.percentile(channel, [low, high])
        if hi <= lo:
            output[:, :, channel_idx] = channel
        else:
            output[:, :, channel_idx] = (channel - lo) * 255.0 / (hi - lo)
    return to_uint8(output)


def unsharp_mask_rgb(img_rgb: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Mild edge restoration after denoising."""
    if alpha <= 0:
        return img_rgb
    arr = img_rgb.astype(np.float32)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return to_uint8(arr + alpha * (arr - blurred))


def postprocess_denoised_image(
    img_rgb: np.ndarray,
    percentile_normalize: bool,
    percentile_low: float,
    percentile_high: float,
    unsharp_alpha: float,
    unsharp_sigma: float,
) -> np.ndarray:
    """Optional denoised-image enhancement before detect evaluation."""
    output = img_rgb
    if percentile_normalize:
        output = percentile_normalize_rgb(output, percentile_low, percentile_high)
    if unsharp_alpha > 0:
        output = unsharp_mask_rgb(output, unsharp_alpha, unsharp_sigma)
    return output


# ============================================================
# IO and pairing
# ============================================================
def get_image_files(
    folder: str,
    extensions: Tuple[str, ...] = IMAGE_EXTENSIONS,
) -> List[str]:
    """Collect image files in a folder, sorted by file name."""
    folder_path = Path(folder)
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files), key=lambda x: x.name)
    return [str(f) for f in image_files]


def strip_suffix(stem: str, suffix: str) -> str:
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def find_image_by_stem(folder: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        for candidate_ext in (ext, ext.upper()):
            candidate = folder / f"{stem}{candidate_ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No GT image found for stem={stem} under {folder}")


def build_pairs(gt_dir: str, noisy_dir: str, noisy_suffix: str) -> List[Dict[str, Path]]:
    """Pair Noisy images with GT images and YOLO labels by normalized stem."""
    gt_root = Path(gt_dir)
    noisy_files = get_image_files(noisy_dir)
    pairs: List[Dict[str, Path]] = []

    for noisy_path_str in noisy_files:
        noisy_path = Path(noisy_path_str)
        clean_stem = strip_suffix(noisy_path.stem, noisy_suffix)
        gt_image_path = find_image_by_stem(gt_root, clean_stem)
        label_path = gt_root / f"{clean_stem}.txt"

        pairs.append(
            {
                "stem": Path(clean_stem),
                "gt_image": gt_image_path,
                "noisy_image": noisy_path,
                "label": label_path,
            }
        )

    return pairs


def generate_noisy_from_gt(
    gt_dir: str,
    noisy_dir: str,
    noisy_suffix: str,
    photons: float,
    rng: np.random.Generator,
    overwrite: bool = False,
) -> int:
    """Generate Poisson-noisy images from GT images and save them to Noisy/."""
    gt_files = get_image_files(gt_dir)
    noisy_root = Path(noisy_dir)
    noisy_root.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for gt_path_str in tqdm(gt_files, desc="Generate Noisy"):
        gt_path = Path(gt_path_str)
        noisy_path = noisy_root / f"{gt_path.stem}{noisy_suffix}{gt_path.suffix}"
        if noisy_path.exists() and not overwrite:
            continue

        gt_rgb = load_rgb_array(gt_path)
        noisy_rgb = apply_poisson_noise(gt_rgb, photons=photons, rng=rng)
        save_rgb_image(noisy_rgb, noisy_path)
        saved_count += 1

    return saved_count


def load_rgb_array(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def save_rgb_image(image_rgb: np.ndarray, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


# ============================================================
# PSNR and statistics
# ============================================================
def compute_psnr(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Compute PSNR against a GT/reference image."""
    if reference_rgb.shape != test_rgb.shape:
        raise ValueError(
            f"PSNR requires equal image shapes, got GT={reference_rgb.shape}, test={test_rgb.shape}"
        )

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
    if reference_rgb.shape != test_rgb.shape:
        raise ValueError(
            f"Pearson requires equal image shapes, got GT={reference_rgb.shape}, test={test_rgb.shape}"
        )

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
    noisy = correlation_df["noisy_ref_corr"].to_numpy(dtype=np.float64)
    den = correlation_df["denoised_ref_corr"].to_numpy(dtype=np.float64)
    mask = np.isfinite(noisy) & np.isfinite(den)
    noisy = noisy[mask]
    den = den[mask]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.0), dpi=220, sharey=True)
    noisy_color = "#9A9A9A"
    den_color = "#5EA6CC"
    paired_line_color = "#B8CAE8"
    paired_point_color = "#78A3D8"
    labels = ["Low SNR", "Denoised"]

    box = axes[0].boxplot(
        [noisy, den],
        tick_labels=labels,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.25", "linewidth": 1.1},
        boxprops={"edgecolor": "0.55", "linewidth": 1.0},
        whiskerprops={"color": "0.45", "linewidth": 1.0},
        capprops={"color": "0.45", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], [noisy_color, den_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    for noisy_value, den_value in zip(noisy, den):
        axes[1].plot([0, 1], [noisy_value, den_value], color=paired_line_color, alpha=0.95, linewidth=1.0)
    axes[1].scatter(np.zeros_like(noisy), noisy, color=paired_point_color, alpha=0.55, s=10, edgecolors="none")
    axes[1].scatter(np.ones_like(den), den, color=paired_point_color, alpha=0.9, s=10, edgecolors="none")

    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(labels)
    axes[1].text(0.82, 0.985, f"100%\n({len(noisy)})", ha="center", va="top", fontsize=9.5)

    p_value = paired_p_value(noisy, den)
    if len(noisy):
        y_max = max(float(np.nanmax(noisy)), float(np.nanmax(den)))
        y_min = min(float(np.nanmin(noisy)), float(np.nanmin(den)))
    else:
        y_max, y_min = 1.0, 0.3
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
# Detection and YOLO metrics
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


def filter_boxes_by_geometry(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    min_area: Optional[float],
    max_area: Optional[float],
    max_aspect_ratio: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optionally remove detections with implausible area or aspect ratio."""
    if len(boxes) == 0:
        return boxes, scores, class_ids
    if min_area is None and max_area is None and max_aspect_ratio is None:
        return boxes, scores, class_ids

    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    aspect_ratios = np.maximum(
        widths / np.maximum(heights, 1e-6),
        heights / np.maximum(widths, 1e-6),
    )

    keep = np.ones((len(boxes),), dtype=bool)
    if min_area is not None:
        keep &= areas >= min_area
    if max_area is not None:
        keep &= areas <= max_area
    if max_aspect_ratio is not None:
        keep &= aspect_ratios <= max_aspect_ratio

    return boxes[keep], scores[keep], class_ids[keep]


def load_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> np.ndarray:
    """Load YOLO txt labels as xyxy boxes in pixel coordinates."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    boxes = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = parts[:5]
            xc = float(xc) * image_width
            yc = float(yc) * image_height
            bw = float(bw) * image_width
            bh = float(bh) * image_height
            x1 = np.clip(xc - bw / 2.0, 0, image_width)
            y1 = np.clip(yc - bh / 2.0, 0, image_height)
            x2 = np.clip(xc + bw / 2.0, 0, image_width)
            y2 = np.clip(yc + bh / 2.0, 0, image_height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


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
        "Noisy": {"color": "0.45", "label": "Raw data", "linewidth": 1.6},
        "Denoised": {"color": "#2A83B9", "label": "denoise enhanced", "linewidth": 1.8},
    }

    for condition in ["Denoised", "Noisy"]:
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


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="denoise denoise + detect denoise evaluation")

    parser.add_argument("--gt-dir", type=str, default=GT_DIR, help="Folder containing GT images and YOLO txt labels")
    parser.add_argument("--noisy-dir", type=str, default=NOISY_DIR, help="Folder containing paired noisy images")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output folder for denoised images and plots")
    parser.add_argument("--noisy-suffix", type=str, default=DEFAULT_NOISY_SUFFIX, help="Suffix used in Noisy file stems")
    parser.add_argument(
        "--denoised-suffix",
        type=str,
        default=DEFAULT_DENOISED_SUFFIX,
        help="Suffix appended after noisy suffix for saved denoised file stems",
    )

    parser.add_argument("--denoise-weights", type=str, default=DENOISE_WEIGHTS, help="Path to denoise denoising weights")
    parser.add_argument("--detect-weights", type=str, default=DETECT_WEIGHTS, help="Path to detect weights")
    parser.add_argument("--resolution", type=int, default=DEFAULT_IMG_SIZE, help="detect input resolution")
    parser.add_argument("--det-thresh", type=float, default=DEFAULT_SCORE_THR, help="Detection confidence threshold")

    parser.add_argument("--iou-min", type=float, default=0.30, help="Minimum IoU threshold for F1 curve")
    parser.add_argument("--iou-max", type=float, default=0.90, help="Maximum IoU threshold for F1 curve")
    parser.add_argument("--iou-step", type=float, default=0.05, help="IoU threshold step for F1 curve")

    noisy_group = parser.add_argument_group("Optional noisy-image degradation")
    noisy_group.add_argument("--seed", type=int, default=20260823, help="Random seed for optional degradation")
    noisy_group.add_argument(
        "--generate-noisy-from-gt",
        action="store_true",
        help="Generate Poisson-noisy images from GT and save them into Noisy before evaluation",
    )
    noisy_group.add_argument(
        "--generated-poisson-photons",
        type=float,
        default=5.0,
        help="Photon level for GT-to-Noisy generation; default is Poisson20",
    )
    noisy_group.add_argument(
        "--overwrite-noisy",
        action="store_true",
        help="Overwrite existing generated Noisy images when generating from GT",
    )
    noisy_group.add_argument(
        "--extra-poisson-photons",
        type=float,
        default=10.0,
        help="Apply extra Poisson resampling to Noisy images; lower positive values are noisier",
    )
    noisy_group.add_argument(
        "--extra-gaussian-sigma",
        type=float,
        default=0.0,
        help="Add extra Gaussian read noise to Noisy images in uint8 intensity units",
    )
    noisy_group.add_argument(
        "--shading-strength",
        type=float,
        default=0.1,
        help="Apply smooth illumination variation to Noisy images; e.g. 0.10 means roughly +/-10%%",
    )
    noisy_group.add_argument(
        "--shading-grid",
        type=int,
        default=4,
        help="Coarse grid size used to generate smooth shading fields",
    )
    noisy_group.add_argument(
        "--save-simulated-noisy",
        action="store_true",
        help="Save the degraded Noisy images used for evaluation",
    )

    denoised_group = parser.add_argument_group("Optional denoised-image enhancement")
    denoised_group.add_argument(
        "--denoise-percentile-normalize",
        action="store_true",
        help="Apply channel-wise percentile contrast normalization after denoising",
    )
    denoised_group.add_argument(
        "--denoise-percentile-low",
        type=float,
        default=0.5,
        help="Lower percentile for denoised contrast normalization",
    )
    denoised_group.add_argument(
        "--denoise-percentile-high",
        type=float,
        default=99.5,
        help="Upper percentile for denoised contrast normalization",
    )
    denoised_group.add_argument(
        "--denoise-unsharp-alpha",
        type=float,
        default=0.0,
        help="Unsharp-mask strength after denoising; try 0.2-0.4",
    )
    denoised_group.add_argument(
        "--denoise-unsharp-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for denoised unsharp masking",
    )

    filter_group = parser.add_argument_group("Optional detection filtering")
    filter_group.add_argument("--bbox-min-area", type=float, default=None, help="Drop predicted boxes below this pixel area")
    filter_group.add_argument("--bbox-max-area", type=float, default=None, help="Drop predicted boxes above this pixel area")
    filter_group.add_argument(
        "--bbox-max-aspect-ratio",
        type=float,
        default=None,
        help="Drop predicted boxes with max(width/height, height/width) above this value",
    )

    parser.add_argument("--skip-detect", action="store_true", help="Only denoise and compute PSNR")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.denoise_percentile_normalize and args.denoise_percentile_high <= args.denoise_percentile_low:
        raise ValueError("--denoise-percentile-high must be larger than --denoise-percentile-low")
    if args.denoise_unsharp_alpha > 0 and args.denoise_unsharp_sigma <= 0:
        raise ValueError("--denoise-unsharp-sigma must be > 0 when unsharp masking is enabled")
    if args.generated_poisson_photons <= 0:
        raise ValueError("--generated-poisson-photons must be > 0")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    denoised_dir = output_dir / "denoised"
    denoised_dir.mkdir(exist_ok=True)

    degrade_noisy = (
        args.extra_poisson_photons > 0
        or args.extra_gaussian_sigma > 0
        or args.shading_strength > 0
    )
    noisy_eval_dir = output_dir / "simulated_noisy"
    if degrade_noisy or args.save_simulated_noisy:
        noisy_eval_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)

    noisy_files = get_image_files(args.noisy_dir)
    if args.generate_noisy_from_gt or not noisy_files:
        print(
            f"Generating Poisson{args.generated_poisson_photons:g} Noisy images "
            f"from GT into {args.noisy_dir}"
        )
        saved_count = generate_noisy_from_gt(
            gt_dir=args.gt_dir,
            noisy_dir=args.noisy_dir,
            noisy_suffix=args.noisy_suffix,
            photons=args.generated_poisson_photons,
            rng=rng,
            overwrite=args.overwrite_noisy,
        )
        print(f"Saved {saved_count} generated Noisy images")

    pairs = build_pairs(args.gt_dir, args.noisy_dir, args.noisy_suffix)
    if not pairs:
        print(f"No paired images found under Noisy dir: {args.noisy_dir}")
        return

    print("=" * 80)
    print(f"Found {len(pairs)} paired GT/Noisy images")
    print(f"GT dir: {args.gt_dir}")
    print(f"Noisy dir: {args.noisy_dir}")
    print(f"Output dir: {output_dir}")
    print(
        "Optional noisy degradation: "
        f"poisson_photons={args.extra_poisson_photons}, "
        f"gaussian_sigma={args.extra_gaussian_sigma}, "
        f"shading_strength={args.shading_strength}"
    )
    print(
        "Optional denoised enhancement: "
        f"percentile_normalize={args.denoise_percentile_normalize}, "
        f"unsharp_alpha={args.denoise_unsharp_alpha}"
    )
    if args.bbox_min_area is not None or args.bbox_max_area is not None or args.bbox_max_aspect_ratio is not None:
        print(
            "Optional bbox filtering: "
            f"min_area={args.bbox_min_area}, "
            f"max_area={args.bbox_max_area}, "
            f"max_aspect_ratio={args.bbox_max_aspect_ratio}"
        )
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
    pred_boxes_for_curve = {"Noisy": [], "Denoised": []}
    counts = {"GT": [], "Noisy": [], "Denoised": []}
    corr_records = []

    print("\nStep 3: Processing pairs")
    total_t0 = _time.time()
    for image_idx, pair in enumerate(tqdm(pairs, desc="Evaluate"), start=1):
        frame_t0 = _time.time()
        stem = pair["stem"].name
        gt_path = pair["gt_image"]
        noisy_path = pair["noisy_image"]
        label_path = pair["label"]

        gt_rgb = load_rgb_array(gt_path)
        noisy_rgb = load_rgb_array(noisy_path)

        if degrade_noisy:
            noisy_rgb = apply_noisy_degradation(
                noisy_rgb,
                rng=rng,
                poisson_photons=args.extra_poisson_photons,
                gaussian_sigma=args.extra_gaussian_sigma,
                shading_strength=args.shading_strength,
                shading_grid=args.shading_grid,
            )

        noisy_eval_path = noisy_path
        if degrade_noisy or args.save_simulated_noisy:
            noisy_eval_name = f"{stem}{args.noisy_suffix}_eval{noisy_path.suffix}"
            noisy_eval_path = noisy_eval_dir / noisy_eval_name
            save_rgb_image(noisy_rgb, noisy_eval_path)

        denoised_rgb = denoise_image(denoise_net, noisy_rgb, device)
        denoised_rgb = postprocess_denoised_image(
            denoised_rgb,
            percentile_normalize=args.denoise_percentile_normalize,
            percentile_low=args.denoise_percentile_low,
            percentile_high=args.denoise_percentile_high,
            unsharp_alpha=args.denoise_unsharp_alpha,
            unsharp_sigma=args.denoise_unsharp_sigma,
        )
        denoised_name = f"{stem}{args.noisy_suffix}{args.denoised_suffix}{noisy_path.suffix}"
        denoised_path = denoised_dir / denoised_name
        save_rgb_image(denoised_rgb, denoised_path)

        psnr_noisy = compute_psnr(gt_rgb, noisy_rgb)
        psnr_denoised = compute_psnr(gt_rgb, denoised_rgb)
        corr_noisy = pixel_pearson(gt_rgb, noisy_rgb)
        corr_denoised = pixel_pearson(gt_rgb, denoised_rgb)
        corr_records.append(
            {
                "image_index": image_idx,
                "stem": stem,
                "noisy_ref_corr": corr_noisy,
                "denoised_ref_corr": corr_denoised,
            }
        )

        h, w = gt_rgb.shape[:2]
        gt_label_boxes = load_yolo_boxes(label_path, image_width=w, image_height=h)
        gt_boxes_all.append(gt_label_boxes)

        row = {
            "image_index": image_idx,
            "stem": stem,
            "gt_image": str(gt_path),
            "noisy_image": str(noisy_path),
            "noisy_eval_image": str(noisy_eval_path),
            "denoised_image": str(denoised_path),
            "label_file": str(label_path),
            "gt_annotation_count": int(len(gt_label_boxes)),
            "psnr_noisy": float(psnr_noisy),
            "psnr_denoised": float(psnr_denoised),
            "noisy_ref_corr": float(corr_noisy),
            "denoised_ref_corr": float(corr_denoised),
            "extra_poisson_photons": float(args.extra_poisson_photons),
            "extra_gaussian_sigma": float(args.extra_gaussian_sigma),
            "shading_strength": float(args.shading_strength),
            "denoise_percentile_normalize": bool(args.denoise_percentile_normalize),
            "denoise_unsharp_alpha": float(args.denoise_unsharp_alpha),
        }

        if model is not None:
            condition_images = {
                "GT": Image.fromarray(gt_rgb),
                "Noisy": Image.fromarray(noisy_rgb),
                "Denoised": Image.fromarray(denoised_rgb),
            }

            for condition, image in condition_images.items():
                pred_boxes, scores, class_ids = predict_boxes(model, image, threshold=args.det_thresh)
                pred_boxes, scores, class_ids = filter_boxes_by_geometry(
                    pred_boxes,
                    scores,
                    class_ids,
                    min_area=args.bbox_min_area,
                    max_area=args.bbox_max_area,
                    max_aspect_ratio=args.bbox_max_aspect_ratio,
                )
                row[f"{condition.lower()}_pred_count"] = int(len(pred_boxes))
                counts[condition].append(int(len(pred_boxes)))

                if condition in pred_boxes_for_curve:
                    pred_boxes_for_curve[condition].append(pred_boxes)

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
                        }
                    )

        records.append(row)
        print(
            f"  {image_idx:>3d}/{len(pairs)} {stem} | "
            f"PSNR noisy={psnr_noisy:.2f}, denoised={psnr_denoised:.2f} | "
            f"corr noisy={corr_noisy:.3f}, denoised={corr_denoised:.3f} | "
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
            "Noisy": metrics_df["psnr_noisy"].to_numpy(),
            "Denoised": metrics_df["psnr_denoised"].to_numpy(),
        },
        comparisons=[("Noisy", "Denoised")],
        ylabel="PSNR (dB)",
        title="Denoising PSNR",
        output_path=output_dir / "psnr_noisy_vs_denoised.pdf",
        colors=["#9A9A9A", "#2A83B9"],
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
                    "total_pred_count": int(np.sum(values)),
                    "mean_pred_count": float(np.mean(values)),
                    "sem_pred_count": sem(values),
                }
                for condition, values in counts.items()
            ]
        )
        count_summary_path = output_dir / "count_summary.csv"
        count_summary.to_csv(count_summary_path, index=False)
        print(f"Saved table: {count_summary_path}")

        plot_paired_stat(
            data=counts,
            comparisons=[("GT", "Noisy"), ("GT", "Denoised"), ("Noisy", "Denoised")],
            ylabel="Detected migrasome count / image",
            title="detect Migrasome Counts",
            output_path=output_dir / "detection_count_gt_noisy_denoised.pdf",
            colors=["#67A9CF", "#BDBDBD", "#2A83B9"],
        )

        iou_thresholds = np.round(np.arange(args.iou_min, args.iou_max + args.iou_step / 2.0, args.iou_step), 4)
        curve_df = evaluate_f1_curve(pred_boxes_for_curve, gt_boxes_all, iou_thresholds)
        curve_csv = output_dir / "f1_iou_curve.csv"
        curve_df.to_csv(curve_csv, index=False)
        print(f"Saved table: {curve_csv}")

        plot_f1_iou_curve(curve_df, output_dir / "f1_score_iou_threshold.pdf")

    print("\nDone.")
    print(f"Denoised images: {denoised_dir}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
