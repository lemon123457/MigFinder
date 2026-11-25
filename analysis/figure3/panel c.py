"""
Fig.3 expert annotation vs MigFinder benchmark.

Workflow:
1. Denoise the annotation images.
2. Detect migrasomes and save YOLO labels.
3. Load GT boxes from LabelMe JSON annotations.
4. Save per-image overlays and summary plots.
"""

from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT
PROJECT_DIR = REPO_ROOT
DENOISE_DIR = REPO_ROOT / 'denoise'


import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import cv2
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'migfinder_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from scipy import stats
except Exception:
    stats = None


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 11})

DENOISE_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'denoise.pth')
DETECT_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'detect.pth')

ANNOTATION_IMAGE_ROOT = '/path/of/images/annotation'
EXPERT_ROOT = '/path/of/annotations/experts'
GT_ROOT = '/path/of/annotations/consensus'
TIME_XLSX = '/path/of/annotations/experts/time.xlsx'
OUTPUT_DIR = '/path/of/results/figure3'

DATASETS = ("L929", "NRK", "4T1")
PERSONS = tuple(f"person{i}" for i in range(1, 6))
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
DEFAULT_IMG_SIZE = 1120
DEFAULT_SCORE_THR = 0.4
DEFAULT_MATCH_IOU_THR = 0.5

EXPERT_COLORS = {
    "person1": "#4C78A8",
    "person2": "#F58518",
    "person3": "#54A24B",
    "person4": "#E45756",
    "person5": "#B279A2",
}
MODEL_COLOR = "#DDAA33"
GT_COLOR = "white"
EXPERT_PLOT_COLORS = ("#A9C7E8", "#F6C78F", "#A9D8A8", "#F1A6A4", "#D7B7D8")
GT_PLOT_COLOR = "#B7CBE8"
MODEL_PLOT_COLOR = "#F0D58A"


def load_denoise_model(weight_path: str):
    import torch

    from denoise import UNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(in_nc=3, out_nc=3, n_feature=48).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    print(f"denoise model loaded from {weight_path}")
    return net, device


def synchronize_cuda():
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def denoise_image(net, img_rgb: np.ndarray, device) -> np.ndarray:
    import torch

    orig_h, orig_w = img_rgb.shape[:2]
    pad_h = (32 - orig_h % 32) % 32
    pad_w = (32 - orig_w % 32) % 32
    if pad_h or pad_w:
        img_rgb = np.pad(
            img_rgb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='reflect'
        )

    img_in = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(tensor)
    pred = pred.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    denoised = (pred * 255.0 + 0.5).astype(np.uint8)
    return denoised[:orig_h, :orig_w]


def get_image_files(folder: str) -> List[Path]:
    folder_path = Path(folder)
    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    return sorted([p for p in image_files if '@eaDir' not in p.parts], key=lambda x: x.name)


def load_image_rgb(img_path: Path) -> np.ndarray:
    return np.array(Image.open(img_path).convert('RGB'))


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    x1 = float(np.clip(x1, 0, width))
    y1 = float(np.clip(y1, 0, height))
    x2 = float(np.clip(x2, 0, width))
    y2 = float(np.clip(y2, 0, height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)


def yolo_to_xyxy(parts: List[str], width: int, height: int) -> np.ndarray:
    xc, yc, bw, bh = [float(v) for v in parts[1:5]]
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return clip_box(np.array([x1, y1, x2, y2], dtype=float), width, height)


def xyxy_to_yolo(box: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = clip_box(box, width, height)
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh


def read_yolo_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 4), dtype=float)
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        boxes.append(yolo_to_xyxy(parts, width, height))
    if not boxes:
        return np.empty((0, 4), dtype=float)
    return np.vstack(boxes).astype(float)


def read_labelme_boxes(json_path: Path, width: int, height: int) -> np.ndarray:
    if not json_path.exists():
        print(f"[Warning] Missing GT json: {json_path}")
        return np.empty((0, 4), dtype=float)
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    boxes = []
    for shape in data.get('shapes', []):
        points = np.asarray(shape.get('points', []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        x1, y1 = points[:, 0].min(), points[:, 1].min()
        x2, y2 = points[:, 0].max(), points[:, 1].max()
        boxes.append(clip_box(np.array([x1, y1, x2, y2], dtype=float), width, height))
    if not boxes:
        return np.empty((0, 4), dtype=float)
    return np.vstack(boxes).astype(float)


def save_yolo_boxes(boxes: np.ndarray, output_path: Path, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for box in boxes:
            xc, yc, bw, bh = xyxy_to_yolo(box, width, height)
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def detection_f1(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thr: float) -> Tuple[int, int, int, float, float, float]:
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, float('nan'), float('nan'), float('nan')
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), 0.0, 0.0, 0.0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0.0, 0.0, 0.0

    candidates = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = box_iou(pred_box, gt_box)
            if iou >= iou_thr:
                candidates.append((iou, pred_idx, gt_idx))
    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_preds = set()
    matched_gts = set()
    for _, pred_idx, gt_idx in candidates:
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)

    tp = len(matched_preds)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def segment_bbox_fluorescence(
        image_rgb: np.ndarray,
        box: np.ndarray,
        channel_index: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the foreground mask and dominant-channel fluorescence crop."""
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    x1i = int(np.clip(np.floor(x1), 0, width))
    y1i = int(np.clip(np.floor(y1), 0, height))
    x2i = int(np.clip(np.ceil(x2), 0, width))
    y2i = int(np.clip(np.ceil(y2), 0, height))
    if x2i <= x1i or y2i <= y1i:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return empty.astype(bool), empty

    crop = image_rgb[y1i:y2i, x1i:x2i]
    if channel_index is None:
        channel_index = int(np.argmax(image_rgb.mean(axis=(0, 1))))
    fluorescence = crop[:, :, channel_index]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 255 * 0.65:
        mask = cv2.bitwise_not(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    num_labels, labels, stats_arr, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask > 0, fluorescence
    largest_label = 1 + int(np.argmax(stats_arr[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label, fluorescence


def bbox_shape_metrics(
        image_rgb: np.ndarray,
        box: np.ndarray,
        channel_index: int = None) -> Tuple[float, float, float]:
    mask, fluorescence = segment_bbox_fluorescence(image_rgb, box, channel_index)
    if mask.size == 0:
        return float('nan'), float('nan'), float('nan')
    area = float(mask.sum())
    if area <= 0:
        return float('nan'), float('nan'), float('nan')

    diameter = float(np.sqrt(4.0 * area / np.pi))
    intensity = float(np.mean(fluorescence[mask]))
    return area, diameter, intensity


def summarize_shape_metrics(image_rgb: np.ndarray, boxes: np.ndarray) -> Dict[str, float]:
    channel_index = int(np.argmax(image_rgb.mean(axis=(0, 1))))
    metrics = [bbox_shape_metrics(image_rgb, box, channel_index) for box in boxes]
    areas = [item[0] for item in metrics if np.isfinite(item[0])]
    diameters = [item[1] for item in metrics if np.isfinite(item[1])]
    intensities = [item[2] for item in metrics if np.isfinite(item[2])]
    return {
        'number_per_image': float(len(boxes)),
        'average_area': float(np.mean(areas)) if areas else float('nan'),
        'average_diameter': float(np.mean(diameters)) if diameters else float('nan'),
        'average_intensity': float(np.mean(intensities)) if intensities else float('nan'),
    }


def discover_expert_label_dirs(expert_root: Path, datasets: Tuple[str, ...]) -> Dict[str, Dict[str, Path]]:
    label_dirs = {dataset: {} for dataset in datasets}
    for person in PERSONS:
        person_dir = expert_root / person
        if not person_dir.is_dir():
            print(f"[Warning] Missing expert folder: {person_dir}")
            continue
        for dataset in datasets:
            matches = sorted([
                p for p in person_dir.iterdir()
                if p.is_dir() and p.name.startswith(dataset) and (p / 'labels').is_dir()
            ], key=lambda p: p.name)
            if not matches:
                print(f"[Warning] Missing {dataset} labels for {person}")
                continue
            label_dirs[dataset][person] = matches[0] / 'labels'
    return label_dirs


def detect_boxes(model, denoise_net, device, image_rgb: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    denoised_rgb = denoise_image(denoise_net, image_rgb, device)
    detections = model.predict(Image.fromarray(denoised_rgb), threshold=threshold)
    if len(detections) == 0:
        return np.empty((0, 4), dtype=float), denoised_rgb
    return np.asarray(detections.xyxy, dtype=float), denoised_rgb


def draw_boxes(ax, boxes: np.ndarray, color: str, label: str, linewidth: float, linestyle: str = '-'):
    first = True
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth, edgecolor=color, facecolor='none',
            linestyle=linestyle, label=label if first else None
        )
        ax.add_patch(rect)
        first = False


def save_overlay_pdf(
        image_rgb: np.ndarray,
        expert_boxes: Dict[str, np.ndarray],
        gt_boxes: np.ndarray,
        model_boxes: np.ndarray,
        title: str,
        output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(width / 130, height / 130), dpi=130)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image_rgb)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    for person in PERSONS:
        draw_boxes(ax, expert_boxes.get(person, np.empty((0, 4))), EXPERT_COLORS[person], person, 0.8)
    draw_boxes(ax, gt_boxes, GT_COLOR, 'GT', 2.0, linestyle='--')
    draw_boxes(ax, model_boxes, MODEL_COLOR, 'MigFinder', 1.5)
    ax.text(
        8, 8, title, color='white', fontsize=8, ha='left', va='top',
        bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=2.5)
    )
    ax.legend(loc='lower left', fontsize=6, framealpha=0.65)
    fig.savefig(output_path, format='pdf', dpi=130)
    plt.close(fig)


def p_to_sig_text(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n/a"
    if pvalue < 0.0001:
        return "<0.0001"
    return f"{pvalue:.4f}"


def p_to_sig_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n/a"
    if pvalue < 0.0001:
        return "****"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def mannwhitney_p(values_a: List[float], values_b: List[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or stats is None:
        return float('nan')
    return float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)


def paired_wilcoxon_p(values_a: List[float], values_b: List[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b) or len(a) < 2 or stats is None:
        return float('nan')
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 2:
        return float('nan')
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b, alternative='two-sided').pvalue)


def paired_group_values(
        df: pd.DataFrame,
        dataset: str,
        value_col: str,
        group_col: str,
        group_order: Tuple[str, str]) -> Tuple[List[float], List[float]]:
    """Aggregate repeated sources per image and return complete paired values."""
    subset = df[
        (df['dataset'] == dataset) &
        (df[group_col].isin(group_order))
    ][['image', group_col, value_col]].copy()
    subset[value_col] = pd.to_numeric(subset[value_col], errors='coerce')
    paired = subset.groupby(['image', group_col], sort=False)[value_col].mean().unstack(group_col)
    if any(group not in paired.columns for group in group_order):
        return [], []
    paired = paired.loc[:, list(group_order)].replace([np.inf, -np.inf], np.nan).dropna()
    return (
        paired[group_order[0]].astype(float).to_list(),
        paired[group_order[1]].astype(float).to_list(),
    )


def friedman_expert_count_p(count_df: pd.DataFrame, dataset: str) -> float:
    """Compare the five experts using image-matched bbox counts."""
    subset = count_df[count_df['dataset'] == dataset]
    paired = subset.pivot_table(
        index='image', columns='person', values='bbox_count', aggfunc='mean'
    )
    if any(person not in paired.columns for person in PERSONS) or stats is None:
        return float('nan')
    paired = paired.loc[:, list(PERSONS)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 2:
        return float('nan')
    groups = [paired[person].to_numpy(dtype=float) for person in PERSONS]
    if np.allclose(np.column_stack(groups), groups[0][:, None]):
        return 1.0
    return float(stats.friedmanchisquare(*groups).pvalue)


def sem(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def load_annotation_time_records(time_xlsx: str, datasets: Tuple[str, ...]) -> pd.DataFrame:
    time_path = Path(time_xlsx)
    if not time_path.exists():
        print(f"[Warning] Missing annotation time xlsx: {time_path}")
        return pd.DataFrame(columns=['dataset', 'image_index', 'person', 'seconds'])

    raw = pd.read_excel(time_path, sheet_name=0, header=None)
    header = raw.iloc[0].astype(str).str.strip().to_list()
    records = []
    for dataset in datasets:
        matching_cols = [idx for idx, value in enumerate(header) if value == dataset]
        if not matching_cols:
            print(f"[Warning] Missing {dataset} block in annotation time xlsx")
            continue
        start_col = matching_cols[0]
        for row_idx in range(1, len(raw)):
            image_index = raw.iat[row_idx, start_col]
            if pd.isna(image_index):
                continue
            for offset, person in enumerate(PERSONS, start=1):
                col = start_col + offset
                if col >= raw.shape[1]:
                    continue
                seconds = pd.to_numeric(raw.iat[row_idx, col], errors='coerce')
                if pd.isna(seconds):
                    continue
                records.append({
                    'dataset': dataset,
                    'image_index': int(image_index),
                    'person': person,
                    'seconds': float(seconds),
                })
    return pd.DataFrame(records)


def plot_annotation_time_bar(human_time_df: pd.DataFrame, model_time_df: pd.DataFrame,
                             datasets: Tuple[str, ...], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(7.4, 5.0), dpi=150,
        gridspec_kw={'height_ratios': [3.0, 1.15], 'hspace': 0.06}
    )
    x = np.arange(len(datasets))
    width = 0.36
    human_means = []
    human_sems = []
    model_means = []
    model_sems = []
    pvalues = {}
    bracket_tops = []

    for dataset in datasets:
        human_values = human_time_df.loc[human_time_df['dataset'] == dataset, 'seconds'].dropna().to_list()
        model_values = model_time_df.loc[model_time_df['dataset'] == dataset, 'seconds'].dropna().to_list()
        human_means.append(float(np.mean(human_values)) if human_values else 0.0)
        human_sems.append(sem(human_values))
        model_means.append(float(np.mean(model_values)) if model_values else 0.0)
        model_sems.append(sem(model_values))
        pvalues[dataset] = mannwhitney_p(human_values, model_values)

    for ax in (ax_top, ax_bottom):
        ax.bar(x - width / 2, human_means, width, yerr=human_sems, label='Experts',
               color=GT_PLOT_COLOR, edgecolor='black', linewidth=0.7, capsize=3)
        ax.bar(x + width / 2, model_means, width, yerr=model_sems, label='MigFinder',
               color=MODEL_PLOT_COLOR, edgecolor='black', linewidth=0.7, capsize=3)

    ymax = max(human_means + model_means + [1.0])
    bar_height = ymax * 0.08
    text_offset = ymax * 0.025
    for idx, dataset in enumerate(datasets):
        y = max(human_means[idx] + human_sems[idx], model_means[idx] + model_sems[idx]) + ymax * 0.18
        bracket_tops.append(y + text_offset)
        add_sig_bar(
            ax_top,
            idx - width / 2,
            idx + width / 2,
            y,
            p_to_sig_text(pvalues[dataset]),
            bar_height,
            text_offset
        )

    model_upper = max([m + s for m, s in zip(model_means, model_sems)] + [1.0]) * 1.45
    human_lower = max(model_upper * 2.0, min([m - s for m, s in zip(human_means, human_sems)] + [20.0]) * 0.72)
    ax_bottom.set_ylim(0, model_upper)
    ax_top.set_ylim(human_lower, max(bracket_tops + [ymax]) * 1.12)

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(datasets)
    fig.supylabel('Annotation / inference time (s)', x=0.03)
    ax_top.legend(frameon=False, loc='upper right')
    for ax in (ax_top, ax_bottom):
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    diagonal_kwargs = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=8,
                           linestyle='none', color='black', mec='black', mew=1,
                           clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **diagonal_kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **diagonal_kwargs)

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.96)
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Annotation time bar saved to: {output_path}")


def plot_expert_bbox_count_bar(count_df: pd.DataFrame, datasets: Tuple[str, ...], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.35, 3.05), dpi=200)
    x = np.arange(len(datasets))
    width = 0.13
    offsets = (np.arange(len(PERSONS)) - (len(PERSONS) - 1) / 2) * width
    summary = {}

    for dataset in datasets:
        dataset_summary = {}
        for person in PERSONS:
            values = pd.to_numeric(
                count_df.loc[
                    (count_df['dataset'] == dataset) & (count_df['person'] == person),
                    'bbox_count'
                ], errors='coerce'
            ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            dataset_summary[person] = (
                float(np.mean(values)) if len(values) else np.nan,
                sem(values) if len(values) else np.nan,
            )
        summary[dataset] = dataset_summary

    for person_idx, person in enumerate(PERSONS):
        means = [summary[dataset][person][0] for dataset in datasets]
        sems = [summary[dataset][person][1] for dataset in datasets]
        ax.bar(
            x + offsets[person_idx], means, width, yerr=sems,
            label=f"E{person_idx + 1}", color=EXPERT_PLOT_COLORS[person_idx],
            edgecolor='none', error_kw={'ecolor': 'black', 'elinewidth': 0.8, 'capsize': 1.2},
        )

    bracket_tops = []
    for idx, dataset in enumerate(datasets):
        pvalue = friedman_expert_count_p(count_df, dataset)
        peak = max(
            (mean + error for mean, error in summary[dataset].values()
             if np.isfinite(mean) and np.isfinite(error)),
            default=0.0,
        )
        y = peak + min(8.0, max(4.0, (75.0 - peak) * 0.2))
        bracket_tops.append(y + 3.0)
        add_sig_bar(
            ax,
            idx + offsets[0],
            idx + offsets[-1],
            y,
            p_to_sig_stars(pvalue),
            2.0,
            1.0,
            fontsize=9,
        )
        print(f"  {dataset}: Friedman p={pvalue:.6g} ({p_to_sig_stars(pvalue)})")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Number of migrasomes per image', fontsize=9)
    ax.set_ylim(0, max(75.0, max(bracket_tops, default=0.0) + 3.0))
    ax.set_yticks(np.arange(0, 71, 10))
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, ncol=5, fontsize=8, loc='lower center',
              bbox_to_anchor=(0.5, 1.01), columnspacing=0.3, handlelength=0.9,
              handletextpad=0.2)
    fig.text(0.02, 0.98, 'c', ha='left', va='top', fontsize=12, fontweight='bold')
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.08, top=0.86)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Expert bbox count per image bar saved to: {output_path}")


def add_sig_bar(ax, x1: float, x2: float, y: float, text: str,
                bar_height: float, text_offset: float, fontsize: int = 9):
    ax.plot([x1, x1, x2, x2], [y - bar_height, y, y, y - bar_height],
            color='black', lw=1.0, clip_on=False)
    ax.text((x1 + x2) / 2, y + text_offset, text,
            ha='center', va='bottom', fontsize=fontsize, clip_on=False)


def plot_two_group_violin(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        group_order: Tuple[str, str],
        datasets: Tuple[str, ...],
        ylabel: str,
        output_path: Path,
        colors: Tuple[str, str] = (GT_PLOT_COLOR, MODEL_PLOT_COLOR),
        ylim: Tuple[float, float] = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 4.0), dpi=150)
    centers = np.arange(len(datasets)) + 1.0
    offset = 0.18
    positions = []
    groups = []
    group_colors = []
    pvalues = {}
    dataset_groups = {}

    for center, dataset in zip(centers, datasets):
        values_a, values_b = paired_group_values(
            df, dataset, value_col, group_col, group_order
        )
        positions.extend([center - offset, center + offset])
        groups.extend([values_a, values_b])
        group_colors.extend(colors)
        dataset_groups[dataset] = values_a + values_b
        pvalues[dataset] = paired_wilcoxon_p(values_a, values_b)

    safe_groups = [group if len(group) else [np.nan] for group in groups]
    parts = ax.violinplot(
        safe_groups,
        positions=positions,
        widths=0.26,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for body, color in zip(parts['bodies'], group_colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.74)

    ax.boxplot(
        safe_groups,
        positions=positions,
        widths=0.055,
        patch_artist=True,
        boxprops=dict(facecolor='none', color='black', alpha=0.65),
        medianprops=dict(color='black', linewidth=1.0),
        whiskerprops=dict(color='black', alpha=0.65),
        capprops=dict(color='black', alpha=0.65),
        flierprops=dict(marker='.', markersize=1.4, alpha=0.2)
    )

    finite_values = np.asarray([value for group in groups for value in group if np.isfinite(value)], dtype=float)
    if finite_values.size:
        y_min = min(0.0, float(finite_values.min()))
        y_max = float(finite_values.max())
    else:
        y_min, y_max = 0.0, 1.0
    y_range = max(y_max - y_min, 1.0)
    bar_height = y_range * 0.045
    text_offset = y_range * 0.018
    bracket_tops = []
    for center, dataset in zip(centers, datasets):
        values = dataset_groups[dataset]
        local_max = max(values + [y_max])
        y = local_max + y_range * 0.10
        bracket_tops.append(y + text_offset)
        add_sig_bar(
            ax,
            center - offset,
            center + offset,
            y,
            p_to_sig_text(pvalues[dataset]),
            bar_height,
            text_offset,
            fontsize=8
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    legend_handles = [
        patches.Patch(facecolor=colors[0], edgecolor='black', label=group_order[0], alpha=0.74),
        patches.Patch(facecolor=colors[1], edgecolor='black', label=group_order[1], alpha=0.74),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        ncol=2
    )
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(y_min, max(bracket_tops + [y_max]) + y_range * 0.08)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.13, top=0.84)
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Violin saved to: {output_path}")
    for dataset in datasets:
        print(
            f"  {dataset}: paired Wilcoxon p={p_to_sig_text(pvalues[dataset])} "
            f"(n={len(dataset_groups[dataset]) // 2} image pairs)"
        )


def plot_gt_migfinder_bbox_count_violin(summary_df: pd.DataFrame, datasets: Tuple[str, ...], output_path: Path):
    plot_two_group_violin(
        summary_df.rename(columns={'gt_boxes': 'GT', 'model_boxes': 'MigFinder'})
        .melt(id_vars=['dataset', 'image'], value_vars=['GT', 'MigFinder'],
              var_name='method', value_name='number_per_image'),
        'number_per_image',
        'method',
        ('GT', 'MigFinder'),
        datasets,
        'Number per image',
        output_path,
        colors=(GT_PLOT_COLOR, MODEL_PLOT_COLOR)
    )


def plot_f1_violin(f1_df: pd.DataFrame, datasets: Tuple[str, ...], output_path: Path):
    plot_two_group_violin(
        f1_df, 'f1', 'method', ('Experts', 'MigFinder'), datasets,
        'F1-score', output_path, colors=(GT_PLOT_COLOR, MODEL_PLOT_COLOR), ylim=(0, 1.15)
    )


def plot_precision_violin(f1_df: pd.DataFrame, datasets: Tuple[str, ...], output_path: Path):
    plot_two_group_violin(
        f1_df, 'precision', 'method', ('Experts', 'MigFinder'), datasets,
        'Precision', output_path, colors=(GT_PLOT_COLOR, MODEL_PLOT_COLOR), ylim=(0, 1.15)
    )


def plot_recall_violin(f1_df: pd.DataFrame, datasets: Tuple[str, ...], output_path: Path):
    plot_two_group_violin(
        f1_df, 'recall', 'method', ('Experts', 'MigFinder'), datasets,
        'Recall', output_path, colors=(GT_PLOT_COLOR, MODEL_PLOT_COLOR), ylim=(0, 1.15)
    )


def plot_shape_metric_violin(shape_df: pd.DataFrame, value_col: str, ylabel: str,
                             datasets: Tuple[str, ...], output_path: Path):
    plot_two_group_violin(
        shape_df, value_col, 'method', ('GT', 'MigFinder'), datasets,
        ylabel, output_path, colors=(GT_PLOT_COLOR, MODEL_PLOT_COLOR)
    )


def run_gt_variant(
        args,
        variant_name: str,
        expert_label_dirs: Dict[str, Dict[str, Path]],
        denoise_net,
        device,
        model) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(args.output) / 'gt_benchmark' / variant_name
    model_label_dir = output_dir / 'model_yolo'
    gt_label_dir = output_dir / 'gt_yolo'
    overlay_dir = output_dir / 'overlays'
    plot_dir = output_dir / 'plots'
    denoised_dir = output_dir / 'denoised'
    for p in (model_label_dir, gt_label_dir, overlay_dir, plot_dir):
        p.mkdir(parents=True, exist_ok=True)
    if args.save_denoised:
        denoised_dir.mkdir(parents=True, exist_ok=True)

    count_records = []
    f1_records = []
    shape_records = []
    image_summary_records = []
    runtime_records = []
    runtime_cache = {}
    if args.reuse_model_yolo and (output_dir / 'model_runtime.csv').exists():
        cached_runtime_df = pd.read_csv(output_dir / 'model_runtime.csv')
        runtime_cache = {
            (row['dataset'], row['image']): float(row['seconds'])
            for _, row in cached_runtime_df.iterrows()
        }

    for dataset in args.datasets:
        image_dir = Path(args.annotation_root) / dataset
        image_files = get_image_files(str(image_dir))
        print(f"\n[{variant_name}] Dataset {dataset}: {len(image_files)} images")
        for image_path in tqdm(image_files, desc=f"{variant_name}/{dataset}"):
            image_rgb = load_image_rgb(image_path)
            height, width = image_rgb.shape[:2]
            stem = image_path.stem

            if args.reuse_model_yolo:
                model_boxes = read_yolo_boxes(model_label_dir / dataset / f"{stem}.txt", width, height)
                denoised_rgb = None
                runtime_seconds = runtime_cache.get((dataset, image_path.name), float('nan'))
            else:
                synchronize_cuda()
                start_time = time.perf_counter()
                model_boxes, denoised_rgb = detect_boxes(
                    model, denoise_net, device, image_rgb, args.det_thresh
                )
                synchronize_cuda()
                runtime_seconds = time.perf_counter() - start_time
            model_boxes = np.vstack([clip_box(b, width, height) for b in model_boxes]) if len(model_boxes) else np.empty((0, 4))
            save_yolo_boxes(model_boxes, model_label_dir / dataset / f"{stem}.txt", width, height)
            runtime_records.append({
                'variant': variant_name,
                'dataset': dataset,
                'image': image_path.name,
                'seconds': runtime_seconds,
            })
            if args.save_denoised and denoised_rgb is not None:
                (denoised_dir / dataset).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(denoised_dir / dataset / image_path.name),
                            cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR))

            gt_boxes = read_labelme_boxes(Path(args.gt_root) / dataset / f"{stem}.json", width, height)
            save_yolo_boxes(gt_boxes, gt_label_dir / dataset / f"{stem}.txt", width, height)

            expert_boxes = {}
            for person in PERSONS:
                label_dir = expert_label_dirs.get(dataset, {}).get(person)
                if label_dir is None:
                    expert_boxes[person] = np.empty((0, 4), dtype=float)
                else:
                    expert_boxes[person] = read_yolo_boxes(label_dir / f"{stem}.txt", width, height)
                count_records.append({
                    'variant': variant_name,
                    'dataset': dataset,
                    'image': image_path.name,
                    'person': person,
                    'bbox_count': len(expert_boxes[person]),
                })

            save_overlay_pdf(
                image_rgb, expert_boxes, gt_boxes, model_boxes,
                f"{dataset} / {image_path.name}",
                overlay_dir / dataset / f"{stem}.pdf"
            )

            for person, boxes in expert_boxes.items():
                tp, fp, fn, precision, recall, f1 = detection_f1(boxes, gt_boxes, args.match_iou_thr)
                f1_records.append({
                    'variant': variant_name,
                    'dataset': dataset,
                    'image': image_path.name,
                    'method': 'Experts',
                    'source': person,
                    'reference': 'GT',
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                })
            tp, fp, fn, precision, recall, f1 = detection_f1(model_boxes, gt_boxes, args.match_iou_thr)
            f1_records.append({
                'variant': variant_name,
                'dataset': dataset,
                'image': image_path.name,
                'method': 'MigFinder',
                'source': 'MigFinder',
                'reference': 'GT',
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
            for method, boxes in (('GT', gt_boxes), ('MigFinder', model_boxes)):
                shape_summary = summarize_shape_metrics(image_rgb, boxes)
                shape_records.append({
                    'variant': variant_name,
                    'dataset': dataset,
                    'image': image_path.name,
                    'method': method,
                    **shape_summary,
                })
            image_summary_records.append({
                'variant': variant_name,
                'dataset': dataset,
                'image': image_path.name,
                'gt_boxes': len(gt_boxes),
                'model_boxes': len(model_boxes),
                **{f'{person}_boxes': len(expert_boxes[person]) for person in PERSONS},
            })

    count_df = pd.DataFrame(count_records)
    f1_df = pd.DataFrame(f1_records)
    shape_df = pd.DataFrame(shape_records)
    summary_df = pd.DataFrame(image_summary_records)
    runtime_df = pd.DataFrame(runtime_records)
    count_df.to_csv(output_dir / 'expert_bbox_count.csv', index=False)
    f1_df.to_csv(output_dir / 'gt_experts_vs_migfinder_f1.csv', index=False)
    shape_df.to_csv(output_dir / 'gt_vs_migfinder_shape_metrics.csv', index=False)
    summary_df.to_csv(output_dir / 'image_box_count_summary.csv', index=False)
    runtime_df.to_csv(output_dir / 'model_runtime.csv', index=False)

    plot_expert_bbox_count_bar(count_df, tuple(args.datasets), plot_dir / 'expert_bbox_count_bar.pdf')
    plot_f1_violin(f1_df, tuple(args.datasets), plot_dir / 'gt_experts_vs_migfinder_f1_violin.pdf')
    plot_precision_violin(
        f1_df,
        tuple(args.datasets),
        plot_dir / 'gt_experts_vs_migfinder_precision_violin.pdf'
    )
    plot_recall_violin(
        f1_df,
        tuple(args.datasets),
        plot_dir / 'gt_experts_vs_migfinder_recall_violin.pdf'
    )
    plot_gt_migfinder_bbox_count_violin(
        summary_df,
        tuple(args.datasets),
        plot_dir / 'gt_vs_migfinder_average_number_per_image_violin.pdf'
    )
    plot_shape_metric_violin(
        shape_df, 'average_area', 'Average area (px^2)',
        tuple(args.datasets), plot_dir / 'gt_vs_migfinder_average_area_violin.pdf'
    )
    plot_shape_metric_violin(
        shape_df, 'average_diameter', 'Average diameter (px)',
        tuple(args.datasets), plot_dir / 'gt_vs_migfinder_average_diameter_violin.pdf'
    )
    plot_shape_metric_violin(
        shape_df, 'average_intensity', 'Average intensity (a.u.)',
        tuple(args.datasets), plot_dir / 'gt_vs_migfinder_average_intensity_violin.pdf'
    )
    return count_df, f1_df, shape_df, summary_df, runtime_df


def run_benchmark(args):
    output_dir = Path(args.output) / 'gt_benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_label_dirs = discover_expert_label_dirs(Path(args.expert_root), tuple(args.datasets))
    human_time_df = load_annotation_time_records(args.time_xlsx, tuple(args.datasets))
    human_time_df.to_csv(output_dir / 'human_annotation_time.csv', index=False)
    if args.reuse_model_yolo:
        denoise_net, device, model = None, None, None
    else:

        from detect import DetectionModel
        denoise_net, device = load_denoise_model(args.denoise_weights)
        model = DetectionModel(pretrain_weights=args.detect_weights, resolution=args.resolution)
        model.optimize_for_inference()

    all_counts = []
    all_f1 = []
    all_shape = []
    all_summary = []
    all_runtime = []
    variants = ['denoise']
    for variant_name in variants:
        count_df, f1_df, shape_df, summary_df, runtime_df = run_gt_variant(
            args,
            variant_name,
            expert_label_dirs,
            denoise_net,
            device,
            model
        )
        all_counts.append(count_df)
        all_f1.append(f1_df)
        all_shape.append(shape_df)
        all_summary.append(summary_df)
        all_runtime.append(runtime_df)

    combined_counts = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    combined_f1 = pd.concat(all_f1, ignore_index=True) if all_f1 else pd.DataFrame()
    combined_shape = pd.concat(all_shape, ignore_index=True) if all_shape else pd.DataFrame()
    combined_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    combined_runtime = pd.concat(all_runtime, ignore_index=True) if all_runtime else pd.DataFrame()
    combined_counts.to_csv(output_dir / 'expert_bbox_count_denoise.csv', index=False)
    combined_f1.to_csv(output_dir / 'gt_experts_vs_migfinder_f1_denoise.csv', index=False)
    combined_shape.to_csv(output_dir / 'gt_vs_migfinder_shape_metrics_denoise.csv', index=False)
    combined_summary.to_csv(output_dir / 'image_box_count_summary_denoise.csv', index=False)
    combined_runtime.to_csv(output_dir / 'model_runtime_denoise.csv', index=False)

    denoise_runtime_df = combined_runtime[combined_runtime['variant'] == 'denoise'] if not combined_runtime.empty else pd.DataFrame()
    plot_annotation_time_bar(
        human_time_df,
        denoise_runtime_df,
        tuple(args.datasets),
        output_dir / 'plots' / 'annotation_time_experts_vs_migfinder_bar.pdf'
    )
    print(f"\nDone. GT benchmark outputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Denoise + detect benchmark against GT JSON labels')
    parser.add_argument('--annotation-root', type=str, default=ANNOTATION_IMAGE_ROOT)
    parser.add_argument('--expert-root', type=str, default=EXPERT_ROOT)
    parser.add_argument('--gt-root', type=str, default=GT_ROOT)
    parser.add_argument('--time-xlsx', type=str, default=TIME_XLSX)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument('--denoise-weights', type=str, default=DENOISE_WEIGHTS)
    parser.add_argument('--detect-weights', type=str, default=DETECT_WEIGHTS)
    parser.add_argument('--resolution', type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument('--det-thresh', type=float, default=DEFAULT_SCORE_THR)
    parser.add_argument('--match-iou-thr', type=float, default=DEFAULT_MATCH_IOU_THR)
    parser.add_argument('--reuse-model-yolo', action='store_true', default=False,
                        help='Reuse existing model_yolo labels and skip detect inference')
    parser.add_argument('--save-denoised', action='store_true', default=False)
    parser.add_argument('--expert-count-csv', type=str,
                        help='Plot the expert count panel directly from an existing per-image CSV')
    parser.add_argument('--expert-count-output', type=str,
                        help='Output PDF or PNG for --expert-count-csv')
    args = parser.parse_args()
    if args.expert_count_csv:
        output_path = Path(args.expert_count_output) if args.expert_count_output else (
            Path(args.output) / 'gt_benchmark' / 'denoise' / 'plots' /
            'expert_bbox_count_bar.pdf'
        )
        plot_expert_bbox_count_bar(
            pd.read_csv(args.expert_count_csv), tuple(args.datasets), output_path
        )
        return
    run_benchmark(args)


if __name__ == '__main__':
    main()
