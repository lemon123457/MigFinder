'Denoise, detect and track an image sequence; measure intensity, area and circularity and summarize trajectories and appearance/disappearance events.'

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT
PROJECT_DIR = REPO_ROOT
DENOISE_DIR = REPO_ROOT / 'denoise'


import os
import argparse
from typing import Dict, List, Optional, Tuple

import time as _time

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm


from detect import DetectionModel
from tracker import Tracker, TrackingConfig, STrack


from denoise import UNet

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 14})

# ============================================================

# ============================================================
DENOISE_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'denoise.pth')

DEFAULT_IMG_SIZE = 1120
DEFAULT_SCORE_THR = 0.25
DEFAULT_IOU_THR = 0.5

DETECT_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'detect.pth')


INPUT_DIR = '/path/of/images/figure4_events'
OUTPUT_DIR = '/path/of/results/figure4_events'


FRAME_INTERVAL_SEC = 525.73859


DEFAULT_MICRONS_PER_PIXEL = 176.7767 / 1024.0


DEFAULT_EVENT_WINDOW_MIN = 30.0
DEFAULT_EVENT_MIN_TRACK_FRAMES = 3
DEFAULT_EVENT_GAP_TOLERANCE = 3
DEFAULT_FRAGMENT_MAX_GAP = 3
DEFAULT_FRAGMENT_MAX_DISTANCE_UM = 5.0
DEFAULT_EVENT_EDGE_MARGIN_UM = 5.0
DEFAULT_CONTOUR_OPENING_UM = 0.5

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128)
]


# ============================================================

# ============================================================
def load_denoise_model(weight_path: str):
    'Load the denoise network and its checkpoint.'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(in_nc=3, out_nc=3, n_feature=48).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    print(f"denoise model loaded from {weight_path}")
    return net, device


def denoise_image(net, img_rgb: np.ndarray, device) -> np.ndarray:
    'Denoise one (H, W, 3) RGB uint8 image on the requested device and return RGB uint8 data.'
    img_in = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_in).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = net(tensor)

    pred = pred.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (pred * 255.0 + 0.5).astype(np.uint8)


# ============================================================


# ============================================================

# ============================================================
def get_image_files(folder: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) -> List[str]:
    'List image files in the folder, sorted by filename.'
    folder_path = Path(folder)
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    image_files = sorted(image_files, key=lambda x: x.name)
    return [str(f) for f in image_files]


def load_frame_for_inference_and_vis(img_path: str) -> Tuple[Image.Image, np.ndarray]:
    'Load an image as RGB PIL data for detection and BGR array data for visualization; support grayscale and RGBA inputs.'
    pil_raw = Image.open(img_path)
    pil_rgb = pil_raw.convert('RGB')

    arr = np.array(pil_raw)
    if arr.ndim == 2:
        vis_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        vis_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        vis_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return pil_rgb, vis_bgr


# ============================================================

# ============================================================
def detections_to_array(detections) -> np.ndarray:
    'Convert detections to tracker input: an (N, 5) array of [x1, y1, x2, y2, score].'
    if len(detections) == 0:
        return np.empty((0, 5))
    boxes = detections.xyxy
    scores = detections.confidence.reshape(-1, 1)
    return np.hstack([boxes, scores])


def _clip_bbox_to_image(bbox, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(np.floor(x1), 0, w - 1))
    y1 = int(np.clip(np.floor(y1), 0, h - 1))
    x2 = int(np.clip(np.ceil(x2), 0, w))
    y2 = int(np.clip(np.ceil(y2), 0, h))
    return x1, y1, x2, y2


def extract_bbox_intensity(gray_image: np.ndarray, bbox: np.ndarray) -> float:
    'Calculate mean grayscale intensity inside a box.'
    x1, y1, x2, y2 = _clip_bbox_to_image(bbox, (*gray_image.shape, 1))
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return float('nan')
    return float(np.mean(roi))


def extract_bbox_fluorescence_metrics(
        gray_image: np.ndarray,
        bbox: np.ndarray,
        microns_per_pixel: float,
        contour_opening_um: float) -> Tuple[dict, np.ndarray, Tuple[int, int, int, int]]:
    'Extract the dominant contour and measure intensity, area and circularity. Apply Otsu thresholding, morphological opening, largest-external-contour selection and hole filling. A contour touching its box is not automatically truncated. Mark area_um2 and circularity as NaN only when the box reaches the original image edge and the contour touches that same edge; record contour_truncated.'
    x1, y1, x2, y2 = _clip_bbox_to_image(bbox, (*gray_image.shape, 1))
    empty_metrics = {
        'intensity': float('nan'),
        'mean_intensity': float('nan'),
        'area_um2': float('nan'),
        'circularity': float('nan'),
        'threshold': float('nan'),
        'contour_truncated': False,
    }
    if x2 <= x1 or y2 <= y1:
        return empty_metrics, np.zeros((0, 0), dtype=bool), (x1, y1, x2, y2)

    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return empty_metrics, np.zeros((0, 0), dtype=bool), (x1, y1, x2, y2)

    roi_u8 = roi
    if roi_u8.dtype != np.uint8:
        roi_u8 = cv2.normalize(roi_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if int(roi_u8.max()) == int(roi_u8.min()):
        threshold_mask = roi_u8 > roi_u8.min()
        threshold = float(roi_u8.min())
    else:
        threshold, mask_u8 = cv2.threshold(
            roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_mask = mask_u8 > 0

    if not np.any(threshold_mask):
        metrics = empty_metrics.copy()
        metrics['threshold'] = float(threshold)
        return metrics, threshold_mask, (x1, y1, x2, y2)

    binary_u8 = threshold_mask.astype(np.uint8) * 255
    opening_size_px = int(round(contour_opening_um / microns_per_pixel))
    if opening_size_px > 1:
        if opening_size_px % 2 == 0:
            opening_size_px += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (opening_size_px, opening_size_px)
        )
        binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        metrics = empty_metrics.copy()
        metrics['threshold'] = float(threshold)
        return metrics, np.zeros_like(threshold_mask), (x1, y1, x2, y2)

    contour = max(contours, key=cv2.contourArea)


    mask_u8 = np.zeros_like(binary_u8)
    cv2.drawContours(mask_u8, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
    mask = mask_u8 > 0

    image_h, image_w = gray_image.shape[:2]
    roi_h, roi_w = roi_u8.shape[:2]
    contour_xy = contour.reshape(-1, 2)

    contour_touches_left = bool(np.any(contour_xy[:, 0] <= 0))
    contour_touches_top = bool(np.any(contour_xy[:, 1] <= 0))
    contour_touches_right = bool(np.any(contour_xy[:, 0] >= roi_w - 1))
    contour_touches_bottom = bool(np.any(contour_xy[:, 1] >= roi_h - 1))

    bbox_touches_left = x1 <= 0
    bbox_touches_top = y1 <= 0
    bbox_touches_right = x2 >= image_w
    bbox_touches_bottom = y2 >= image_h

    contour_truncated = bool(
        (bbox_touches_left and contour_touches_left)
        or (bbox_touches_top and contour_touches_top)
        or (bbox_touches_right and contour_touches_right)
        or (bbox_touches_bottom and contour_touches_bottom)
    )

    foreground = roi_u8[mask].astype(np.float32)
    intensity = float(np.sum(foreground))
    mean_intensity = float(np.mean(foreground))

    area_um2 = float('nan')
    circularity = float('nan')
    if not contour_truncated:
        area_px2 = float(cv2.contourArea(contour))
        perimeter_px = float(cv2.arcLength(contour, closed=True))
        if area_px2 > 0 and perimeter_px > 0:
            area_um2 = area_px2 * microns_per_pixel ** 2
            circularity = float(np.clip(4.0 * np.pi * area_px2 / perimeter_px ** 2, 0.0, 1.0))

    return {
        'intensity': intensity,
        'mean_intensity': mean_intensity,
        'area_um2': area_um2,
        'circularity': circularity,
        'threshold': float(threshold),
        'contour_truncated': contour_truncated,
    }, mask, (x1, y1, x2, y2)


# ============================================================

# ============================================================
class MatplotlibVis:
    'Reuse the plotting context to avoid creating and destroying a figure for every frame.'

    def __init__(self, h: int, w: int, dpi: int = 100):
        self.h, self.w = h, w
        self.fig, self.ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.dpi = dpi

    def draw(self, image_bgr: np.ndarray, tracks: List[STrack],
             time_min: float, line_width: int = 1) -> np.ndarray:
        'Render one frame and return a BGR image.'
        ax = self.ax


        ax.cla()
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis('off')


        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)


        for track in tracks:
            track_id = track.track_id
            bbox = track.tlbr
            color_rgb = tuple(c / 255.0 for c in PALETTE[track_id % len(PALETTE)])

            x1, y1, x2, y2 = _clip_bbox_to_image(bbox, image_bgr.shape)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=line_width, edgecolor=color_rgb, facecolor='none'
            )
            ax.add_patch(rect)


        h, w = image_bgr.shape[:2]
        ax.text(
            w - 10, 10, f"{time_min:.2f} min",
            fontsize=10, color='white', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
        )


        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        canvas_shape = self.fig.canvas.get_width_height()[::-1] + (4,)
        buf = buf.reshape(canvas_shape)

        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    def save_pdf(self, pdf_path: str):
        'Save the current figure as a vector PDF with editable boxes and text.'
        self.fig.savefig(pdf_path, format='pdf', dpi=self.dpi)

    def close(self):
        plt.close(self.fig)


# ============================================================

# ============================================================
def save_mot_results(results: List[Tuple], output_path: str):
    'Save MOT rows: frame, id, left, top, width, height, confidence, x, y, z.'
    with open(output_path, 'w') as f:
        for r in results:
            frame_id, track_id, x1, y1, x2, y2, score = r
            width = x2 - x1
            height = y2 - y1
            f.write(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{score:.4f},-1,-1,-1\n")
    print(f"MOT results saved to: {output_path}")


def save_intensity_xlsx(intensity_records: List[dict], output_path: str):
    'Export per-box mask measurements over time to an Excel workbook.'
    df = pd.DataFrame(intensity_records)
    df.to_excel(output_path, index=False)
    print(f"Metric table saved to: {output_path}")


def plot_bbox_count_over_time(counts: List[Tuple[int, int]], output_path: str):
    'Plot box count against frame index.'
    if not counts:
        return
    frames = [c[0] for c in counts]
    bbox_counts = [c[1] for c in counts]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(frames, bbox_counts, color='#0072B2', linewidth=1.2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('BBox Count')
    ax.set_title('BBox Count Over Time')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"BBox count plot saved to: {output_path}")


def save_binary_mask_pdf(mask: np.ndarray, output_path: str, time_min: float, dpi: int = 100):
    'Save a visualization PDF for the combined binary mask of each frame.'
    h, w = mask.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    ax.text(
        w - 10, 10, f"{time_min:.2f} min",
        fontsize=10, color='white', ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
    )
    fig.savefig(output_path, format='pdf', dpi=dpi)
    plt.close(fig)


def save_tracking_points_pdf(
        image_bgr: np.ndarray,
        track_histories: Dict[int, List[dict]],
        frame_id: int,
        time_min: float,
        max_track_duration: int,
        output_path: str,
        line_width: int = 1,
        dpi: int = 100):
    'Annotate observed track start and end-to-current positions and IDs on the original image.'
    h, w = image_bgr.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    max_track_duration = max(1, max_track_duration)
    for track_id, history in sorted(track_histories.items()):
        visible_history = [rec for rec in history if rec['frame'] <= frame_id]
        if not visible_history:
            continue

        start_rec = visible_history[0]
        rec = visible_history[-1]
        duration = int(rec['frame'] - start_rec['frame'] + 1)
        alpha = float(np.clip(1.0 - 0.8 * duration / max_track_duration, 0.2, 1.0))

        sx, sy = start_rec['center_x'], start_rec['center_y']
        ex, ey = rec['center_x'], rec['center_y']
        x1, y1, x2, y2 = rec['x1'], rec['y1'], rec['x2'], rec['y2']

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=line_width, edgecolor=(0.45, 0.45, 0.45, alpha),
            facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.scatter([sx], [sy], s=30, c='#1f5ac8', alpha=alpha,
                   edgecolors='black', linewidths=0.25, zorder=3)
        ax.scatter([ex], [ey], s=30, c='#e41a1c', alpha=alpha,
                   edgecolors='black', linewidths=0.25, zorder=3)
        ax.text(
            ex + 3, ey - 3, str(track_id),
            fontsize=8, color='black', ha='left', va='bottom',
            bbox=dict(boxstyle='square,pad=0.12', facecolor='white',
                      edgecolor='none', alpha=0.65)
        )

    ax.text(
        w - 10, 10, f"{time_min:.2f} min",
        fontsize=10, color='black', ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.65)
    )
    ax.scatter([28], [h - 58], s=180, c='#1f5ac8', alpha=0.9,
               edgecolors='black', linewidths=0.8, zorder=4)
    ax.text(58, h - 58, 'Start', fontsize=11, color='black', va='center')
    ax.scatter([28], [h - 28], s=180, c='#e41a1c', alpha=0.9,
               edgecolors='black', linewidths=0.8, zorder=4)
    ax.text(58, h - 28, 'End', fontsize=11, color='black', va='center')

    fig.savefig(output_path, format='pdf', dpi=dpi)
    plt.close(fig)


def save_tracking_point_pdfs(
        image_files: List[str],
        track_histories: Dict[int, List[dict]],
        frame_times_min: List[float],
        output_dir: Path,
        line_width: int):
    'Generate per-frame track endpoint PDFs on original images after processing the sequence.'
    if not track_histories:
        return

    max_track_duration = max(
        history[-1]['frame'] - history[0]['frame'] + 1
        for history in track_histories.values()
        if history
    )
    for frame_id, img_path in enumerate(tqdm(image_files, desc="Saving tracking point PDFs"), start=1):
        _, cv_image_raw = load_frame_for_inference_and_vis(img_path)
        pdf_path = output_dir / f"{Path(img_path).stem}.pdf"
        save_tracking_points_pdf(
            cv_image_raw,
            track_histories,
            frame_id,
            frame_times_min[frame_id - 1],
            max_track_duration,
            str(pdf_path),
            line_width=line_width
        )


METRIC_LABELS = {
    'intensity': 'Total Intensity',
    'area_um2': 'Area (\u03bcm\u00b2)',
    'circularity': 'Circularity',
}


def _records_to_metrics_df(metric_records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(metric_records)
    if df.empty:
        return df
    sort_cols = ['track_id', 'frame']
    return df.sort_values(sort_cols).reset_index(drop=True)


def plot_population_metric(df: pd.DataFrame, metric: str, output_path: str):
    'Plot the track mean and a translucent standard-deviation band at each time point.'
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return

    grouped = valid.groupby('time_min')[metric].agg(['mean', 'std']).reset_index()
    grouped['std'] = grouped['std'].fillna(0.0)
    x = grouped['time_min'].to_numpy(dtype=float)
    y = grouped['mean'].to_numpy(dtype=float)
    err = grouped['std'].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.fill_between(x, y - err, y + err, color='#4C78A8', alpha=0.22, linewidth=0)
    ax.plot(x, y, color='#1F4E79', linewidth=1.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f'All Tracks {METRIC_LABELS[metric]} Over Time')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Population {metric} plot saved to: {output_path}")


def plot_all_population_metrics(metric_records: List[dict], output_dir: Path):
    'Save intensity, area and circularity time courses for all tracks.'
    df = _records_to_metrics_df(metric_records)
    if df.empty:
        return
    for metric in ('intensity', 'area_um2', 'circularity'):
        plot_population_metric(df, metric, str(output_dir / f'all_tracks_{metric}_over_time.pdf'))


def plot_single_track_metrics(track_df: pd.DataFrame, output_path: str):
    'Save one PDF per track with intensity, area and circularity time courses.'
    track_id = int(track_df['track_id'].iloc[0])
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), dpi=150, sharex=True)
    for ax, metric in zip(axes, ('intensity', 'area_um2', 'circularity')):
        valid = track_df.dropna(subset=[metric])
        ax.plot(valid['time_min'], valid[metric], color='#1F4E79', linewidth=1.4, marker='o', markersize=2.5)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, linestyle='--', alpha=0.35)
    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Track {track_id} Metrics Over Time')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, format='pdf')
    plt.close(fig)


def plot_per_track_metrics(metric_records: List[dict], output_dir: Path):
    'Save per-track measurement PDFs, named by track ID.'
    df = _records_to_metrics_df(metric_records)
    if df.empty:
        return
    for track_id, track_df in tqdm(df.groupby('track_id'), desc="Saving per-track metric PDFs"):
        pdf_path = output_dir / f'track_{int(track_id):03d}_metrics.pdf'
        plot_single_track_metrics(track_df, str(pdf_path))
    print(f"Per-track metric plots saved to: {output_dir}")


def _summarize_track(track_id: int, history: List[dict]) -> dict:
    records = sorted(history, key=lambda rec: rec['frame'])
    first_frame = min(int(record.get('track_start_frame', record['frame'])) for record in records)
    return {
        'track_id': track_id,
        'records': records,
        'first_frame': first_frame,
        'last_frame': int(records[-1]['frame']),
        'first_record': records[0],
        'last_record': records[-1],
    }


def _predicted_fragment_distance_um(
        predecessor: dict,
        successor: dict,
        microns_per_pixel: float) -> float:
    'Predict a fragment endpoint using its terminal velocity and measure distance to the next fragment start.'
    records = predecessor['records']
    last = predecessor['last_record']
    pred_x = float(last['center_x'])
    pred_y = float(last['center_y'])
    frame_delta = successor['first_frame'] - predecessor['last_frame']

    if len(records) >= 2:
        prev = records[-2]
        observed_delta = int(last['frame'] - prev['frame'])
        if observed_delta > 0:
            vx = (last['center_x'] - prev['center_x']) / observed_delta
            vy = (last['center_y'] - prev['center_y']) / observed_delta
            pred_x += vx * frame_delta
            pred_y += vy * frame_delta

    first = successor['first_record']
    distance_px = np.hypot(first['center_x'] - pred_x, first['center_y'] - pred_y)
    return float(distance_px * microns_per_pixel)


def merge_track_fragments(
        track_histories: Dict[int, List[dict]],
        max_gap_frames: int,
        max_distance_um: float,
        microns_per_pixel: float) -> List[dict]:
    'Merge candidate identity fragments one-to-one using temporal gap and spatial distance.'
    summaries = {
        track_id: _summarize_track(track_id, history)
        for track_id, history in track_histories.items()
        if history
    }

    candidates = []
    for predecessor_id, predecessor in summaries.items():
        for successor_id, successor in summaries.items():
            if predecessor_id == successor_id:
                continue
            missing_frames = successor['first_frame'] - predecessor['last_frame'] - 1
            if missing_frames < 0 or missing_frames > max_gap_frames:
                continue
            distance_um = _predicted_fragment_distance_um(
                predecessor, successor, microns_per_pixel
            )
            if distance_um <= max_distance_um:
                candidates.append((missing_frames, distance_um, predecessor_id, successor_id))


    successors: Dict[int, int] = {}
    predecessors: Dict[int, int] = {}
    for _, _, predecessor_id, successor_id in sorted(candidates):
        if predecessor_id in successors or successor_id in predecessors:
            continue
        successors[predecessor_id] = successor_id
        predecessors[successor_id] = predecessor_id

    merged_tracks = []
    visited = set()
    for start_id in sorted(summaries, key=lambda tid: summaries[tid]['first_frame']):
        if start_id in predecessors or start_id in visited:
            continue
        chain_ids = []
        current_id: Optional[int] = start_id
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            chain_ids.append(current_id)
            current_id = successors.get(current_id)

        records = sorted(
            (record for track_id in chain_ids for record in summaries[track_id]['records']),
            key=lambda rec: rec['frame']
        )
        merged_tracks.append({
            'track_ids': chain_ids,
            'records': records,
            'first_frame': summaries[chain_ids[0]]['first_frame'],
            'last_frame': int(records[-1]['frame']),
            'first_record': records[0],
            'last_record': records[-1],
        })


    for track_id in summaries:
        if track_id not in visited:
            summary = summaries[track_id]
            merged_tracks.append({
                'track_ids': [track_id],
                'records': summary['records'],
                'first_frame': summary['first_frame'],
                'last_frame': summary['last_frame'],
                'first_record': summary['first_record'],
                'last_record': summary['last_record'],
            })

    return merged_tracks


def _record_is_near_image_edge(record: dict, image_shape: Tuple[int, int], margin_px: float) -> bool:
    h, w = image_shape
    return (
        record['x1'] <= margin_px
        or record['y1'] <= margin_px
        or record['x2'] >= w - margin_px
        or record['y2'] >= h - margin_px
    )


def _visible_frames(records: List[dict], gap_tolerance: int) -> set:
    'Use observed frames and fill missed-detection gaps no longer than gap_tolerance.'
    frames = sorted({int(record['frame']) for record in records})
    visible = set(frames)
    for previous, current in zip(frames, frames[1:]):
        if current - previous - 1 <= gap_tolerance:
            visible.update(range(previous + 1, current))
    return visible


def analyze_visibility_and_events(
        track_histories: Dict[int, List[dict]],
        num_frames: int,
        image_shape: Tuple[int, int],
        microns_per_pixel: float,
        min_track_frames: int,
        gap_tolerance: int,
        fragment_max_gap: int,
        fragment_max_distance_um: float,
        edge_margin_um: float) -> Tuple[List[int], List[int], List[int], dict]:
    'Construct stable trajectories and calculate visible counts and valid appearance/disappearance events.'
    merged_tracks = merge_track_fragments(
        track_histories,
        max_gap_frames=fragment_max_gap,
        max_distance_um=fragment_max_distance_um,
        microns_per_pixel=microns_per_pixel,
    )
    valid_tracks = [
        track for track in merged_tracks
        if len({record['frame'] for record in track['records']}) >= min_track_frames
    ]

    visible_counts = np.zeros(num_frames, dtype=int)
    appearance_frames = []
    disappearance_frames = []
    margin_px = edge_margin_um / microns_per_pixel

    for track in valid_tracks:
        first_near_edge = _record_is_near_image_edge(
            track['first_record'], image_shape, margin_px
        )
        last_near_edge = _record_is_near_image_edge(
            track['last_record'], image_shape, margin_px
        )
        visible_frames = _visible_frames(track['records'], gap_tolerance)
        first_detected_frame = min(visible_frames)
        if first_detected_frame - track['first_frame'] <= gap_tolerance:
            visible_frames.update(range(track['first_frame'], first_detected_frame))

        if track['first_frame'] - 1 <= gap_tolerance and not first_near_edge:
            visible_frames.update(range(1, track['first_frame']))
        if num_frames - track['last_frame'] <= gap_tolerance and not last_near_edge:
            visible_frames.update(range(track['last_frame'] + 1, num_frames + 1))
        for frame in visible_frames:
            if 1 <= frame <= num_frames:
                visible_counts[frame - 1] += 1

        if (
            track['first_frame'] > 1 + gap_tolerance
            and not first_near_edge
        ):
            appearance_frames.append(track['first_frame'])

        if (
            track['last_frame'] < num_frames - gap_tolerance
            and not last_near_edge
        ):
            disappearance_frames.append(track['last_frame'])

    diagnostics = {
        'raw_tracks': len(track_histories),
        'merged_tracks': len(merged_tracks),
        'valid_tracks': len(valid_tracks),
        'fragment_links': sum(max(0, len(track['track_ids']) - 1) for track in merged_tracks),
    }
    return visible_counts.tolist(), appearance_frames, disappearance_frames, diagnostics


def plot_visibility_and_events(
        frame_times_min: List[float],
        visible_counts: List[int],
        appearance_frames: List[int],
        disappearance_frames: List[int],
        window_min: float,
        output_path: str):
    'Plot visible object counts and appearance/disappearance events in fixed time windows.'
    if not frame_times_min:
        return

    end_time = float(frame_times_min[-1])
    plot_end = max(end_time, window_min)
    bin_edges = np.arange(0.0, plot_end + window_min, window_min)
    if len(bin_edges) < 2:
        bin_edges = np.array([0.0, window_min])

    appearance_times = [frame_times_min[frame - 1] for frame in appearance_frames]
    disappearance_times = [frame_times_min[frame - 1] for frame in disappearance_frames]
    appearance_counts, _ = np.histogram(appearance_times, bins=bin_edges)
    disappearance_counts, _ = np.histogram(disappearance_times, bins=bin_edges)

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bar_width = window_min * 0.38

    fig, (ax_visible, ax_events) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), dpi=150, sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0.12}
    )
    ax_visible.plot(frame_times_min, visible_counts, color='#174A8B', linewidth=2.0)
    ax_visible.set_ylabel('Number of visible\nmigrasomes')
    ax_visible.grid(True, axis='y', linestyle='--', alpha=0.28)

    ax_events.bar(
        centers - bar_width / 2.0, appearance_counts, width=bar_width,
        color='#9ED9E6', edgecolor='none', label='Appearance'
    )
    ax_events.bar(
        centers + bar_width / 2.0, disappearance_counts, width=bar_width,
        color='#F3A6A6', edgecolor='none', label='Disappearance'
    )
    ax_events.set_ylabel('Events')
    ax_events.set_xlabel('Time (min)')
    ax_events.legend(frameon=False, loc='upper right')
    ax_events.grid(True, axis='y', linestyle='--', alpha=0.28)
    axis_end = float(bin_edges[-1])
    ax_events.set_xlim(0.0, axis_end)
    ax_events.set_xticks(bin_edges)
    ax_events.yaxis.get_major_locator().set_params(integer=True)
    ax_visible.yaxis.get_major_locator().set_params(integer=True)

    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.11)
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Visibility/event plot saved to: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='denoise Denoise + detect + tracker Tracking')


    parser.add_argument('--denoise-weights', type=str, default=DENOISE_WEIGHTS,
                        help='Path to denoise denoising weights')
    parser.add_argument('--save-denoised', action='store_true', default=True,
                        help='Save denoised images to disk')


    parser.add_argument('--detect-weights', type=str, default=DETECT_WEIGHTS,
                        help='Path to detect weights')
    parser.add_argument('--resolution', type=int, default=DEFAULT_IMG_SIZE,
                        help='detect input resolution')
    parser.add_argument('--det-thresh', type=float, default=DEFAULT_SCORE_THR,
                        help='Detection confidence threshold')


    parser.add_argument('--input', type=str, default=INPUT_DIR,
                        help='Input folder containing images')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output folder for results')
    parser.add_argument('--frame-interval', type=float, default=FRAME_INTERVAL_SEC,
                        help=f'Frame interval in seconds (default: {FRAME_INTERVAL_SEC})')
    parser.add_argument('--microns-per-pixel', type=float, default=DEFAULT_MICRONS_PER_PIXEL,
                        help='Spatial calibration in microns per pixel (default: 176.7767 / 1024)')
    parser.add_argument('--contour-opening-um', type=float, default=DEFAULT_CONTOUR_OPENING_UM,
                        help='Elliptical opening-kernel diameter in microns for removing thin filaments')
    parser.add_argument('--save-images', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--save-mot', action='store_true', default=True,
                        help='Save results in MOT format')
    parser.add_argument('--save-video', action='store_true',
                        help='Save output as video')
    parser.add_argument('--bbox-line-width', type=int, default=1,
                        help='BBox line width for visualization')


    parser.add_argument('--track-high-thresh', type=float, default=0.40,
                        help='High confidence threshold for tracking')
    parser.add_argument('--track-low-thresh', type=float, default=0.25,
                        help='Low confidence threshold for tracking')
    parser.add_argument('--new-track-thresh', type=float, default=0.55,
                        help='Threshold for creating new tracks')
    parser.add_argument('--track-buffer', type=int, default=60,
                        help='Track buffer frames')
    parser.add_argument('--match-thresh', type=float, default=0.75,
                        help='Matching threshold (IoU)')
    parser.add_argument('--disable-appearance', action='store_true',
                        help='Disable bbox appearance feature matching')
    parser.add_argument('--appearance-weight', type=float, default=0.25,
                        help='Weight of bbox appearance distance in matching cost')
    parser.add_argument('--appearance-thresh', type=float, default=0.7,
                        help='Reject matches with appearance distance above this threshold')
    parser.add_argument('--appearance-momentum', type=float, default=0.9,
                        help='EMA momentum for updating each track appearance feature')


    parser.add_argument('--event-window-min', type=float, default=DEFAULT_EVENT_WINDOW_MIN,
                        help='Time-window width in minutes for appearance/disappearance bars')
    parser.add_argument('--event-min-track-frames', type=int, default=DEFAULT_EVENT_MIN_TRACK_FRAMES,
                        help='Minimum detected frames required for a valid track')
    parser.add_argument('--event-gap-tolerance', type=int, default=DEFAULT_EVENT_GAP_TOLERANCE,
                        help='Maximum missed frames filled when counting visible migrasomes')
    parser.add_argument('--fragment-max-gap', type=int, default=DEFAULT_FRAGMENT_MAX_GAP,
                        help='Maximum missing-frame gap for merging broken track IDs')
    parser.add_argument('--fragment-max-distance-um', type=float, default=DEFAULT_FRAGMENT_MAX_DISTANCE_UM,
                        help='Maximum endpoint distance in microns for merging broken track IDs')
    parser.add_argument('--event-edge-margin-um', type=float, default=DEFAULT_EVENT_EDGE_MARGIN_UM,
                        help='Image-edge margin in microns used to reject entry/exit events')

    args = parser.parse_args()

    if args.microns_per_pixel <= 0:
        parser.error('--microns-per-pixel must be > 0')
    if args.contour_opening_um < 0:
        parser.error('--contour-opening-um must be >= 0')
    if args.event_window_min <= 0:
        parser.error('--event-window-min must be > 0')
    if args.event_min_track_frames < 1:
        parser.error('--event-min-track-frames must be >= 1')
    if args.event_gap_tolerance < 0 or args.fragment_max_gap < 0:
        parser.error('gap tolerances must be >= 0')
    if args.fragment_max_distance_um < 0 or args.event_edge_margin_um < 0:
        parser.error('distance and edge-margin values must be >= 0')


    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    denoised_dir = output_dir / 'denoised'
    if args.save_denoised:
        denoised_dir.mkdir(exist_ok=True)

    if args.save_images:
        pdfs_dir = output_dir / 'pdfs'
        pdfs_dir.mkdir(exist_ok=True)
        tracking_points_dir = output_dir / 'tracking_points_pdfs'
        tracking_points_dir.mkdir(exist_ok=True)
        mask_pdfs_dir = output_dir / 'binary_mask_pdfs'
        mask_pdfs_dir.mkdir(exist_ok=True)

    population_plots_dir = output_dir / 'population_metric_pdfs'
    population_plots_dir.mkdir(exist_ok=True)
    per_track_plots_dir = output_dir / 'track_metric_pdfs'
    per_track_plots_dir.mkdir(exist_ok=True)

    # ============================================================

    # ============================================================
    print("=" * 60)
    print("Step 1: Loading denoise model")
    print("=" * 60)
    denoise_net, device = load_denoise_model(args.denoise_weights)

    # ============================================================

    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: Loading detect model")
    print("=" * 60)
    model = DetectionModel(
        pretrain_weights=args.detect_weights,
        resolution=args.resolution,
    )
    model.optimize_for_inference()

    # ============================================================

    # ============================================================
    image_files = get_image_files(args.input)
    if len(image_files) == 0:
        print(f"No images found in {args.input}")
        return
    print(f"\nFound {len(image_files)} images in {args.input}")


    frame_interval_sec = args.frame_interval
    frame_times_min = [(i * frame_interval_sec) / 60.0 for i in range(len(image_files))]
    print(f"Frame interval: {frame_interval_sec} sec ({frame_interval_sec / 60.0:.2f} min)")
    print(f"Time range: {frame_times_min[0]:.2f} ~ {frame_times_min[-1]:.2f} min")
    print(f"Spatial calibration: {args.microns_per_pixel:.6f} microns/pixel")

    # ============================================================

    # ============================================================
    tracking_config = TrackingConfig(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        use_appearance=not args.disable_appearance,
        appearance_weight=args.appearance_weight,
        appearance_thresh=args.appearance_thresh,
        appearance_momentum=args.appearance_momentum,
    )
    tracker = Tracker(tracking_config)
    tracker_desc = "Kalman + IoU"
    if not args.disable_appearance:
        tracker_desc += " + appearance"
    print(f"Using tracker tracker ({tracker_desc})")

    if args.det_thresh > args.track_low_thresh:
        print(
            f"[Warning] det-thresh ({args.det_thresh:.2f}) > track-low-thresh ({args.track_low_thresh:.2f}). "
            "Detections below --det-thresh are filtered before tracking."
        )

    # ============================================================

    # ============================================================
    mot_results = []
    intensity_records = []
    metric_records = []
    frame_bbox_counts = []
    track_histories = {}
    video_writer = None
    vis_ctx = None  
    frame_shape = None

    print("\n" + "=" * 60)
    print("Step 3: Processing images (denoise → detect → track)")
    print("=" * 60)

    total_t0 = _time.time()

    for frame_id, img_path in enumerate(tqdm(image_files, desc="Processing"), start=1):
        frame_t0 = _time.time()
        fname = Path(img_path).name


        t0 = _time.time()
        pil_image, cv_image_raw = load_frame_for_inference_and_vis(img_path)
        if frame_shape is None:
            frame_shape = cv_image_raw.shape[:2]
        t_load = _time.time() - t0


        t0 = _time.time()
        img_rgb_np = np.array(pil_image)
        denoised_rgb = denoise_image(denoise_net, img_rgb_np, device)
        t_denoise = _time.time() - t0

        if args.save_denoised:
            denoised_save_path = denoised_dir / fname
            cv2.imwrite(str(denoised_save_path), cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR))


        t0 = _time.time()
        denoised_pil = Image.fromarray(denoised_rgb)
        detections = model.predict(denoised_pil, threshold=args.det_thresh)
        t_detect = _time.time() - t0


        t0 = _time.time()
        det_array = detections_to_array(detections)
        cv_image_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)
        tracks = tracker.update(det_array, cv_image_bgr)
        t_track = _time.time() - t0


        gray_frame = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)
        full_frame_mask = np.zeros(gray_frame.shape, dtype=bool)
        for track in tracks:
            bbox = track.tlbr
            metrics, bbox_mask, clipped_bbox = extract_bbox_fluorescence_metrics(
                gray_frame, bbox, args.microns_per_pixel, args.contour_opening_um
            )
            x1, y1, x2, y2 = clipped_bbox
            if bbox_mask.size > 0:
                full_frame_mask[y1:y2, x1:x2] |= bbox_mask

            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            time_min = frame_times_min[frame_id - 1]

            mot_results.append((
                frame_id, track.track_id,
                bbox[0], bbox[1], bbox[2], bbox[3],
                track.score
            ))
            record = {
                'frame': frame_id,
                'track_start_frame': int(track.start_frame),
                'time_min': time_min,
                'track_id': track.track_id,
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'center_x': center_x,
                'center_y': center_y,
                'x1_um': float(x1 * args.microns_per_pixel),
                'y1_um': float(y1 * args.microns_per_pixel),
                'x2_um': float(x2 * args.microns_per_pixel),
                'y2_um': float(y2 * args.microns_per_pixel),
                'center_x_um': float(center_x * args.microns_per_pixel),
                'center_y_um': float(center_y * args.microns_per_pixel),
                'score': float(track.score),
                'intensity': metrics['intensity'],
                'mean_intensity': metrics['mean_intensity'],
                'area_um2': metrics['area_um2'],
                'circularity': metrics['circularity'],
                'threshold': metrics['threshold'],
                'contour_truncated': metrics['contour_truncated'],
            }
            intensity_records.append(record)
            metric_records.append(record)
            track_histories.setdefault(track.track_id, []).append(record)

        frame_bbox_counts.append((frame_id, len(tracks)))

        if args.save_images:
            mask_pdf_path = str(mask_pdfs_dir / Path(fname).stem) + '.pdf'
            save_binary_mask_pdf(full_frame_mask, mask_pdf_path, frame_times_min[frame_id - 1])


        t_vis = 0.0
        if args.save_images or args.save_video:
            t0 = _time.time()
            time_min = frame_times_min[frame_id - 1]


            if vis_ctx is None:
                h, w = cv_image_bgr.shape[:2]
                vis_ctx = MatplotlibVis(h, w)

            vis_image = vis_ctx.draw(cv_image_bgr, tracks, time_min,
                                     line_width=args.bbox_line_width)

            if args.save_images:

                pdf_path = str(pdfs_dir / Path(fname).stem) + '.pdf'
                vis_ctx.save_pdf(pdf_path)

            t_vis = _time.time() - t0

            if args.save_video:
                if video_writer is None:
                    h, w = vis_image.shape[:2]
                    video_path = str(output_dir / 'tracking_output.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
                video_writer.write(vis_image)

        frame_total = _time.time() - frame_t0
        det_count = len(detections)
        track_count = len(tracks)
        print(
            f"  Frame {frame_id:>3d}/{len(image_files)} {fname} | "
            f"dets={det_count:>3d} tracks={track_count:>3d} | "
            f"load={t_load:.2f}s denoise={t_denoise:.2f}s "
            f"detect={t_detect:.2f}s track={t_track:.2f}s "
            f"vis={t_vis:.2f}s total={frame_total:.2f}s"
        )


    if vis_ctx is not None:
        vis_ctx.close()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {output_dir / 'tracking_output.mp4'}")

    if args.save_images:
        save_tracking_point_pdfs(
            image_files,
            track_histories,
            frame_times_min,
            tracking_points_dir,
            args.bbox_line_width
        )

    total_elapsed = _time.time() - total_t0
    print(f"\nAll {len(image_files)} frames processed in {total_elapsed:.1f}s "
          f"({total_elapsed / len(image_files):.2f}s/frame)")

    # ============================================================

    # ============================================================
    if args.save_mot:
        mot_path = output_dir / 'tracking_results.txt'
        save_mot_results(mot_results, str(mot_path))

    intensity_path = output_dir / 'bbox_metrics_over_time.xlsx'
    save_intensity_xlsx(intensity_records, str(intensity_path))

    plot_all_population_metrics(metric_records, population_plots_dir)
    plot_per_track_metrics(metric_records, per_track_plots_dir)

    if frame_shape is not None:
        visible_counts, appearance_frames, disappearance_frames, event_diagnostics = (
            analyze_visibility_and_events(
                track_histories=track_histories,
                num_frames=len(image_files),
                image_shape=frame_shape,
                microns_per_pixel=args.microns_per_pixel,
                min_track_frames=args.event_min_track_frames,
                gap_tolerance=args.event_gap_tolerance,
                fragment_max_gap=args.fragment_max_gap,
                fragment_max_distance_um=args.fragment_max_distance_um,
                edge_margin_um=args.event_edge_margin_um,
            )
        )
        event_plot_path = output_dir / 'migrasome_visibility_and_events.pdf'
        plot_visibility_and_events(
            frame_times_min,
            visible_counts,
            appearance_frames,
            disappearance_frames,
            args.event_window_min,
            str(event_plot_path),
        )
        print(
            "Event analysis: "
            f"raw_tracks={event_diagnostics['raw_tracks']}, "
            f"merged_tracks={event_diagnostics['merged_tracks']}, "
            f"valid_tracks={event_diagnostics['valid_tracks']}, "
            f"fragment_links={event_diagnostics['fragment_links']}, "
            f"appearances={len(appearance_frames)}, "
            f"disappearances={len(disappearance_frames)}"
        )

    bbox_plot_path = output_dir / 'bbox_count_over_time.png'
    plot_bbox_count_over_time(frame_bbox_counts, str(bbox_plot_path))


    print("\n" + "=" * 60)
    print("Tracking Statistics")
    print("=" * 60)
    print(f"Total frames: {len(image_files)}")
    print(f"Total unique tracks: {len(set(r[1] for r in mot_results))}")
    print(f"Total detections: {len(mot_results)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
