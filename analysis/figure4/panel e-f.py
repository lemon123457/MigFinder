'Denoise, detect and track multiple green-channel recordings; summarize calibrated area, circularity, observed duration and speed.'

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
from typing import List, Optional, Tuple

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


INPUT_DIR = '/path/of/images/sequences'
OUTPUT_DIR = '/path/of/results/figure4_phenotypes'


FRAME_INTERVAL_SEC = 525.79828
MIN_TRACK_FRAMES = 3
DEFAULT_MICRONS_PER_PIXEL = 176.7767 / 1024.0
DEFAULT_CONTOUR_OPENING_UM = 0.5


DEFAULT_AREA_CHANGE_THRESHOLD = 0.20
DEFAULT_AREA_ENDPOINT_WINDOW = 3
DEFAULT_CIRCULARITY_LOW_THRESHOLD = 0.50
DEFAULT_CIRCULARITY_HIGH_THRESHOLD = 0.80
DEFAULT_EDGE_MARGIN_UM = 5.0
DEFAULT_END_CENSOR_TOLERANCE_FRAMES = 1
SPEED_THRESHOLDS = (2.0, 5.0)
GROUP_COLORS = ('#A9D9F0', '#BFE3AE', '#F3B4B0')
SEQUENCE_PREFIX = "20210903_SMS2GFP_TBRI-MC001_s"
SEQUENCE_START = 1
SEQUENCE_END = 30

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


def get_sequence_g_folders(input_root: str, prefix: str, start: int, end: int) -> List[Path]:
    'Find the selected s001-s030 sequence G-channel folders under the input root.'
    root = Path(input_root)
    if root.name == 'G':
        return [root]

    sequence_dirs = []
    for idx in range(start, end + 1):
        g_dir = root / f"{prefix}{idx:03d}" / 'G'
        if g_dir.is_dir():
            sequence_dirs.append(g_dir)
        else:
            print(f"[Warning] Missing sequence G folder: {g_dir}")
    return sequence_dirs


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


def extract_bbox_contour_metrics(
        gray_image: np.ndarray,
        bbox: np.ndarray,
        microns_per_pixel: float,
        contour_opening_um: float) -> Tuple[dict, Tuple[int, int, int, int]]:
    'Extract the dominant contour and measure calibrated area and circularity. Apply Otsu thresholding, elliptical opening, largest-external-contour selection and hole filling consistently across sequences. Set area_um2 and circularity to NaN and contour_truncated=True only when the box and contour both touch the same original-image edge.'
    x1, y1, x2, y2 = _clip_bbox_to_image(bbox, (*gray_image.shape, 1))
    empty_metrics = {
        'area_um2': float('nan'),
        'circularity': float('nan'),
        'threshold': float('nan'),
        'contour_truncated': False,
        'contour_valid': False,
    }
    if x2 <= x1 or y2 <= y1:
        return empty_metrics, (x1, y1, x2, y2)

    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return empty_metrics, (x1, y1, x2, y2)

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
        return metrics, (x1, y1, x2, y2)

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
        return metrics, (x1, y1, x2, y2)

    contour = max(contours, key=cv2.contourArea)


    filled_mask = np.zeros_like(binary_u8)
    cv2.drawContours(filled_mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

    image_h, image_w = gray_image.shape[:2]
    roi_h, roi_w = roi_u8.shape[:2]
    contour_xy = contour.reshape(-1, 2)
    contour_touches_left = bool(np.any(contour_xy[:, 0] <= 0))
    contour_touches_top = bool(np.any(contour_xy[:, 1] <= 0))
    contour_touches_right = bool(np.any(contour_xy[:, 0] >= roi_w - 1))
    contour_touches_bottom = bool(np.any(contour_xy[:, 1] >= roi_h - 1))

    contour_truncated = bool(
        (x1 <= 0 and contour_touches_left)
        or (y1 <= 0 and contour_touches_top)
        or (x2 >= image_w and contour_touches_right)
        or (y2 >= image_h and contour_touches_bottom)
    )

    area_um2 = float('nan')
    circularity = float('nan')
    if not contour_truncated:
        area_px2 = float(cv2.contourArea(contour))
        perimeter_px = float(cv2.arcLength(contour, closed=True))
        if area_px2 > 0 and perimeter_px > 0:
            area_um2 = area_px2 * microns_per_pixel ** 2
            circularity = float(np.clip(
                4.0 * np.pi * area_px2 / perimeter_px ** 2, 0.0, 1.0
            ))

    return {
        'area_um2': area_um2,
        'circularity': circularity,
        'threshold': float(threshold),
        'contour_truncated': contour_truncated,
        'contour_valid': bool(np.isfinite(area_um2) and np.isfinite(circularity)),
    }, (x1, y1, x2, y2)


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


def save_metrics_xlsx(metric_records: List[dict], output_path: str):
    'Export per-box mask measurements over time to an Excel workbook.'
    df = pd.DataFrame(metric_records)
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


def filter_short_tracks(metric_records: List[dict], min_frames: int) -> pd.DataFrame:
    'Retain tracks with at least min_frames observed frames and add time-alignment fields.'
    df = pd.DataFrame(metric_records)
    if df.empty:
        return df
    counts = df.groupby('global_track_id')['frame'].nunique()
    keep_ids = counts[counts >= min_frames].index
    filtered = df[df['global_track_id'].isin(keep_ids)].copy()
    if filtered.empty:
        return filtered

    first_frame = filtered.groupby('global_track_id')['frame'].transform('min').astype(int)
    last_frame = filtered.groupby('global_track_id')['frame'].transform('max').astype(int)
    frame_count = filtered.groupby('global_track_id')['frame'].transform('nunique')
    filtered['first_detection_frame'] = first_frame
    filtered['last_detection_frame'] = last_frame
    filtered['track_frame_count'] = frame_count
    filtered['time_since_first_detection_min'] = (
        filtered['frame'] - filtered['first_detection_frame']
    ) * filtered['frame_interval_min']
    filtered['observed_duration_min'] = (
        filtered['last_detection_frame'] - filtered['first_detection_frame']
    ) * filtered['frame_interval_min']
    return filtered.sort_values(['sequence_id', 'global_track_id', 'frame']).reset_index(drop=True)


def _record_is_near_image_edge(record: pd.Series, image_h: int, image_w: int,
                               margin_px: float) -> bool:
    return bool(
        float(record['x1']) <= margin_px
        or float(record['y1']) <= margin_px
        or float(record['x2']) >= image_w - margin_px
        or float(record['y2']) >= image_h - margin_px
    )


def _classify_area_change(relative_change: float, threshold: float) -> str:
    if not np.isfinite(relative_change):
        return ''
    if relative_change > threshold:
        return 'Expanding'
    if relative_change < -threshold:
        return 'Contracting'
    return 'Stable'


def _classify_circularity(value: float, low_threshold: float,
                          high_threshold: float) -> str:
    if not np.isfinite(value):
        return ''
    if value < low_threshold:
        return 'Low'
    if value < high_threshold:
        return 'Medium'
    return 'High'


def _classify_speed(value: float, slow_threshold: float,
                    fast_threshold: float) -> str:
    if not np.isfinite(value):
        return ''
    if value <= slow_threshold:
        return 'Slow'
    if value <= fast_threshold:
        return 'Medium'
    return 'Fast'


def build_track_summary(
        df: pd.DataFrame,
        sequence_metadata: List[dict],
        min_reliable_frames: int,
        area_endpoint_window: int,
        area_change_threshold: float,
        circularity_low_threshold: float,
        circularity_high_threshold: float,
        microns_per_pixel: float,
        edge_margin_um: float,
        end_censor_tolerance_frames: int) -> pd.DataFrame:
    'Create one summary row per track and record censoring reasons.'
    if df.empty:
        return pd.DataFrame()

    metadata = {str(item['sequence_id']): item for item in sequence_metadata}
    records = []
    for global_track_id, track_df in df.groupby('global_track_id', sort=False):
        track_df = track_df.sort_values('frame').drop_duplicates('frame', keep='last')
        first = track_df.iloc[0]
        last = track_df.iloc[-1]
        sequence_id = str(first['sequence_id'])
        sequence_meta = metadata[sequence_id]
        first_frame = int(track_df['frame'].min())
        last_frame = int(track_df['frame'].max())
        observed_duration_min = float(last['time_min'] - first['time_min'])

        area_values = track_df['area_um2'].to_numpy(dtype=float)
        area_values = area_values[np.isfinite(area_values)]
        circularity_values = track_df['circularity'].to_numpy(dtype=float)
        circularity_values = circularity_values[np.isfinite(circularity_values)]
        mean_area_um2 = float(np.mean(area_values)) if area_values.size else float('nan')
        mean_circularity = (
            float(np.mean(circularity_values)) if circularity_values.size else float('nan')
        )
        area_reliable = area_values.size >= min_reliable_frames
        circularity_reliable = circularity_values.size >= min_reliable_frames

        initial_area_um2 = float('nan')
        final_area_um2 = float('nan')
        relative_area_change = float('nan')
        endpoint_frames_used = 0
        if area_reliable:

            endpoint_frames_used = min(area_endpoint_window, max(1, area_values.size // 2))
            initial_area_um2 = float(np.mean(area_values[:endpoint_frames_used]))
            final_area_um2 = float(np.mean(area_values[-endpoint_frames_used:]))
            if initial_area_um2 > 0:
                relative_area_change = (final_area_um2 - initial_area_um2) / initial_area_um2

        centers_x_um = track_df['center_x_um'].to_numpy(dtype=float)
        centers_y_um = track_df['center_y_um'].to_numpy(dtype=float)
        times_min = track_df['time_min'].to_numpy(dtype=float)
        frame_steps = np.diff(track_df['frame'].to_numpy(dtype=int))
        dt_h = np.diff(times_min) / 60.0
        distances_um = np.hypot(np.diff(centers_x_um), np.diff(centers_y_um))

        valid_segments = (
            np.isfinite(distances_um)
            & np.isfinite(dt_h)
            & (dt_h > 0)
            & (frame_steps == 1)
        )
        segment_speeds = distances_um[valid_segments] / dt_h[valid_segments]
        mean_speed_um_h = (
            float(np.mean(segment_speeds)) if segment_speeds.size else float('nan')
        )
        total_path_length_um = (
            float(np.sum(distances_um[valid_segments])) if np.any(valid_segments) else float('nan')
        )
        speed_reliable = segment_speeds.size >= max(1, min_reliable_frames - 1)

        num_frames = int(sequence_meta['num_frames'])
        near_edge = _record_is_near_image_edge(
            last,
            int(sequence_meta['image_height_px']),
            int(sequence_meta['image_width_px']),
            edge_margin_um / microns_per_pixel,
        )
        if last_frame >= num_frames - end_censor_tolerance_frames:
            event_status = 'right_censored_video_end'
            disappearance_confirmed = False
        elif near_edge:
            event_status = 'right_censored_field_exit'
            disappearance_confirmed = False
        else:
            event_status = 'confirmed_disappearance'
            disappearance_confirmed = True

        records.append({
            'sequence_id': sequence_id,
            'local_track_id': int(first['local_track_id']),
            'global_track_id': global_track_id,
            'first_detection_frame': first_frame,
            'last_detection_frame': last_frame,
            'track_frame_count': int(track_df['frame'].nunique()),
            'valid_area_frame_count': int(area_values.size),
            'valid_circularity_frame_count': int(circularity_values.size),
            'observed_duration_min': observed_duration_min,
            'mean_area_um2': mean_area_um2,
            'mean_circularity': mean_circularity,
            'area_reliable': bool(area_reliable),
            'circularity_reliable': bool(circularity_reliable),
            'area_endpoint_frames_used': endpoint_frames_used,
            'initial_area_um2': initial_area_um2,
            'final_area_um2': final_area_um2,
            'relative_area_change': relative_area_change,
            'relative_area_change_percent': relative_area_change * 100.0,
            'area_change_type': _classify_area_change(
                relative_area_change, area_change_threshold
            ),
            'speed_segment_count': int(segment_speeds.size),
            'speed_reliable': bool(speed_reliable),
            'mean_speed_um_h': mean_speed_um_h,
            'total_path_length_um': total_path_length_um,
            'speed_group': _classify_speed(
                mean_speed_um_h if speed_reliable else float('nan'),
                *SPEED_THRESHOLDS,
            ),
            'circularity_class': _classify_circularity(
                mean_circularity if circularity_reliable else float('nan'),
                circularity_low_threshold,
                circularity_high_threshold,
            ),
            'last_detection_near_edge': near_edge,
            'disappearance_confirmed': disappearance_confirmed,
            'right_censored': not disappearance_confirmed,
            'event_status': event_status,
            'sequence_num_frames': num_frames,
        })

    summary = pd.DataFrame(records).sort_values(
        ['sequence_id', 'local_track_id']
    ).reset_index(drop=True)
    summary['sequence_valid_track_count'] = summary.groupby('sequence_id')[
        'global_track_id'
    ].transform('nunique')
    for category_col, output_col in (
        ('area_change_type', 'sequence_area_reliable_track_count'),
        ('circularity_class', 'sequence_circularity_reliable_track_count'),
        ('speed_group', 'sequence_speed_reliable_track_count'),
    ):
        counts = summary[summary[category_col] != ''].groupby('sequence_id')[
            'global_track_id'
        ].nunique()
        summary[output_col] = summary['sequence_id'].map(counts).fillna(0).astype(int)
    return summary


def build_category_proportions(
        track_summary: pd.DataFrame,
        sequence_ids: List[str],
        category_column: str,
        categories: List[str]) -> pd.DataFrame:
    'Calculate class fractions within each sequence; keep NaN when reliable tracks are unavailable.'
    rows = []
    for sequence_id in sequence_ids:
        sequence_df = track_summary[
            (track_summary['sequence_id'] == sequence_id)
            & (track_summary[category_column] != '')
        ]
        denominator = int(sequence_df['global_track_id'].nunique())
        for category in categories:
            count = int((sequence_df[category_column] == category).sum())
            rows.append({
                'sequence_id': sequence_id,
                'category': category,
                'track_count': count,
                'valid_track_total': denominator,
                'fraction_percent': 100.0 * count / denominator if denominator else float('nan'),
            })
    return pd.DataFrame(rows)


def summarize_category_proportions(proportion_df: pd.DataFrame) -> pd.DataFrame:
    'Summarize fraction means and sample standard deviations across sequences.'
    if proportion_df.empty:
        return pd.DataFrame()
    valid = proportion_df.dropna(subset=['fraction_percent'])
    summary = (
        valid.groupby('category', sort=False)['fraction_percent']
        .agg(sequence_count='count', mean_fraction_percent='mean',
             sd_fraction_percent='std')
        .reset_index()
    )
    summary['sd_fraction_percent'] = summary['sd_fraction_percent'].fillna(0.0)
    return summary


def build_aligned_heatmap_matrices(
        df: pd.DataFrame,
        track_summary: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    'Build area and circularity matrices with the same tracks, row order and time axis.'
    if df.empty or track_summary.empty:
        empty = np.empty((0, 0))
        return empty, empty, np.empty((0,)), pd.DataFrame()

    eligible = track_summary[
        (track_summary['valid_area_frame_count'] > 0)
        & (track_summary['valid_circularity_frame_count'] > 0)
    ].copy()
    sort_columns = [
        'observed_duration_min', 'track_frame_count',
        'sequence_id', 'local_track_id',
    ]
    eligible = eligible.sort_values(sort_columns, ascending=True).reset_index(drop=True)
    if eligible.empty:
        empty = np.empty((0, 0))
        return empty, empty, np.empty((0,)), eligible

    max_age_frame = int(
        (eligible['last_detection_frame'] - eligible['first_detection_frame']).max()
    )
    interval_min = float(df['frame_interval_min'].iloc[0])
    time_values = np.arange(max_age_frame + 1, dtype=float) * interval_min
    area_matrix = np.full((len(eligible), max_age_frame + 1), np.nan, dtype=float)
    circularity_matrix = np.full_like(area_matrix, np.nan)
    row_lookup = {
        track_id: row
        for row, track_id in enumerate(eligible['global_track_id'].tolist())
    }

    for track_id, track_df in df[df['global_track_id'].isin(row_lookup)].groupby(
            'global_track_id'):
        row = row_lookup[track_id]
        first_frame = int(eligible.iloc[row]['first_detection_frame'])
        ages = (track_df['frame'].astype(int) - first_frame).to_numpy()
        valid_age = (ages >= 0) & (ages <= max_age_frame)
        ages = ages[valid_age]
        area_matrix[row, ages] = track_df.loc[valid_age, 'area_um2'].to_numpy(dtype=float)
        circularity_matrix[row, ages] = track_df.loc[
            valid_age, 'circularity'
        ].to_numpy(dtype=float)

    row_order = eligible[[
        'sequence_id', 'local_track_id', 'global_track_id',
        'first_detection_frame', 'last_detection_frame',
        'track_frame_count', 'observed_duration_min',
        'area_change_type', 'relative_area_change',
    ]].copy()
    row_order.insert(0, 'heatmap_row', np.arange(1, len(row_order) + 1))
    return area_matrix, circularity_matrix, time_values, row_order


def heatmap_matrix_to_dataframe(
        matrix: np.ndarray,
        time_values: np.ndarray,
        row_order: pd.DataFrame) -> pd.DataFrame:
    columns = [f'time_min_{value:.6f}' for value in time_values]
    source = pd.DataFrame(matrix, columns=columns)
    if not row_order.empty:
        source.insert(0, 'global_track_id', row_order['global_track_id'].to_numpy())
        source.insert(0, 'heatmap_row', row_order['heatmap_row'].to_numpy())
    return source


def _label_saturated_colorbar_endpoints(
        colorbar,
        vmin: float,
        vmax: float,
        clipped_below: bool,
        clipped_above: bool):
    'Indicate clipped color limits with endpoint labels instead of extension triangles.'
    ticks = colorbar.get_ticks()
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
    ticks = np.unique(np.concatenate(([vmin], ticks, [vmax])))
    labels = [f'{tick:g}' for tick in ticks]
    if clipped_below:
        labels[0] = f'≤{vmin:g}'
    if clipped_above:
        labels[-1] = f'≥{vmax:g}'
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(labels)


def plot_area_circularity_heatmaps(
        area_matrix: np.ndarray,
        circularity_matrix: np.ndarray,
        time_values: np.ndarray,
        row_order: pd.DataFrame,
        output_path: str,
        area_vmin: float = 0.0,
        area_vmax: Optional[float] = None,
        circularity_vmin: float = 0.0,
        circularity_vmax: float = 1.0):
    'Plot a heatmap with transparent missing values over a separate light-gray vector background.'
    if area_matrix.size == 0:
        return

    n_tracks = area_matrix.shape[0]
    n_sequences = int(row_order['sequence_id'].nunique())
    fig_height = min(14.0, max(4.8, 3.8 + n_tracks * 0.025))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, fig_height), dpi=150,
                             constrained_layout=True, sharey=True)
    half_interval = (time_values[1] - time_values[0]) / 2.0 if len(time_values) > 1 else 0.5
    extent = [
        -half_interval,
        float(time_values[-1] + half_interval),
        n_tracks - 0.5,
        -0.5,
    ]

    area_colors = ['#dce2df', '#204281']
    circularity_colors = ['#dfe7dd', '#2a5f32']
    area_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'area_blue_green', area_colors
    )
    circularity_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'circularity_blue_green', circularity_colors
    )

    
    area_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    circularity_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    area_cmap.set_under(area_colors[0])
    area_cmap.set_over(area_colors[-1])
    circularity_cmap.set_under(circularity_colors[0])
    circularity_cmap.set_over(circularity_colors[-1])
    finite_area = area_matrix[np.isfinite(area_matrix)]
    if area_vmax is None:
        area_vmax = float(np.max(finite_area)) if finite_area.size else 1.0
    if area_vmax <= area_vmin:
        raise ValueError('area_vmax must be greater than area_vmin')
    if circularity_vmax <= circularity_vmin:
        raise ValueError('circularity_vmax must be greater than circularity_vmin')
    area_below = bool(
        finite_area.size and float(np.min(finite_area)) < area_vmin
    )
    area_above = bool(
        finite_area.size and float(np.max(finite_area)) > area_vmax
    )
    finite_circularity = circularity_matrix[np.isfinite(circularity_matrix)]
    circularity_below = bool(
        finite_circularity.size
        and float(np.min(finite_circularity)) < circularity_vmin
    )
    circularity_above = bool(
        finite_circularity.size
        and float(np.max(finite_circularity)) > circularity_vmax
    )

    background_y = min(extent[2], extent[3])
    background_height = abs(extent[2] - extent[3])
    for ax in axes:
        ax.add_patch(patches.Rectangle(
            (extent[0], background_y),
            extent[1] - extent[0],
            background_height,
            facecolor='#D9D9D9',
            edgecolor='none',
            zorder=-2,
        ))

    area_im = axes[0].imshow(
        np.ma.masked_invalid(area_matrix), aspect='auto', interpolation='nearest',
        cmap=area_cmap, extent=extent, vmin=area_vmin, vmax=area_vmax,
    )
    circularity_im = axes[1].imshow(
        np.ma.masked_invalid(circularity_matrix), aspect='auto', interpolation='nearest',
        cmap=circularity_cmap, extent=extent,
        vmin=circularity_vmin, vmax=circularity_vmax,
    )
    axes[0].set_title('Area (absolute)')
    axes[1].set_title('Circularity')
    for ax in axes:
        ax.set_xlabel('Time since first detection (min)')

        ax.set_xlim(left=0.0)
    axes[0].set_ylabel('Tracks (short to long duration)')
    area_colorbar = fig.colorbar(
        area_im, ax=axes[0], fraction=0.046, pad=0.02,
        label='Area (μm²)',
    )
    circularity_colorbar = fig.colorbar(
        circularity_im, ax=axes[1], fraction=0.046, pad=0.02,
        label='Circularity',
    )
    _label_saturated_colorbar_endpoints(
        area_colorbar, area_vmin, area_vmax, area_below, area_above
    )
    _label_saturated_colorbar_endpoints(
        circularity_colorbar,
        circularity_vmin,
        circularity_vmax,
        circularity_below,
        circularity_above,
    )

    if n_tracks <= 40:
        labels = row_order['global_track_id'].tolist()
        axes[0].set_yticks(np.arange(n_tracks))
        axes[0].set_yticklabels(labels, fontsize=5)
    else:
        axes[0].set_yticks([])
    fig.suptitle(
        'First-detection-aligned migrasome tracks; sorted by observed duration\n'
        f'n={n_tracks}, sequences={n_sequences}'
    )
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Area/circularity heatmaps saved to: {output_path}")


def _paired_wilcoxon_p(
        proportion_df: pd.DataFrame, category_a: str, category_b: str) -> Tuple[float, int]:
    'Apply a paired two-sided Wilcoxon test to two class fractions from the same sequences.'
    paired = proportion_df.pivot(
        index='sequence_id', columns='category', values='fraction_percent'
    )
    if category_a not in paired or category_b not in paired:
        return float('nan'), 0
    paired = paired[[category_a, category_b]].dropna()
    if len(paired) < 2:
        return float('nan'), len(paired)
    differences = paired[category_a].to_numpy() - paired[category_b].to_numpy()
    if np.allclose(differences, 0.0):
        return 1.0, len(paired)
    try:
        from scipy import stats
        pvalue = stats.wilcoxon(
            paired[category_a], paired[category_b],
            alternative='two-sided', zero_method='wilcox'
        ).pvalue
        return float(pvalue), len(paired)
    except Exception:
        return float('nan'), len(paired)


def _format_pvalue(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return 'P = n/a'
    if pvalue < 0.0001:
        return 'P < 0.0001'
    if pvalue < 0.001:
        return f'P = {pvalue:.4f}'
    if pvalue < 0.05:
        return f'P = {pvalue:.3f}'
    return f'ns\nP = {pvalue:.2f}'


def build_pairwise_wilcoxon_results(
        proportion_df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    rows = []
    for left, right in ((0, 1), (1, 2), (0, 2)):
        pvalue, pair_count = _paired_wilcoxon_p(
            proportion_df, categories[left], categories[right]
        )
        rows.append({
            'group_a': categories[left],
            'group_b': categories[right],
            'test': 'paired two-sided Wilcoxon signed-rank',
            'paired_sequence_count': pair_count,
            'pvalue_raw': pvalue,
        })
    results = pd.DataFrame(rows)
    results['pvalue_holm'] = np.nan
    finite_indices = results.index[np.isfinite(results['pvalue_raw'])].tolist()
    ranked_indices = sorted(finite_indices, key=lambda index: results.at[index, 'pvalue_raw'])
    running_max = 0.0
    comparison_count = len(ranked_indices)
    for rank, index in enumerate(ranked_indices):
        adjusted = min(1.0, (comparison_count - rank) * results.at[index, 'pvalue_raw'])
        running_max = max(running_max, adjusted)
        results.at[index, 'pvalue_holm'] = running_max
    return results


def _add_significance_bar(ax, x1: float, x2: float, y: float,
                          height: float, text: str):
    ax.plot(
        [x1, x1, x2, x2], [y, y + height, y + height, y],
        color='black', linewidth=0.9, clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2.0, y + height + 0.8, text,
        ha='center', va='bottom', fontsize=8,
    )


def plot_fraction_violin(
        proportion_df: pd.DataFrame,
        categories: List[str],
        title: str,
        output_path: str) -> pd.DataFrame:
    'Plot group-colored violins and sequence points with paired pairwise tests for the three groups.'
    if proportion_df.empty:
        return pd.DataFrame()

    groups = []
    for category in categories:
        values = proportion_df.loc[
            proportion_df['category'] == category, 'fraction_percent'
        ].to_numpy(dtype=float)
        groups.append(values[np.isfinite(values)])

    positions = np.arange(1, len(categories) + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.3), dpi=150)
    for position, values, color in zip(positions, groups, GROUP_COLORS):
        if len(values) >= 2 and not np.allclose(values, values[0]):
            parts = ax.violinplot(
                [values], positions=[position], widths=0.72,
                showmeans=False, showmedians=False, showextrema=False,
            )
            body = parts['bodies'][0]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)

        rng = np.random.default_rng(100 + int(position))
        jitter = rng.normal(0.0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter, values,
            s=22, color=color, alpha=0.85, edgecolors='white',
            linewidths=0.35, zorder=3,
        )

    for position, values in zip(positions, groups):
        if values.size:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ax.errorbar(
                position, mean, yerr=std, fmt='D', color='black',
                markersize=5, capsize=4, linewidth=1.4, zorder=4,
            )

    test_results = build_pairwise_wilcoxon_results(proportion_df, categories)
    bar_bottoms = [105.0, 117.0, 129.0]
    category_positions = {category: position for category, position in zip(categories, positions)}
    for (_, result), bar_bottom in zip(test_results.iterrows(), bar_bottoms):
        pvalue = float(result['pvalue_holm'])
        _add_significance_bar(
            ax,
            category_positions[result['group_a']],
            category_positions[result['group_b']],
            bar_bottom, 2.0,
            _format_pvalue(pvalue),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Fraction of tracks (%)')
    ax.set_xlim(0.45, len(categories) + 0.55)
    ax.set_ylim(0, 145)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.28)
    ax.text(
        0.0, -0.15,
        'Each point is one sequence; black diamond shows mean ± SD.\n'
        'Pairwise tests: paired two-sided Wilcoxon signed-rank; Holm-adjusted P.',
        transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
    )
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.22)
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Fraction violin plot saved to: {output_path}")
    return test_results


def plot_lifetime_survival(track_summary: pd.DataFrame, output_path: str):
    'Plot a Kaplan-Meier curve with right censoring at the recording end or field-of-view exit.'
    valid = track_summary.dropna(subset=['observed_duration_min']).copy()
    if valid.empty:
        return

    durations = valid['observed_duration_min'].to_numpy(dtype=float)
    events = valid['disappearance_confirmed'].to_numpy(dtype=bool)
    unique_times = np.sort(np.unique(durations))
    at_risk = len(valid)
    survival = 1.0
    plot_times = [0.0]
    plot_survival = [1.0]
    median_duration = float('nan')
    for current_time in unique_times:
        at_time = durations == current_time
        event_count = int(np.count_nonzero(at_time & events))
        censor_count = int(np.count_nonzero(at_time & ~events))
        if event_count and at_risk > 0:
            survival *= 1.0 - event_count / at_risk
            if not np.isfinite(median_duration) and survival <= 0.5:
                median_duration = float(current_time)
        plot_times.append(float(current_time))
        plot_survival.append(survival)
        at_risk -= event_count + censor_count

    n_tracks = len(valid)
    n_sequences = int(valid['sequence_id'].nunique())
    confirmed = int(np.count_nonzero(events))
    video_end = int((valid['event_status'] == 'right_censored_video_end').sum())
    field_exit = int((valid['event_status'] == 'right_censored_field_exit').sum())

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.step(plot_times, plot_survival, where='post', color='black', linewidth=1.7)
    if np.isfinite(median_duration):
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(median_duration, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Observed duration (min)')
    ax.set_ylabel('Survival probability')
    ax.set_title('Migrasome track duration (Kaplan–Meier)', fontsize=11)
    ax.text(
        0.04, 0.05,
        f'n={n_tracks} tracks; {n_sequences} sequences\n'
        f'confirmed disappearance={confirmed}\n'
        f'censored: video end={video_end}, field exit={field_exit}',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Censor-aware survival plot saved to: {output_path}")


def process_sequence(
        g_dir: Path,
        sequence_output_dir: Path,
        denoise_net,
        device,
        model,
        args) -> Tuple[List[Tuple], List[dict], List[Tuple[int, int]], dict]:
    'Process one G-channel folder through denoise, detect and track.'
    sequence_name = g_dir.parent.name
    sequence_output_dir.mkdir(parents=True, exist_ok=True)

    image_files = get_image_files(str(g_dir))
    if len(image_files) == 0:
        print(f"[Warning] No images found in {g_dir}")
        return [], [], [], {
            'sequence_id': sequence_name,
            'num_frames': 0,
            'image_height_px': 0,
            'image_width_px': 0,
        }

    frame_interval_min = args.frame_interval / 60.0
    frame_times_min = [(i * args.frame_interval) / 60.0 for i in range(len(image_files))]

    tracking_config = TrackingConfig(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        use_appearance=True,
        appearance_weight=args.appearance_weight,
        appearance_thresh=args.appearance_thresh,
        appearance_momentum=args.appearance_momentum,
    )
    STrack.reset_id()
    tracker = Tracker(tracking_config)

    mot_results = []
    metric_records = []
    frame_bbox_counts = []
    video_writer = None
    vis_ctx = None
    image_height_px = 0
    image_width_px = 0

    print("\n" + "=" * 60)
    print(f"Processing sequence: {sequence_name}")
    print(f"Input G folder: {g_dir}")
    print(f"Frames: {len(image_files)}")
    print("=" * 60)

    for frame_id, img_path in enumerate(tqdm(image_files, desc=sequence_name), start=1):
        frame_t0 = _time.time()
        fname = Path(img_path).name

        t0 = _time.time()
        pil_image, _ = load_frame_for_inference_and_vis(img_path)
        t_load = _time.time() - t0

        t0 = _time.time()
        img_rgb_np = np.array(pil_image)
        denoised_rgb = denoise_image(denoise_net, img_rgb_np, device)
        t_denoise = _time.time() - t0

        t0 = _time.time()
        denoised_pil = Image.fromarray(denoised_rgb)
        detections = model.predict(denoised_pil, threshold=args.det_thresh)
        t_detect = _time.time() - t0

        t0 = _time.time()
        det_array = detections_to_array(detections)
        cv_image_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)
        if image_height_px == 0:
            image_height_px, image_width_px = cv_image_bgr.shape[:2]
        tracks = tracker.update(det_array, cv_image_bgr)
        t_track = _time.time() - t0

        gray_frame = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)
        time_min = frame_times_min[frame_id - 1]
        for track in tracks:
            bbox = track.tlbr
            metrics, clipped_bbox = extract_bbox_contour_metrics(
                gray_frame,
                bbox,
                args.microns_per_pixel,
                args.contour_opening_um,
            )
            x1, y1, x2, y2 = clipped_bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            local_track_id = int(track.track_id)
            global_track_id = f"{sequence_name}_T{local_track_id:04d}"

            mot_results.append((
                frame_id, local_track_id,
                bbox[0], bbox[1], bbox[2], bbox[3],
                track.score
            ))
            metric_records.append({
                'sequence': sequence_name,
                'sequence_id': sequence_name,
                'frame': frame_id,
                'time_min': time_min,
                'frame_interval_min': frame_interval_min,
                'local_track_id': local_track_id,
                'global_track_id': global_track_id,
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'center_x': float(center_x),
                'center_y': float(center_y),
                'x1_um': float(x1 * args.microns_per_pixel),
                'y1_um': float(y1 * args.microns_per_pixel),
                'x2_um': float(x2 * args.microns_per_pixel),
                'y2_um': float(y2 * args.microns_per_pixel),
                'center_x_um': float(center_x * args.microns_per_pixel),
                'center_y_um': float(center_y * args.microns_per_pixel),
                'score': float(track.score),
                'area_um2': metrics['area_um2'],
                'circularity': metrics['circularity'],
                'threshold': metrics['threshold'],
                'contour_truncated': metrics['contour_truncated'],
                'contour_valid': metrics['contour_valid'],
            })

        frame_bbox_counts.append((frame_id, len(tracks)))

        t_vis = 0.0
        if args.save_video:
            t0 = _time.time()
            if vis_ctx is None:
                h, w = cv_image_bgr.shape[:2]
                vis_ctx = MatplotlibVis(h, w)

            vis_image = vis_ctx.draw(cv_image_bgr, tracks, time_min,
                                     line_width=args.bbox_line_width)

            t_vis = _time.time() - t0

            if args.save_video:
                if video_writer is None:
                    h, w = vis_image.shape[:2]
                    video_path = str(sequence_output_dir / 'tracking_output.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
                video_writer.write(vis_image)

        frame_total = _time.time() - frame_t0
        print(
            f"  Frame {frame_id:>3d}/{len(image_files)} {fname} | "
            f"dets={len(detections):>3d} tracks={len(tracks):>3d} | "
            f"load={t_load:.2f}s denoise={t_denoise:.2f}s "
            f"detect={t_detect:.2f}s track={t_track:.2f}s "
            f"vis={t_vis:.2f}s total={frame_total:.2f}s"
        )

    if vis_ctx is not None:
        vis_ctx.close()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {sequence_output_dir / 'tracking_output.mp4'}")

    if args.save_mot:
        mot_path = sequence_output_dir / 'tracking_results.txt'
        save_mot_results(mot_results, str(mot_path))

    metrics_path = sequence_output_dir / 'bbox_metrics_over_time.xlsx'
    save_metrics_xlsx(metric_records, str(metrics_path))

    bbox_plot_path = sequence_output_dir / 'bbox_count_over_time.png'
    plot_bbox_count_over_time(frame_bbox_counts, str(bbox_plot_path))

    sequence_metadata = {
        'sequence_id': sequence_name,
        'num_frames': len(image_files),
        'image_height_px': image_height_px,
        'image_width_px': image_width_px,
    }
    return mot_results, metric_records, frame_bbox_counts, sequence_metadata


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='denoise Denoise + detect + tracker Tracking')


    parser.add_argument('--denoise-weights', type=str, default=DENOISE_WEIGHTS,
                        help='Path to denoise denoising weights')


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
    parser.add_argument('--microns-per-pixel', '--pixel-size-um',
                        dest='microns_per_pixel', type=float,
                        default=DEFAULT_MICRONS_PER_PIXEL,
                        help='Spatial calibration in μm/pixel (default: 176.7767 / 1024)')
    parser.add_argument('--contour-opening-um', type=float,
                        default=DEFAULT_CONTOUR_OPENING_UM,
                        help='Opening-kernel diameter in μm for removing thin filaments')
    parser.add_argument('--save-mot', action=argparse.BooleanOptionalAction,
                        default=True, help='Save results in MOT format')
    parser.add_argument('--save-video', action=argparse.BooleanOptionalAction,
                        default=False, help='Save output as video')
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
    parser.add_argument('--appearance-weight', type=float, default=0.25,
                        help='Weight of bbox appearance distance in matching cost')
    parser.add_argument('--appearance-thresh', type=float, default=0.70,
                        help='Reject matches above this appearance distance')
    parser.add_argument('--appearance-momentum', type=float, default=0.90,
                        help='EMA momentum for track appearance features')
    parser.add_argument('--min-track-frames', type=int, default=MIN_TRACK_FRAMES,
                        help='Minimum observed frames included in any statistics (minimum allowed: 3)')
    parser.add_argument('--sequence-prefix', type=str, default=SEQUENCE_PREFIX,
                        help='Sequence folder prefix under split_tif')
    parser.add_argument('--sequence-start', type=int, default=SEQUENCE_START,
                        help='First sequence index')
    parser.add_argument('--sequence-end', type=int, default=SEQUENCE_END,
                        help='Last sequence index')
    parser.add_argument('--area-change-threshold', type=float,
                        default=DEFAULT_AREA_CHANGE_THRESHOLD,
                        help='Symmetric relative-change threshold for Stable area tracks')
    parser.add_argument('--area-endpoint-window', type=int,
                        default=DEFAULT_AREA_ENDPOINT_WINDOW,
                        help='Maximum valid frames averaged at each end for net area change')
    parser.add_argument('--circularity-low-thr', type=float,
                        default=DEFAULT_CIRCULARITY_LOW_THRESHOLD,
                        help='Low/Medium mean circularity threshold')
    parser.add_argument('--circularity-high-thr', type=float,
                        default=DEFAULT_CIRCULARITY_HIGH_THRESHOLD,
                        help='Medium/High mean circularity threshold')
    parser.add_argument('--edge-margin-um', type=float, default=DEFAULT_EDGE_MARGIN_UM,
                        help='Image-edge margin used to mark field-exit censoring')
    parser.add_argument('--end-censor-tolerance-frames', type=int,
                        default=DEFAULT_END_CENSOR_TOLERANCE_FRAMES,
                        help='Last frames treated as right-censored at video end')

    args = parser.parse_args()

    if args.frame_interval <= 0:
        parser.error('--frame-interval must be > 0')
    if args.microns_per_pixel <= 0:
        parser.error('--microns-per-pixel must be > 0')
    if args.contour_opening_um < 0:
        parser.error('--contour-opening-um must be >= 0')
    if args.min_track_frames < 3:
        parser.error('--min-track-frames must be >= 3')
    if not (0 <= args.track_low_thresh <= args.track_high_thresh <= 1):
        parser.error('tracking thresholds must satisfy 0 <= low <= high <= 1')
    if not (0 <= args.new_track_thresh <= 1):
        parser.error('--new-track-thresh must be between 0 and 1')
    if not (0 <= args.appearance_weight <= 1 and 0 <= args.appearance_thresh <= 2):
        parser.error('appearance weight/threshold are outside valid ranges')
    if not (0 <= args.appearance_momentum < 1):
        parser.error('--appearance-momentum must satisfy 0 <= value < 1')
    if args.area_change_threshold < 0 or args.area_endpoint_window < 1:
        parser.error('area-change threshold must be >= 0 and endpoint window >= 1')
    if not (0 <= args.circularity_low_thr < args.circularity_high_thr <= 1):
        parser.error('circularity thresholds must satisfy 0 <= low < high <= 1')
    if args.edge_margin_um < 0 or args.end_censor_tolerance_frames < 0:
        parser.error('censoring margin/tolerance must be >= 0')


    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences_output_dir = output_dir / 'sequences'
    sequences_output_dir.mkdir(exist_ok=True)
    summary_output_dir = output_dir / 'summary'
    summary_output_dir.mkdir(exist_ok=True)

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
    sequence_g_dirs = get_sequence_g_folders(
        args.input,
        args.sequence_prefix,
        args.sequence_start,
        args.sequence_end
    )
    if len(sequence_g_dirs) == 0:
        print(f"No sequence G folders found in {args.input}")
        return
    print(f"\nFound {len(sequence_g_dirs)} sequence G folders")


    frame_interval_sec = args.frame_interval
    print(f"Frame interval: {frame_interval_sec} sec ({frame_interval_sec / 60.0:.2f} min)")
    print(f"Filtering summary tracks with < {args.min_track_frames} observed frames")

    # ============================================================

    # ============================================================
    print(
        "Using tracker tracker "
        f"(Kalman + IoU + appearance, weight={args.appearance_weight:.2f})"
    )

    if args.det_thresh > args.track_low_thresh:
        print(
            f"[Warning] det-thresh ({args.det_thresh:.2f}) > "
            f"track-low-thresh ({args.track_low_thresh:.2f}); detections below "
            "det-thresh cannot be used by second-stage matching."
        )

    all_metric_records = []
    sequence_metadata = []
    total_t0 = _time.time()

    for g_dir in sequence_g_dirs:
        sequence_name = g_dir.parent.name
        _, seq_metrics, _, seq_metadata = process_sequence(
            g_dir,
            sequences_output_dir / sequence_name,
            denoise_net,
            device,
            model,
            args
        )
        all_metric_records.extend(seq_metrics)
        sequence_metadata.append(seq_metadata)

    total_elapsed = _time.time() - total_t0
    print(f"\nAll sequences processed in {total_elapsed:.1f}s")

    # ============================================================

    # ============================================================
    filtered_df = filter_short_tracks(all_metric_records, args.min_track_frames)
    all_metrics_path = summary_output_dir / 'all_bbox_metrics_unfiltered.xlsx'
    pd.DataFrame(all_metric_records).to_excel(all_metrics_path, index=False)
    filtered_metrics_path = summary_output_dir / 'filtered_track_frame_metrics.xlsx'
    filtered_df.to_excel(filtered_metrics_path, index=False)
    print(f"Unfiltered metrics saved to: {all_metrics_path}")
    print(f"Filtered metrics saved to: {filtered_metrics_path}")

    if filtered_df.empty:
        print('No tracks met the minimum-frame criterion; summary plots were skipped.')
        return

    track_summary = build_track_summary(
        filtered_df,
        sequence_metadata,
        args.min_track_frames,
        args.area_endpoint_window,
        args.area_change_threshold,
        args.circularity_low_thr,
        args.circularity_high_thr,
        args.microns_per_pixel,
        args.edge_margin_um,
        args.end_censor_tolerance_frames,
    )
    track_summary_path = summary_output_dir / 'track_level_summary.xlsx'
    track_summary.to_excel(track_summary_path, index=False)

    sequence_ids = [
        str(item['sequence_id']) for item in sequence_metadata
        if int(item['num_frames']) > 0
    ]
    area_proportions = build_category_proportions(
        track_summary, sequence_ids, 'area_change_type',
        ['Expanding', 'Stable', 'Contracting'],
    )
    circularity_proportions = build_category_proportions(
        track_summary, sequence_ids, 'circularity_class',
        ['Low', 'Medium', 'High'],
    )
    speed_proportions = build_category_proportions(
        track_summary, sequence_ids, 'speed_group',
        ['Slow', 'Medium', 'Fast'],
    )
    area_proportion_stats = summarize_category_proportions(area_proportions)
    circularity_proportion_stats = summarize_category_proportions(
        circularity_proportions
    )
    speed_proportion_stats = summarize_category_proportions(speed_proportions)
    area_tests = build_pairwise_wilcoxon_results(
        area_proportions, ['Expanding', 'Stable', 'Contracting']
    )
    circularity_tests = build_pairwise_wilcoxon_results(
        circularity_proportions, ['Low', 'Medium', 'High']
    )
    speed_tests = build_pairwise_wilcoxon_results(
        speed_proportions, ['Slow', 'Medium', 'Fast']
    )
    area_duration, circularity_duration, heatmap_times_duration, row_order_duration = (
        build_aligned_heatmap_matrices(filtered_df, track_summary)
    )
    parameter_rows = [
        ('microns_per_pixel', args.microns_per_pixel, '176.7767 μm / 1024 pixels'),
        ('frame_interval_sec', args.frame_interval, 'fixed for all sequences'),
        ('minimum_track_frames', args.min_track_frames, 'tracks below this are excluded'),
        ('contour_opening_um', args.contour_opening_um, 'elliptical morphological opening'),
        ('area_change_threshold', args.area_change_threshold,
         'Stable when absolute relative net change <= threshold'),
        ('area_endpoint_window', args.area_endpoint_window,
         'maximum valid frames averaged at each endpoint'),
        ('circularity_low_threshold', args.circularity_low_thr, 'Low/Medium boundary'),
        ('circularity_high_threshold', args.circularity_high_thr, 'Medium/High boundary'),
        ('speed_slow_medium_um_h', SPEED_THRESHOLDS[0], '0–2 Slow, 2–5 Medium, >5 Fast'),
        ('speed_medium_fast_um_h', SPEED_THRESHOLDS[1], '0–2 Slow, 2–5 Medium, >5 Fast'),
        ('edge_margin_um', args.edge_margin_um, 'field-exit right-censoring'),
        ('end_censor_tolerance_frames', args.end_censor_tolerance_frames,
         'video-end right-censoring'),
        ('fraction_pairwise_test', 'paired two-sided Wilcoxon signed-rank',
         'each sequence contributes one paired fraction per category; Holm correction'),
    ]
    parameter_df = pd.DataFrame(parameter_rows, columns=['parameter', 'value', 'definition'])
    area_heatmap_duration_source = heatmap_matrix_to_dataframe(
        area_duration, heatmap_times_duration, row_order_duration
    )
    circularity_heatmap_duration_source = heatmap_matrix_to_dataframe(
        circularity_duration, heatmap_times_duration, row_order_duration
    )
    source_path = summary_output_dir / 'analysis_source_data.xlsx'
    with pd.ExcelWriter(source_path) as writer:
        parameter_df.to_excel(writer, sheet_name='parameters', index=False)
        pd.DataFrame(sequence_metadata).to_excel(
            writer, sheet_name='sequence_metadata', index=False
        )
        filtered_df.to_excel(writer, sheet_name='per_frame_filtered', index=False)
        track_summary.to_excel(writer, sheet_name='track_summary', index=False)
        row_order_duration.to_excel(writer, sheet_name='heatmap_order_duration', index=False)
        area_heatmap_duration_source.to_excel(
            writer, sheet_name='area_heatmap_duration', index=False
        )
        circularity_heatmap_duration_source.to_excel(
            writer, sheet_name='circ_heatmap_duration', index=False
        )
        area_proportions.to_excel(writer, sheet_name='area_change_fraction', index=False)
        area_proportion_stats.to_excel(writer, sheet_name='area_fraction_stats', index=False)
        area_tests.to_excel(writer, sheet_name='area_pairwise_tests', index=False)
        circularity_proportions.to_excel(
            writer, sheet_name='circularity_fraction', index=False
        )
        circularity_proportion_stats.to_excel(
            writer, sheet_name='circularity_stats', index=False
        )
        circularity_tests.to_excel(writer, sheet_name='circ_pairwise_tests', index=False)
        speed_proportions.to_excel(writer, sheet_name='speed_fraction', index=False)
        speed_proportion_stats.to_excel(writer, sheet_name='speed_stats', index=False)
        speed_tests.to_excel(writer, sheet_name='speed_tests', index=False)
    print(f"Track summary saved to: {track_summary_path}")
    print(f"Complete analysis source data saved to: {source_path}")

    # ============================================================

    # ============================================================
    plot_area_circularity_heatmaps(
        area_duration,
        circularity_duration,
        heatmap_times_duration,
        row_order_duration,
        str(summary_output_dir / (
            'a1_heatmaps_sorted_by_duration_blue_green_'
            'bar_area0to6_circ0p5to1_separate_gray_background_xaxis0.pdf'
        )),
        area_vmin=0.0,
        area_vmax=6.0,
        circularity_vmin=0.5,
        circularity_vmax=1.0,
    )
    plot_fraction_violin(
        area_proportions,
        ['Expanding', 'Stable', 'Contracting'],
        f'Area-change composition (Stable: |net change| ≤ {args.area_change_threshold * 100:.0f}%)',
        str(summary_output_dir / 'b_area_change_type_fraction.pdf'),
    )
    plot_fraction_violin(
        circularity_proportions,
        ['Low', 'Medium', 'High'],
        f'Mean-circularity composition ({args.circularity_low_thr:.2f}, {args.circularity_high_thr:.2f})',
        str(summary_output_dir / 'c_circularity_class_fraction.pdf'),
    )
    plot_lifetime_survival(
        track_summary,
        str(summary_output_dir / 'd_track_duration_survival.pdf'),
    )
    plot_fraction_violin(
        speed_proportions,
        ['Slow', 'Medium', 'Fast'],
        'Migrasome speed composition: 0–2 / 2–5 / >5 μm h⁻¹',
        str(summary_output_dir / 'e_speed_fraction_thresholds_2_5.pdf'),
    )


    print("\n" + "=" * 60)
    print("Tracking Statistics")
    print("=" * 60)
    print(f"Total sequences: {len(sequence_g_dirs)}")
    print(f"Total detections: {len(all_metric_records)}")
    print(f"Total raw tracks: {pd.DataFrame(all_metric_records)['global_track_id'].nunique() if all_metric_records else 0}")
    print(f"Tracks after >= {args.min_track_frames} frame filter: {len(track_summary)}")
    print(f"Spatial calibration: {args.microns_per_pixel:.8f} μm/pixel")
    print("Appearance matching: enabled")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
