'Denoise, detect and track an image sequence; export overlays, masks and diameter/count time courses.'

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

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 11})


from detect import DetectionModel
from tracker import Tracker, TrackingConfig, STrack


from denoise import UNet

# ============================================================

# ============================================================
DENOISE_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'denoise.pth')

DEFAULT_IMG_SIZE = 1120
DEFAULT_SCORE_THR = 0.25
DEFAULT_IOU_THR = 0.5
DEFAULT_MIN_TRACK_FRAMES = 10

DETECT_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'detect.pth')


INPUT_DIR = '/path/of/images/figure2_sequence'
OUTPUT_DIR = '/path/of/results/tracking/figure2'

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


def extract_bbox_diameter_metrics(gray_image: np.ndarray, bbox: np.ndarray) -> Tuple[dict, np.ndarray, Tuple[int, int, int, int]]:
    'Threshold inside a box and measure mask area and equivalent diameter.'
    empty_metrics = {
        'area': 0.0,
        'diameter': float('nan'),
        'threshold': float('nan'),
    }
    empty_mask = np.zeros((0, 0), dtype=bool)
    x1, y1, x2, y2 = _clip_bbox_to_image(bbox, (*gray_image.shape, 1))
    if x2 <= x1 or y2 <= y1:
        return empty_metrics, empty_mask, (x1, y1, x2, y2)

    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return empty_metrics, empty_mask, (x1, y1, x2, y2)

    roi_u8 = roi
    if roi_u8.dtype != np.uint8:
        roi_u8 = cv2.normalize(roi_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if int(roi_u8.max()) == int(roi_u8.min()):
        mask = roi_u8 > roi_u8.min()
        threshold = float(roi_u8.min())
    else:
        threshold, mask_u8 = cv2.threshold(
            roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if mask_u8.mean() > 255 * 0.65:
            mask_u8 = cv2.bitwise_not(mask_u8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

        num_labels, labels, stats_arr, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats_arr[1:, cv2.CC_STAT_AREA]))
            mask = labels == largest_label
        else:
            mask = mask_u8 > 0

    area = float(np.count_nonzero(mask))
    if area <= 0:
        metrics = empty_metrics.copy()
        metrics['threshold'] = float(threshold)
        return metrics, mask, (x1, y1, x2, y2)

    return {
        'area': area,
        'diameter': float(np.sqrt(4.0 * area / np.pi)),
        'threshold': float(threshold),
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
             line_width: int = 1) -> np.ndarray:
        'Render one frame and return a BGR image.'
        records = []
        for track in tracks:
            bbox = track.tlbr
            records.append({
                'track_id': track.track_id,
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3]),
            })
        return self.draw_records(image_bgr, records, line_width)

    def draw_records(self, image_bgr: np.ndarray, records: List[dict],
                     line_width: int = 1) -> np.ndarray:
        'Render box records for one frame and return a BGR image.'
        ax = self.ax


        ax.cla()
        ax.set_xlim(0, self.w)
        ax.set_ylim(self.h, 0)
        ax.axis('off')


        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)


        for record in sorted(records, key=lambda r: r['track_id']):
            track_id = int(record['track_id'])
            bbox = (record['x1'], record['y1'], record['x2'], record['y2'])
            color_rgb = tuple(c / 255.0 for c in PALETTE[track_id % len(PALETTE)])

            x1, y1, x2, y2 = _clip_bbox_to_image(bbox, image_bgr.shape)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=line_width, edgecolor=color_rgb, facecolor='none'
            )
            ax.add_patch(rect)


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


def save_metrics_xlsx(metric_records: List[dict], output_path: str):
    'Export per-box measurements over time to an Excel workbook.'
    df = pd.DataFrame(metric_records)
    df.to_excel(output_path, index=False)
    print(f"Metrics table saved to: {output_path}")


def filter_records_by_min_track_frames(records: List[dict], min_track_frames: int) -> Tuple[List[dict], set]:
    'Remove tracks observed in fewer than min_track_frames frames.'
    if not records:
        return [], set()

    df = pd.DataFrame(records)
    track_frame_counts = df.groupby('track_id')['frame'].nunique()
    keep_track_ids = set(track_frame_counts[track_frame_counts >= min_track_frames].index.tolist())
    filtered_records = [r for r in records if r['track_id'] in keep_track_ids]
    return filtered_records, keep_track_ids


def records_to_mot_results(records: List[dict]) -> List[Tuple]:
    'Convert box records to MOT output rows.'
    mot_results = []
    for record in sorted(records, key=lambda r: (r['frame'], r['track_id'])):
        mot_results.append((
            record['frame'],
            record['track_id'],
            record['x1'],
            record['y1'],
            record['x2'],
            record['y2'],
            record['score'],
        ))
    return mot_results


def index_records_by_frame(records: List[dict]) -> Dict[int, List[dict]]:
    records_by_frame: Dict[int, List[dict]] = {}
    for record in records:
        records_by_frame.setdefault(record['frame'], []).append(record)
    return records_by_frame


def frame_count_series(records: List[dict], total_frames: int) -> pd.Series:
    frames = pd.Index(range(1, total_frames + 1), name='frame')
    if not records:
        return pd.Series(0, index=frames, dtype=int)
    df = pd.DataFrame(records)
    return df.groupby('frame').size().reindex(frames, fill_value=0)


def diameter_summary_by_frame(records: List[dict], total_frames: int) -> pd.DataFrame:
    frames = pd.Index(range(1, total_frames + 1), name='frame')
    if not records:
        return pd.DataFrame({
            'mean': pd.Series(np.nan, index=frames, dtype=float),
            'std': pd.Series(np.nan, index=frames, dtype=float),
            'count': pd.Series(0, index=frames, dtype=int),
        })

    df = pd.DataFrame(records)
    summary = df.groupby('frame')['diameter'].agg(['mean', 'std', 'count']).reindex(frames)
    summary['count'] = summary['count'].fillna(0).astype(int)
    summary['std'] = summary['std'].fillna(0.0)
    return summary


def plot_diameter_error_band(ax, frames: np.ndarray, summary: pd.DataFrame):
    mean_values = summary['mean'].to_numpy(dtype=float)
    std_values = summary['std'].to_numpy(dtype=float)
    ax.fill_between(
        frames, mean_values - std_values, mean_values + std_values,
        color='0.86', alpha=0.65, linewidth=0, zorder=1
    )
    ax.plot(frames, mean_values, color='black', linewidth=1.5, zorder=2)


def _style_time_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.8, pad=2)


def plot_mean_diameter_over_time(records: List[dict], total_frames: int, output_path: str):
    'Plot the per-frame mean diameter with a light-gray within-frame standard-deviation band.'
    if total_frames <= 0:
        return
    frames = np.arange(1, total_frames + 1)
    diameter_summary = diameter_summary_by_frame(records, total_frames)

    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=150)
    plot_diameter_error_band(ax, frames, diameter_summary)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean diameter')
    _style_time_axis(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Mean diameter plot saved to: {output_path}")


def plot_bbox_count_over_time(records: List[dict], total_frames: int, output_path: str):
    'Plot the filtered box count over frames.'
    if total_frames <= 0:
        return
    frames = np.arange(1, total_frames + 1)
    bbox_counts = frame_count_series(records, total_frames)

    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=150)
    ax.plot(frames, bbox_counts.values, color='black', linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Number')
    _style_time_axis(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"BBox count plot saved to: {output_path}")


def plot_diameter_and_count_over_time(records: List[dict], total_frames: int, output_path: str):
    'Plot mean diameter and box count in two vertically arranged panels.'
    if total_frames <= 0:
        return
    frames = np.arange(1, total_frames + 1)
    diameter_summary = diameter_summary_by_frame(records, total_frames)
    bbox_counts = frame_count_series(records, total_frames)

    fig, axes = plt.subplots(
        2, 1, figsize=(3.2, 3.5), dpi=150, sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.28}
    )

    plot_diameter_error_band(axes[0], frames, diameter_summary)
    axes[0].set_ylabel('Mean diameter')
    axes[0].set_xlabel('Frame')

    axes[1].plot(frames, bbox_counts.values, color='black', linewidth=1.5)
    axes[1].set_ylabel('Number')
    axes[1].set_xlabel('Frame')

    for ax in axes:
        _style_time_axis(ax)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, format='pdf')
    plt.close(fig)
    print(f"Diameter/count summary plot saved to: {output_path}")


def save_filtered_visual_outputs(
        frame_infos: List[dict],
        records_by_frame: Dict[int, List[dict]],
        pdfs_dir: Optional[Path],
        bbox_line_width: int,
        save_pdfs: bool,
        save_video: bool,
        video_path: Path):
    'Save per-frame overlay PDFs from filtered box records and optionally assemble a video.'
    if not frame_infos:
        return

    vis_ctx = None
    video_writer = None

    for frame_info in tqdm(frame_infos, desc="Saving filtered overlays"):
        frame_id = frame_info['frame']
        image_bgr = cv2.imread(str(frame_info['image_path']), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[Warning] Failed to read image for overlay: {frame_info['image_path']}")
            continue

        if vis_ctx is None:
            h, w = image_bgr.shape[:2]
            vis_ctx = MatplotlibVis(h, w)

        frame_records = records_by_frame.get(frame_id, [])
        vis_image = vis_ctx.draw_records(
            image_bgr, frame_records, line_width=bbox_line_width
        )

        if save_pdfs and pdfs_dir is not None:
            pdf_path = pdfs_dir / f"{Path(frame_info['fname']).stem}.pdf"
            vis_ctx.save_pdf(str(pdf_path))

        if save_video:
            if video_writer is None:
                h, w = vis_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
            video_writer.write(vis_image)

    if vis_ctx is not None:
        vis_ctx.close()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_path}")


def save_frame_masks(
        frame_infos: List[dict],
        records_by_frame: Dict[int, List[dict]],
        masks_dir: Path):
    'Save full-frame binary-mask PNGs for the retained boxes.'
    masks_dir.mkdir(exist_ok=True)

    for frame_info in tqdm(frame_infos, desc="Saving masks"):
        image_bgr = cv2.imread(str(frame_info['image_path']), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[Warning] Failed to read image for mask: {frame_info['image_path']}")
            continue

        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        frame_mask = np.zeros(gray_image.shape, dtype=np.uint8)
        for record in records_by_frame.get(frame_info['frame'], []):
            bbox = (record['x1'], record['y1'], record['x2'], record['y2'])
            _, mask, (x1, y1, x2, y2) = extract_bbox_diameter_metrics(gray_image, bbox)
            if mask.size == 0 or x2 <= x1 or y2 <= y1:
                continue
            mask_roi = frame_mask[y1:y2, x1:x2]
            mask_roi[mask] = 255

        mask_path = masks_dir / f"{Path(frame_info['fname']).stem}.png"
        cv2.imwrite(str(mask_path), frame_mask)


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
    parser.add_argument('--save-images', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--save-mot', action='store_true', default=True,
                        help='Save results in MOT format')
    parser.add_argument('--save-video', action='store_true',
                        help='Save output as video')
    parser.add_argument('--bbox-line-width', type=int, default=1,
                        help='BBox line width for visualization')
    parser.add_argument('--min-track-frames', type=int, default=DEFAULT_MIN_TRACK_FRAMES,
                        help='Filter out tracks appearing in fewer than this many frames')


    parser.add_argument('--track-high-thresh', type=float, default=0.45,
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

    args = parser.parse_args()


    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    denoised_dir = output_dir / 'denoised'
    needs_saved_frames = args.save_denoised or args.save_images or args.save_video
    if needs_saved_frames:
        denoised_dir.mkdir(exist_ok=True)

    pdfs_dir = None
    if args.save_images:
        pdfs_dir = output_dir / 'pdfs'
        pdfs_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    masks_dir = output_dir / 'masks'

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
    metric_records = []
    frame_infos = []

    print("\n" + "=" * 60)
    print("Step 3: Processing images (denoise → detect → track)")
    print("=" * 60)

    total_t0 = _time.time()

    for frame_id, img_path in enumerate(tqdm(image_files, desc="Processing"), start=1):
        frame_t0 = _time.time()
        fname = Path(img_path).name


        t0 = _time.time()
        pil_image, cv_image_raw = load_frame_for_inference_and_vis(img_path)
        t_load = _time.time() - t0


        t0 = _time.time()
        img_rgb_np = np.array(pil_image)
        denoised_rgb = denoise_image(denoise_net, img_rgb_np, device)
        t_denoise = _time.time() - t0

        denoised_save_path = denoised_dir / fname
        if needs_saved_frames:
            cv2.imwrite(str(denoised_save_path), cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR))
        frame_infos.append({
            'frame': frame_id,
            'fname': fname,
            'image_path': denoised_save_path,
        })


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
        for track in tracks:
            bbox = track.tlbr
            diameter_metrics, _, _ = extract_bbox_diameter_metrics(gray_frame, bbox)
            metric_records.append({
                'frame': frame_id,
                'track_id': track.track_id,
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3]),
                'score': float(track.score),
                'area': diameter_metrics['area'],
                'diameter': diameter_metrics['diameter'],
                'threshold': diameter_metrics['threshold'],
            })

        t_vis = 0.0

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

    total_elapsed = _time.time() - total_t0
    print(f"\nAll {len(image_files)} frames processed in {total_elapsed:.1f}s "
          f"({total_elapsed / len(image_files):.2f}s/frame)")

    # ============================================================

    # ============================================================
    filtered_metric_records, keep_track_ids = filter_records_by_min_track_frames(
        metric_records, args.min_track_frames
    )
    filtered_mot_results = records_to_mot_results(filtered_metric_records)
    filtered_records_by_frame = index_records_by_frame(filtered_metric_records)
    all_track_ids = set(r['track_id'] for r in metric_records)
    removed_track_ids = all_track_ids - keep_track_ids

    print(
        f"\nFiltered tracks with total duration < {args.min_track_frames} frames: "
        f"removed {len(removed_track_ids)} / {len(all_track_ids)} tracks"
    )

    if args.save_images or args.save_video:
        save_filtered_visual_outputs(
            frame_infos=frame_infos,
            records_by_frame=filtered_records_by_frame,
            pdfs_dir=pdfs_dir,
            bbox_line_width=args.bbox_line_width,
            save_pdfs=args.save_images,
            save_video=args.save_video,
            video_path=output_dir / 'tracking_output.mp4',
        )

    save_frame_masks(
        frame_infos=frame_infos,
        records_by_frame=filtered_records_by_frame,
        masks_dir=masks_dir,
    )

    if args.save_mot:
        mot_path = output_dir / 'tracking_results.txt'
        save_mot_results(filtered_mot_results, str(mot_path))

    metrics_path = output_dir / 'bbox_diameter_over_time.xlsx'
    save_metrics_xlsx(filtered_metric_records, str(metrics_path))

    plot_mean_diameter_over_time(
        filtered_metric_records,
        len(image_files),
        str(plots_dir / 'mean_diameter_over_time.pdf')
    )
    plot_bbox_count_over_time(
        filtered_metric_records,
        len(image_files),
        str(plots_dir / 'bbox_count_over_time.pdf')
    )
    plot_diameter_and_count_over_time(
        filtered_metric_records,
        len(image_files),
        str(plots_dir / 'diameter_and_bbox_count_over_time.pdf')
    )


    print("\n" + "=" * 60)
    print("Tracking Statistics")
    print("=" * 60)
    print(f"Total frames: {len(image_files)}")
    print(f"Total unique tracks before filtering: {len(all_track_ids)}")
    print(f"Total unique tracks after filtering: {len(keep_track_ids)}")
    print(f"Total detections after filtering: {len(filtered_mot_results)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
