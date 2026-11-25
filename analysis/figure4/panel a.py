'Denoise, detect and track an image sequence; render annotated frames and box counts.'

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
from typing import List, Tuple

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

# ============================================================

# ============================================================
DENOISE_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'denoise.pth')

DEFAULT_IMG_SIZE = 1120
DEFAULT_SCORE_THR = 0.25
DEFAULT_IOU_THR = 0.5

DETECT_WEIGHTS = str(REPO_ROOT / 'checkpoint' / 'detect.pth')


INPUT_DIR = '/path/of/images/figure4_overlays'
OUTPUT_DIR = '/path/of/results/figure4_overlays'


FRAME_INTERVAL_SEC = 525.79620

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
    'Export per-box intensity over time to an Excel workbook.'
    df = pd.DataFrame(intensity_records)
    df.to_excel(output_path, index=False)
    print(f"Intensity table saved to: {output_path}")


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
                        help='Frame interval in seconds (default: 525.79828)')
    parser.add_argument('--save-images', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--save-mot', action='store_true', default=True,
                        help='Save results in MOT format')
    parser.add_argument('--save-video', action='store_true',
                        help='Save output as video')
    parser.add_argument('--bbox-line-width', type=int, default=1,
                        help='BBox line width for visualization')


    parser.add_argument('--track-high-thresh', type=float, default=0.35,
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
    if args.save_denoised:
        denoised_dir.mkdir(exist_ok=True)

    if args.save_images:
        pdfs_dir = output_dir / 'pdfs'
        pdfs_dir.mkdir(exist_ok=True)

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
    frame_bbox_counts = []
    video_writer = None
    vis_ctx = None  

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
        for track in tracks:
            bbox = track.tlbr
            mot_results.append((
                frame_id, track.track_id,
                bbox[0], bbox[1], bbox[2], bbox[3],
                track.score
            ))
            intensity_records.append({
                'frame': frame_id,
                'track_id': track.track_id,
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3]),
                'score': float(track.score),
                'intensity': extract_bbox_intensity(gray_frame, bbox),
            })

        frame_bbox_counts.append((frame_id, len(tracks)))


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

    total_elapsed = _time.time() - total_t0
    print(f"\nAll {len(image_files)} frames processed in {total_elapsed:.1f}s "
          f"({total_elapsed / len(image_files):.2f}s/frame)")

    # ============================================================

    # ============================================================
    if args.save_mot:
        mot_path = output_dir / 'tracking_results.txt'
        save_mot_results(mot_results, str(mot_path))

    intensity_path = output_dir / 'bbox_intensity_over_time.xlsx'
    save_intensity_xlsx(intensity_records, str(intensity_path))

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
