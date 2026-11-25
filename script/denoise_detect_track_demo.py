"""Run denoise, detect and track on a single image or an ordered frame folder."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys

# Locate shared modules and checkpoints in the repository above script/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='An image or a folder containing ordered image frames.')
    parser.add_argument('--output', type=Path, required=True, help='A new or empty output folder.')
    parser.add_argument('--denoise-weights', type=Path, default=REPO_ROOT / 'checkpoint/denoise.pth')
    parser.add_argument('--detect-weights', type=Path, default=REPO_ROOT / 'checkpoint/detect.pth')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resolution', type=int, default=1120)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--min-track-frames', type=int, default=1)
    parser.add_argument('--no-track', action='store_true', help='Process unrelated images independently.')
    parser.add_argument('--max-frames', type=int, help='Limit the number of frames for a short trial.')
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1.')
    if args.resolution <= 0 or args.resolution % 56:
        parser.error('--resolution must be a positive multiple of 56.')
    if args.min_track_frames < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('Frame counts must be positive.')
    return args


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', path.name)]


def find_images(source):
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if not source.is_dir():
        raise ValueError(f'Input does not exist or is not a supported image: {source}')
    images = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)
    if not images:
        raise ValueError(f'No supported images found in {source}')
    return images


def check_weights(path):
    if not path.is_file():
        raise ValueError(f'Missing checkpoint: {path}. Supply the real .pth file or set its --*-weights option.')
    with path.open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/'):
            raise ValueError(f'{path} is a Git LFS pointer. Run git lfs pull in the repository to download the checkpoint.')


def read_image(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError(f'{path} contains multiple pages. Export them as ordered 2D frame files first.')
        raw = np.asarray(image).copy()
    if raw.ndim == 2:
        raw_gray = raw.astype(np.float64)
        display = raw[..., None]
        display = np.repeat(display, 3, axis=2)
    elif raw.ndim == 3 and raw.shape[2] in (3, 4):
        display = raw[..., :3]
        raw_gray = display.astype(np.float64).mean(axis=2)
    else:
        raise ValueError(f'Unsupported image dimensions {raw.shape} in {path}; provide 2D grayscale or RGB frames.')
    if display.dtype != np.uint8:
        display = display.astype(np.float32)
        valid = display[np.isfinite(display)]
        low, high = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 0.0)
        display = np.nan_to_num((display - low) / (high - low) if high > low else np.zeros_like(display))
        display = np.clip(display * 255.0, 0, 255).astype(np.uint8)
    return raw_gray, np.ascontiguousarray(display)


def measure_box(raw_gray, denoised_gray, bbox):
    import cv2
    import numpy as np

    height, width = raw_gray.shape
    x1, y1, x2, y2 = bbox
    left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    raw_patch = raw_gray[top:bottom, left:right]
    finite = raw_patch[np.isfinite(raw_patch)]
    intensity = float(finite.mean()) if finite.size else float('nan')
    patch = denoised_gray[top:bottom, left:right]
    if not patch.size or patch.max() == patch.min():
        return intensity, float('nan'), float('nan')
    _, mask = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    candidates = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= 3]
    if not candidates:
        return intensity, float('nan'), float('nan')
    centre = (patch.shape[1] / 2.0, patch.shape[0] / 2.0)
    central_label = int(labels[min(patch.shape[0] - 1, int(centre[1])), min(patch.shape[1] - 1, int(centre[0]))])
    selected = central_label if central_label in candidates else min(candidates, key=lambda label: float(np.sum((centres[label] - centre) ** 2)))
    area = int(stats[selected, cv2.CC_STAT_AREA])
    return intensity, area, float(2.0 * np.sqrt(area / np.pi))


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    images = find_images(args.input.expanduser().resolve())
    if args.max_frames:
        images = images[:args.max_frames]
    check_weights(args.denoise_weights)
    check_weights(args.detect_weights)
    if args.output.exists() and not args.output.is_dir():
        raise ValueError(f'Output is not a folder: {args.output}')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Output folder is not empty: {args.output}. Choose a new folder.')

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from denoise import load_denoiser, denoise_rgb
    from detect import DetectionModel
    from tracker import Tracker, TrackingConfig

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Install a compatible NVIDIA driver and CUDA-enabled PyTorch, or choose --device cpu.')
    device = torch.device(args.device)
    print('Loading denoise and detect models...')
    denoiser = load_denoiser(args.denoise_weights, device)
    detector = DetectionModel(pretrain_weights=str(args.detect_weights.resolve()), resolution=args.resolution, device=args.device)
    tracking = not args.no_track and len(images) > 1
    tracker = Tracker(TrackingConfig(track_high_thresh=0.45, track_low_thresh=0.25, new_track_thresh=0.55, track_buffer=60, match_thresh=0.75, appearance_weight=0.25)) if tracking else None
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ['denoised', 'overlays']:
        (args.output / folder).mkdir()
    records, track_records, frames = [], [], []
    previous_shape = None
    for frame_index, path in enumerate(images, start=1):
        raw_gray, rgb = read_image(path)
        if tracking and previous_shape is not None and rgb.shape[:2] != previous_shape:
            raise ValueError('All frames in a tracked sequence must have the same image dimensions. Use --no-track for independent images.')
        previous_shape = rgb.shape[:2]
        denoised = denoise_rgb(denoiser, rgb, device)
        detections = detector.predict(Image.fromarray(denoised), threshold=args.threshold)
        boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(detections.confidence, dtype=np.float32).reshape(-1)
        det_array = np.column_stack([boxes, scores]).astype(np.float32)
        bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        tracks = tracker.update(det_array, bgr) if tracker else []
        gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        for object_index, (bbox, score) in enumerate(zip(boxes, scores), start=1):
            intensity, area, diameter = measure_box(raw_gray, gray, bbox)
            records.append(dict(frame=frame_index, image=path.name, object_id=object_index, x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]), score=float(score), raw_mean_intensity=intensity, area_px=area, diameter_px=diameter))
        for track in tracks:
            bbox = track.tlbr
            intensity, area, diameter = measure_box(raw_gray, gray, bbox)
            track_records.append(dict(frame=frame_index, image=path.name, track_id=int(track.track_id), x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]), score=float(track.score), raw_mean_intensity=intensity, area_px=area, diameter_px=diameter))
        save_name = f'{frame_index:06d}_{path.stem}'
        Image.fromarray(denoised).save(args.output / 'denoised' / f'{save_name}.tif')
        frames.append(dict(frame=frame_index, image=path.name, detections=len(boxes), active_tracks=len(tracks), overlay_name=save_name))
        print(f'Frame {frame_index}/{len(images)}: {path.name}, detections={len(boxes)}, active tracks={len(tracks)}')

    from collections import Counter
    counts = Counter(row['track_id'] for row in track_records)
    track_records = [row for row in track_records if counts[row['track_id']] >= args.min_track_frames]
    from collections import defaultdict
    records_by_frame = defaultdict(list)
    for row in (track_records if tracking else records):
        records_by_frame[row['frame']].append(row)
    for frame in frames:
        rows = records_by_frame[frame['frame']]
        if tracking:
            frame['active_tracks'] = len(rows)
        with Image.open(args.output / 'denoised' / f"{frame['overlay_name']}.tif") as saved_image:
            overlay = cv2.cvtColor(np.asarray(saved_image), cv2.COLOR_RGB2BGR)
        for row in rows:
            p1, p2 = (int(row['x1']), int(row['y1'])), (int(row['x2']), int(row['y2']))
            cv2.rectangle(overlay, p1, p2, (0, 255, 0), 1)
            label = str(row['track_id'] if tracking else row['object_id'])
            cv2.putText(overlay, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(args.output / 'overlays' / f"{frame.pop('overlay_name')}.png")
    measurement_fields = ['frame', 'image', 'object_id', 'x1', 'y1', 'x2', 'y2', 'score', 'raw_mean_intensity', 'area_px', 'diameter_px']
    write_csv(args.output / 'detections.csv', records, measurement_fields)
    write_csv(args.output / 'tracks.csv', track_records, [field.replace('object_id', 'track_id') for field in measurement_fields])
    write_csv(args.output / 'frame_summary.csv', frames, ['frame', 'image', 'detections', 'active_tracks'])
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config.update(tracking_enabled=tracking, processed_images=[str(p) for p in images], detection_rows=len(records), unique_tracks=len({r['track_id'] for r in track_records}))
    (args.output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Results saved to {args.output.resolve()}')


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
