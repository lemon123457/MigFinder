from __future__ import annotations

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
import os
import random
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
from PIL import Image, ImageOps, ImageSequence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.size": 11, "font.family": "Arial"})


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
DATASET_ORDER = [
    "L929",
    "NRK",
    "4T1",
    "MGC-803",
    "Neutrophil (mouse)",
    "Neutrophil (human)",
    "Mouse liver",
    "Plasma",
]
GRADIENT_COLORS = ["#6A1B9A", "#1F77B4", "#2CA02C"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    roots: tuple[Path, ...]
    mode: str = "recursive"
    subfolders: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    path: Path
    frame_index: int | None = None


DATASETS = [
    DatasetSpec(
        "L929",
        (
            Path('/path/of/images/L929'),
            Path('/path/of/images/sequences'),
        ),
        mode="l929_combined",
    ),
    DatasetSpec("NRK", (Path('/path/of/images/NRK/video/NRK-migrasome'),)),
    DatasetSpec("4T1", (Path('/path/of/images/4T1/TIF all/TIF all'),)),
    DatasetSpec("MGC-803", (Path('/path/of/images/High-content/IMAGES'),)),
    DatasetSpec("Neutrophil (mouse)", (Path('/path/of/images/Neutrophils'),)),
    DatasetSpec(
        "Neutrophil (human)",
        (
            Path('/path/of/images/human_neutrophils/Control'),
            Path('/path/of/images/human_neutrophils/LPS'),
        ),
        mode="selected_subfolders",
        subfolders=("1", "2", "3"),
    ),
    DatasetSpec(
        "Mouse liver",
        (Path('/path/of/images/liver'),),
        mode="selected_subfolders",
        subfolders=(
            'control/1-rgb',
            'control/2-rgb',
            'control/3-rgb',
            'ecoli/1-rgb',
            'ecoli/2-rgb',
            'ecoli/3-rgb',
        ),
    ),
    DatasetSpec(
        "Plasma",
        (Path('/path/of/images/plasma'),),
        mode="selected_subfolders",
        subfolders=("6","7","8","9"),
    ),
]


def iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def iter_spec_files(spec: DatasetSpec) -> Iterable[Path]:
    yielded: set[Path] = set()
    for root in spec.roots:
        if not root.exists():
            print(f"Warning: missing path: {root}")
            continue

        if spec.mode == "selected_subfolders":
            paths = (path for name in spec.subfolders for path in iter_image_files(root / name))
        elif spec.mode == "l929_combined" and root.name == "split_tif":
            paths = (
                path
                for video_dir in sorted(path for path in root.iterdir() if path.is_dir())
                for path in iter_image_files(video_dir / "G")
            )
        else:
            paths = iter_image_files(root)

        for path in paths:
            resolved = path.resolve()
            if resolved not in yielded:
                yielded.add(resolved)
                yield path


def gif_frame_count(path: Path) -> int:
    try:
        with Image.open(path) as image:
            return sum(1 for _ in ImageSequence.Iterator(image))
    except Exception as exc:
        print(f"Warning: failed to inspect GIF {path}: {exc}")
        return 1


def add_reservoir(
    reservoir: list[ImageRecord],
    record: ImageRecord,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        reservoir.append(record)
        return
    if len(reservoir) < limit:
        reservoir.append(record)
        return
    j = rng.randrange(seen)
    if j < limit:
        reservoir[j] = record


def scan_datasets(sample_per_dataset: int, seed: int) -> tuple[dict[str, int], list[ImageRecord]]:
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    plot_records: list[ImageRecord] = []

    for spec in DATASETS:
        total = 0
        reservoir: list[ImageRecord] = []
        for path in iter_spec_files(spec):
            if path.suffix.lower() == ".gif":
                for frame_index in range(gif_frame_count(path)):
                    total += 1
                    add_reservoir(reservoir, ImageRecord(spec.name, path, frame_index), total, sample_per_dataset, rng)
            else:
                total += 1
                add_reservoir(reservoir, ImageRecord(spec.name, path), total, sample_per_dataset, rng)
        counts[spec.name] = total
        plot_records.extend(reservoir)
        print(f"{spec.name}: {total} images/frames, {len(reservoir)} used for t-SNE")
    return counts, plot_records


def load_record_image(record: ImageRecord, size: int | None = None) -> Image.Image:
    with Image.open(record.path) as image:
        if record.frame_index is not None:
            image.seek(record.frame_index)
        image = ImageOps.autocontrast(image.convert("RGB"))
        if size is None:
            return image.copy()
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        return canvas


def image_to_feature(record: ImageRecord, size: int = 40) -> np.ndarray | None:
    try:
        image = load_record_image(record, size=size).convert("L")
    except Exception as exc:
        print(f"Warning: failed to read {record.path}: {exc}")
        return None
    arr = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    arr -= arr.mean()
    std = arr.std()
    if std > 1e-6:
        arr /= std
    return arr


def format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def gradient_color_map(order: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = LinearSegmentedColormap.from_list("dataset_gradient", GRADIENT_COLORS)
    n = len(order)
    return {name: cmap(0.0 if n == 1 else i / (n - 1)) for i, name in enumerate(order)}


def scheme1_order(_: dict[str, int]) -> list[str]:
    return DATASET_ORDER.copy()


def scheme2_order(counts: dict[str, int]) -> list[str]:
    return sorted(DATASET_ORDER, key=lambda name: (-counts.get(name, 0), DATASET_ORDER.index(name)))


def plot_count_bars(
    counts: dict[str, int],
    output_pdf: Path,
    order: list[str],
    colors: dict[str, tuple[float, float, float, float]],
    title: str,
) -> None:
    values = np.array([counts.get(name, 0) for name in order], dtype=float)
    plot_values = np.maximum(values, 1)
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(5.1, 4.5), dpi=300)
    ax.barh(y, plot_values, color=[colors[name] for name in order], height=0.68)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Number of images / frames")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#CFCFCF", lw=0.6, alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for yi, value, plot_value in zip(y, values, plot_values):
        ax.text(plot_value * 1.1, yi, format_count(int(value)), va="center", ha="left", fontsize=8.5)

    xmax = max(plot_values.max() * 2.2, 10) if len(plot_values) else 10
    ax.set_xlim(0.8, xmax)
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def compute_tsne_embedding(features: np.ndarray, seed: int) -> np.ndarray:
    if len(features) < 3:
        return np.zeros((len(features), 2), dtype=float)

    from sklearn.manifold import TSNE

    effective_perplexity = min(50.0, max(1.0, float(len(features) - 1)))
    print(f"Running t-SNE: perplexity={effective_perplexity:g}, learning_rate=50")
    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=50.0,
        init="random",
        random_state=seed,
        max_iter=1000,
        method="barnes_hut",
        angle=0.7,
        n_jobs=-1,
    )
    return reducer.fit_transform(features)


def normalize_embedding(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    return (points - mins) / span


def grid_cell_axes(fig: plt.Figure, row: int, col: int, margin: float = 0.015, gap: float = 0.002):
    cell = (1.0 - 2 * margin) / 8
    x = margin + col * cell + gap / 2
    y = margin + (7 - row) * cell + gap / 2
    return fig.add_axes([x, y, cell - gap, cell - gap])


def center_grid_axes(fig: plt.Figure, margin: float = 0.015, gap: float = 0.002):
    cell = (1.0 - 2 * margin) / 8
    x = margin + 2 * cell + gap / 2
    y = margin + 2 * cell + gap / 2
    return fig.add_axes([x, y, 4 * cell - gap, 4 * cell - gap])


def outer_grid_cells() -> list[tuple[int, int]]:
    return [(row, col) for row in range(8) for col in range(8) if not (2 <= row <= 5 and 2 <= col <= 5)]


def outer_cell_anchor(row: int, col: int) -> np.ndarray:
    x = np.clip((col + 0.5 - 2.0) / 4.0, 0.0, 1.0)
    y = np.clip((7.0 - row + 0.5 - 2.0) / 4.0, 0.0, 1.0)
    return np.array([x, y], dtype=float)


def select_spatial_thumbnail_records(
    records: list[ImageRecord],
    points: np.ndarray,
    max_tiles: int,
    seed: int,
) -> list[tuple[ImageRecord, int, int]]:
    rng = random.Random(seed)
    cells = outer_grid_cells()[:max_tiles]
    rng.shuffle(cells)

    dataset_names = [name for name in DATASET_ORDER if any(record.dataset == name for record in records)]
    quota = max(1, int(np.ceil(max_tiles / max(len(dataset_names), 1))))
    dataset_usage = {name: 0 for name in dataset_names}
    unused = set(range(len(records)))
    placements: list[tuple[ImageRecord, int, int]] = []

    for row, col in cells:
        anchor = outer_cell_anchor(row, col)
        candidates = [
            idx
            for idx in unused
            if dataset_usage.get(records[idx].dataset, 0) < quota
        ]
        if not candidates:
            candidates = list(unused)
        if not candidates:
            break

        best_idx = min(candidates, key=lambda idx: float(np.sum((points[idx] - anchor) ** 2)))
        unused.remove(best_idx)
        dataset_usage[records[best_idx].dataset] = dataset_usage.get(records[best_idx].dataset, 0) + 1
        placements.append((records[best_idx], row, col))

    return placements


def plot_tsne_montage(
    records: list[ImageRecord],
    output_pdf: Path,
    seed: int,
    colors: dict[str, tuple[float, float, float, float]],
    points: np.ndarray,
    valid_records: list[ImageRecord],
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = center_grid_axes(fig)
    ax.set_facecolor("white")

    for dataset_name in DATASET_ORDER:
        mask = np.array([record.dataset == dataset_name for record in valid_records])
        if mask.any():
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=6,
                c=[colors[dataset_name]],
                alpha=0.88,
                linewidths=0,
                zorder=3,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_title(title, fontsize=9, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.8)

    placements = select_spatial_thumbnail_records(valid_records, points, 48, seed + 17)
    record_to_index = {record: i for i, record in enumerate(valid_records)}

    for record, row, col in placements:
        tile_ax = grid_cell_axes(fig, row, col)
        tile_ax.imshow(load_record_image(record, size=160))
        tile_ax.set_xticks([])
        tile_ax.set_yticks([])
        for spine in tile_ax.spines.values():
            spine.set_color(colors[record.dataset])
            spine.set_linewidth(0.75)

        idx = record_to_index[record]
        connection = ConnectionPatch(
            xyA=(0.5, 0.5),
            coordsA=tile_ax.transAxes,
            xyB=(points[idx, 0], points[idx, 1]),
            coordsB=ax.transData,
            axesA=tile_ax,
            axesB=ax,
            color=colors[record.dataset],
            alpha=0.42,
            lw=0.55,
            zorder=2,
        )
        fig.add_artist(connection)

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def prepare_tsne(records: list[ImageRecord], seed: int) -> tuple[np.ndarray, list[ImageRecord]]:
    features = []
    valid_records = []
    for record in tqdm(records, desc="Extracting image features", unit="image"):
        feature = image_to_feature(record)
        if feature is not None:
            features.append(feature)
            valid_records.append(record)
    if not features:
        raise RuntimeError("No readable images found for t-SNE.")
    points = normalize_embedding(compute_tsne_embedding(np.vstack(features), seed))
    return points, valid_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset count bars and t-SNE montage figures.")
    parser.add_argument("--outdir", type=Path, default=Path.cwd() / "overall_figures")
    parser.add_argument(
        "--sample-per-dataset",
        type=int,
        default=0,
        help="Maximum records used for t-SNE per dataset. Use 0 to use all records. Count bars always use all records.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
    args.outdir.mkdir(parents=True, exist_ok=True)

    counts, plot_records = scan_datasets(args.sample_per_dataset, args.seed)
    points, valid_records = prepare_tsne(plot_records, args.seed)

    schemes = {
        "scheme1_fixed_order": scheme1_order(counts),
        "scheme2_count_sorted": scheme2_order(counts),
    }
    for scheme_name, order in schemes.items():
        colors = gradient_color_map(order)
        bars_pdf = args.outdir / f"dataset_count_bars_{scheme_name}.pdf"
        tsne_pdf = args.outdir / f"dataset_tsne_montage_{scheme_name}.pdf"
        plot_count_bars(counts, bars_pdf, order, colors, scheme_name.replace("_", " "))
        plot_tsne_montage(
            plot_records,
            tsne_pdf,
            args.seed,
            colors,
            points,
            valid_records,
            f"t-SNE  perplexity=50, lr=50  ({scheme_name.replace('_', ' ')})",
        )
        print(f"Saved: {bars_pdf}")
        print(f"Saved: {tsne_pdf}")


if __name__ == "__main__":
    main()
