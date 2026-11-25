"""Generate Fig. 3 panels d and g from existing annotation benchmark data.

Panel d quantifies human inter-expert heterogeneity. For each expert pair and
dataset, F1 is calculated independently for every image using the project's
one-to-one IoU matching rule, then averaged across images. The ten unique pair
means form both the heatmap entries and the summary violin distribution.

Panel g summarizes the accuracy-efficiency trade-off using mean per-image
annotation/inference time and mean F1 against Consensus GT.
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
import itertools
import os
from typing import Dict, List, Tuple

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'migfinder_matplotlib'))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


from analysis.figure3 import benchmark as core


DEFAULT_BENCHMARK_ROOT = Path(core.OUTPUT_DIR) / "gt_benchmark"
DEFAULT_F1_CSV = DEFAULT_BENCHMARK_ROOT / "denoise" / "gt_experts_vs_migfinder_f1.csv"
DEFAULT_RUNTIME_CSV = DEFAULT_BENCHMARK_ROOT / "denoise" / "model_runtime.csv"
DEFAULT_HUMAN_TIME_CSV = DEFAULT_BENCHMARK_ROOT / "human_annotation_time.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "fig3_panel_cd_human_heterogeneity_tradeoff.pdf"

DATASETS = ("L929", "NRK", "4T1")
PERSONS = tuple(f"person{i}" for i in range(1, 6))
EXPERT_LABELS = tuple(f"E{i}" for i in range(1, 6))

DATASET_STYLE = {
    "L929": {"color": "#A778D2", "marker": "o"},
    "NRK": {"color": "#5B83C5", "marker": "s"},
    "4T1": {"color": "#DF4E9B", "marker": "^"},
}
EXPERT_COLOR = core.GT_PLOT_COLOR
MIGFINDER_COLOR = core.MODEL_PLOT_COLOR
EXPERT_EDGE_COLOR = "#506A86"
MIGFINDER_EDGE_COLOR = "#8B6A16"
VIOLIN_COLOR = "#C8B1E3"


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def sem(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def calculate_pairwise_expert_f1(
        annotation_root: Path,
        expert_root: Path,
        datasets: Tuple[str, ...],
        iou_threshold: float) -> pd.DataFrame:
    """Calculate one macro mean F1 for each unique expert pair and dataset."""
    label_dirs = core.discover_expert_label_dirs(expert_root, datasets)
    records: List[Dict[str, object]] = []

    for dataset in datasets:
        missing_people = [person for person in PERSONS if person not in label_dirs[dataset]]
        if missing_people:
            raise FileNotFoundError(
                f"Missing expert label directories for {dataset}: {missing_people}"
            )

        pair_values: Dict[Tuple[str, str], List[float]] = {
            pair: [] for pair in itertools.combinations(PERSONS, 2)
        }
        image_files = core.get_image_files(str(annotation_root / dataset))
        if not image_files:
            raise FileNotFoundError(f"No annotation images found for {dataset}")

        for image_path in image_files:
            image_rgb = core.load_image_rgb(image_path)
            height, width = image_rgb.shape[:2]
            boxes_by_person = {
                person: core.read_yolo_boxes(
                    label_dirs[dataset][person] / f"{image_path.stem}.txt",
                    width,
                    height,
                )
                for person in PERSONS
            }
            for person_a, person_b in itertools.combinations(PERSONS, 2):
                *_, f1 = core.detection_f1(
                    boxes_by_person[person_a],
                    boxes_by_person[person_b],
                    iou_threshold,
                )
                pair_values[(person_a, person_b)].append(f1)

        for (person_a, person_b), values in pair_values.items():
            clean = np.asarray(values, dtype=float)
            clean = clean[np.isfinite(clean)]
            records.append({
                "dataset": dataset,
                "expert_a": f"E{person_a[-1]}",
                "expert_b": f"E{person_b[-1]}",
                "mean_f1": float(clean.mean()) if len(clean) else float("nan"),
                "sem_f1": sem(clean.tolist()),
                "n_images": int(len(clean)),
                "iou_threshold": float(iou_threshold),
            })

    pairwise_df = pd.DataFrame(records)
    expected_rows = len(datasets) * 10
    if len(pairwise_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} expert-pair records, found {len(pairwise_df)}"
        )
    return pairwise_df


def load_tradeoff_summary(
        f1_csv: Path,
        human_time_csv: Path,
        runtime_csv: Path,
        datasets: Tuple[str, ...],
        variant: str) -> pd.DataFrame:
    """Summarize mean time and GT F1 for every expert and MigFinder."""
    for path in (f1_csv, human_time_csv, runtime_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    f1_df = pd.read_csv(f1_csv)
    human_time_df = pd.read_csv(human_time_csv)
    runtime_df = pd.read_csv(runtime_csv)
    if "variant" in f1_df.columns:
        f1_df = f1_df[f1_df["variant"] == variant]
    if "variant" in runtime_df.columns:
        runtime_df = runtime_df[runtime_df["variant"] == variant]

    records: List[Dict[str, object]] = []
    for dataset in datasets:
        for person in PERSONS:
            f1_values = pd.to_numeric(
                f1_df.loc[
                    (f1_df["dataset"] == dataset)
                    & (f1_df["method"] == "Experts")
                    & (f1_df["source"] == person),
                    "f1",
                ],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            time_values = pd.to_numeric(
                human_time_df.loc[
                    (human_time_df["dataset"] == dataset)
                    & (human_time_df["person"] == person),
                    "seconds",
                ],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            if not len(f1_values) or not len(time_values):
                raise ValueError(f"Missing trade-off data for {dataset}/{person}")
            records.append({
                "dataset": dataset,
                "method": "Experts",
                "source": f"E{person[-1]}",
                "mean_time_s": float(time_values.mean()),
                "sem_time_s": sem(time_values.tolist()),
                "mean_f1": float(f1_values.mean()),
                "sem_f1": sem(f1_values.tolist()),
                "n_time": int(len(time_values)),
                "n_f1": int(len(f1_values)),
            })

        model_f1 = pd.to_numeric(
            f1_df.loc[
                (f1_df["dataset"] == dataset)
                & (f1_df["method"] == "MigFinder"),
                "f1",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        model_time = pd.to_numeric(
            runtime_df.loc[runtime_df["dataset"] == dataset, "seconds"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        if not len(model_f1) or not len(model_time):
            raise ValueError(f"Missing MigFinder trade-off data for {dataset}")
        records.append({
            "dataset": dataset,
            "method": "MigFinder",
            "source": "MigFinder",
            "mean_time_s": float(model_time.mean()),
            "sem_time_s": sem(model_time.tolist()),
            "mean_f1": float(model_f1.mean()),
            "sem_f1": sem(model_f1.tolist()),
            "n_time": int(len(model_time)),
            "n_f1": int(len(model_f1)),
        })

    return pd.DataFrame(records)


def pairwise_matrix(pairwise_df: pd.DataFrame, dataset: str) -> np.ndarray:
    matrix = np.full((len(EXPERT_LABELS), len(EXPERT_LABELS)), np.nan, dtype=float)
    expert_index = {expert: idx for idx, expert in enumerate(EXPERT_LABELS)}
    for row in pairwise_df[pairwise_df["dataset"] == dataset].itertuples(index=False):
        idx_a = expert_index[row.expert_a]
        idx_b = expert_index[row.expert_b]
        matrix[idx_a, idx_b] = row.mean_f1
        matrix[idx_b, idx_a] = row.mean_f1
    return matrix


def plot_heatmap(ax, matrix: np.ndarray, dataset: str, cmap):
    edges = np.arange(len(EXPERT_LABELS) + 1, dtype=float) - 0.5
    image = ax.pcolormesh(
        edges,
        edges,
        np.ma.masked_invalid(matrix),
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        shading="flat",
        edgecolors="white",
        linewidth=0.8,
    )
    ax.set_xlim(-0.5, len(EXPERT_LABELS) - 0.5)
    ax.set_ylim(len(EXPERT_LABELS) - 0.5, -0.5)
    ax.set_title(dataset, pad=4, fontweight="bold")
    ax.set_xticks(np.arange(len(EXPERT_LABELS)))
    ax.set_yticks(np.arange(len(EXPERT_LABELS)))
    ax.set_xticklabels(EXPERT_LABELS)
    ax.set_yticklabels(EXPERT_LABELS)
    ax.tick_params(length=0)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            red, green, blue, _ = cmap(value)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.52 else "#222222"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_pairwise_summary(ax, pairwise_df: pd.DataFrame, datasets: Tuple[str, ...]):
    positions = np.arange(1, len(datasets) + 1, dtype=float)
    groups = [
        pairwise_df.loc[pairwise_df["dataset"] == dataset, "mean_f1"]
        .dropna().to_numpy(dtype=float)
        for dataset in datasets
    ]
    parts = ax.violinplot(
        groups,
        positions=positions,
        widths=0.64,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(VIOLIN_COLOR)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.7)
        body.set_alpha(0.68)

    ax.boxplot(
        groups,
        positions=positions,
        widths=0.14,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#222222", "linewidth": 0.8},
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#333333", "linewidth": 0.7},
        capprops={"color": "#333333", "linewidth": 0.7},
        flierprops={"marker": "", "markersize": 0},
    )
    rng = np.random.default_rng(7)
    for position, values in zip(positions, groups):
        jitter = rng.uniform(-0.075, 0.075, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=10,
            color=VIOLIN_COLOR,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.85,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Pairwise F1-score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.set_box_aspect(4.0 / 3.0)
    ax.set_anchor("C")


def plot_tradeoff(ax, summary_df: pd.DataFrame, datasets: Tuple[str, ...]):
    for dataset in datasets:
        marker = DATASET_STYLE[dataset]["marker"]
        dataset_df = summary_df[summary_df["dataset"] == dataset]
        for row in dataset_df.itertuples(index=False):
            is_model = row.method == "MigFinder"
            facecolor = MIGFINDER_COLOR if is_model else EXPERT_COLOR
            edgecolor = MIGFINDER_EDGE_COLOR if is_model else EXPERT_EDGE_COLOR
            xerr = min(float(row.sem_time_s), float(row.mean_time_s) * 0.80)
            ax.errorbar(
                row.mean_time_s,
                row.mean_f1,
                xerr=xerr,
                yerr=row.sem_f1,
                fmt=marker,
                markersize=7.0 if is_model else 5.7,
                markerfacecolor=facecolor,
                markeredgecolor="none",
                markeredgewidth=0.0,
                ecolor=edgecolor,
                elinewidth=0.65,
                capsize=1.8,
                alpha=0.95 if is_model else 0.82,
                zorder=4 if is_model else 3,
            )
    finite_time = summary_df["mean_time_s"].to_numpy(dtype=float)
    finite_time = finite_time[np.isfinite(finite_time) & (finite_time > 0)]
    ax.set_xscale("log")
    ax.set_xlim(finite_time.min() * 0.55, finite_time.max() * 1.65)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Annotation time per image (s)")
    ax.set_ylabel("F1-score against Consensus GT")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", which="both", top=False, right=False)
    ax.set_box_aspect(1.13)
    ax.set_anchor("C")

    method_handles = [
        Patch(facecolor=EXPERT_COLOR, edgecolor="none", label="Experts"),
        Patch(facecolor=MIGFINDER_COLOR, edgecolor="none", label="MigFinder"),
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=DATASET_STYLE[dataset]["marker"],
            markersize=5.5,
            markerfacecolor="0.65",
            markeredgecolor="none",
            label=dataset,
        )
        for dataset in datasets
    ]
    method_legend = ax.legend(
        handles=method_handles,
        loc="upper left",
        frameon=False,
        borderaxespad=0.3,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=dataset_handles,
        loc="lower right",
        frameon=False,
        handletextpad=0.35,
        borderaxespad=0.3,
    )


def plot_vector_gradient_colorbar(ax, cmap):
    """Draw a smooth cool-warm scale as vector Gouraud-shaded polygons."""
    values = np.linspace(0.0, 1.0, 65)
    x_grid, y_grid = np.meshgrid([0.0, 1.0], values)
    color_values = np.repeat(values[:, None], 2, axis=1)
    ax.pcolormesh(
        x_grid,
        y_grid,
        color_values,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="gouraud",
        rasterized=False,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Mean pairwise F1", labelpad=5)
    ax.tick_params(axis="y", length=2.5, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def plot_combined_figure(
        pairwise_df: pd.DataFrame,
        tradeoff_df: pd.DataFrame,
        datasets: Tuple[str, ...],
        output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.6, 5.0), dpi=300)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.45, 1.05],
        left=0.055,
        right=0.985,
        bottom=0.12,
        top=0.88,
        wspace=0.24,
    )
    panel_c = outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)
    heat_grid = panel_c[0].subgridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], wspace=0.22
    )
    heat_axes = [fig.add_subplot(heat_grid[0, idx]) for idx in range(3)]
    colorbar_ax = fig.add_subplot(heat_grid[0, 3])
    summary_ax = fig.add_subplot(panel_c[1, 0])
    tradeoff_ax = fig.add_subplot(outer[0, 1])

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#F2F2F2")
    heatmap_image = None
    for ax, dataset in zip(heat_axes, datasets):
        heatmap_image = plot_heatmap(ax, pairwise_matrix(pairwise_df, dataset), dataset, cmap)
    plot_vector_gradient_colorbar(colorbar_ax, cmap)

    plot_pairwise_summary(summary_ax, pairwise_df, datasets)
    plot_tradeoff(tradeoff_ax, tradeoff_df, datasets)

    panel_c_left = heat_axes[0].get_position().x0
    panel_c_right = colorbar_ax.get_position().x1
    panel_d_left = tradeoff_ax.get_position().x0
    fig.text(0.012, 0.965, "c", fontsize=12, fontweight="bold", ha="left", va="top")
    fig.text(
        (panel_c_left + panel_c_right) / 2,
        0.955,
        "Human annotation heterogeneity",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="top",
    )
    fig.text(
        panel_d_left - 0.030,
        0.965,
        "d",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        (panel_d_left + tradeoff_ax.get_position().x1) / 2,
        0.955,
        "Accuracy–efficiency trade-off",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="top",
    )
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Fig. 3 panels d and g: human heterogeneity and accuracy-efficiency",
            "Subject": "Pairwise expert F1 and annotation time versus GT accuracy",
        },
    )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Fig. 3 panels d and g from existing annotation benchmark data."
    )
    parser.add_argument("--annotation-root", type=Path, default=Path(core.ANNOTATION_IMAGE_ROOT))
    parser.add_argument("--expert-root", type=Path, default=Path(core.EXPERT_ROOT))
    parser.add_argument("--f1-csv", type=Path, default=DEFAULT_F1_CSV)
    parser.add_argument("--human-time-csv", type=Path, default=DEFAULT_HUMAN_TIME_CSV)
    parser.add_argument("--runtime-csv", type=Path, default=DEFAULT_RUNTIME_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    parser.add_argument("--variant", type=str, default="denoise")
    parser.add_argument("--match-iou-thr", type=float, default=core.DEFAULT_MATCH_IOU_THR)
    args = parser.parse_args()
    if not 0.0 <= args.match_iou_thr <= 1.0:
        parser.error("--match-iou-thr must be between 0 and 1")
    return args


def main():
    args = parse_args()
    datasets = tuple(args.datasets)
    pairwise_df = calculate_pairwise_expert_f1(
        args.annotation_root,
        args.expert_root,
        datasets,
        args.match_iou_thr,
    )
    tradeoff_df = load_tradeoff_summary(
        args.f1_csv,
        args.human_time_csv,
        args.runtime_csv,
        datasets,
        args.variant,
    )
    plot_combined_figure(pairwise_df, tradeoff_df, datasets, args.output)

    pairwise_csv = args.output.with_name(f"{args.output.stem}_pairwise_f1.csv")
    tradeoff_csv = args.output.with_name(f"{args.output.stem}_tradeoff_summary.csv")
    pairwise_df.to_csv(pairwise_csv, index=False)
    tradeoff_df.to_csv(tradeoff_csv, index=False)

    print(f"Vector PDF saved to: {args.output}")
    print(f"Pairwise summary saved to: {pairwise_csv}")
    print(f"Trade-off summary saved to: {tradeoff_csv}")
    print("\nPairwise expert F1 means:")
    print(
        pairwise_df.groupby("dataset")["mean_f1"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
