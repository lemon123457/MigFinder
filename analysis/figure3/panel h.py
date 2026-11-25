"""Generate Fig. 3 Panel h: MigFinder vs. Consensus GT agreement.

The input is the per-image shape-metric table produced by
``panel c-f.py``. Consensus GT and MigFinder values are paired by
dataset and image. Area and diameter are converted from pixels using the
specified pixel size. The displayed R-squared is the squared Pearson
correlation calculated from all paired images pooled across datasets.
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
import os
from typing import Dict, List, Tuple

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'migfinder_matplotlib'))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    '/path/of/results/figure3/gt_benchmark/denoise/gt_vs_migfinder_shape_metrics.csv'
)
DEFAULT_OUTPUT = (
    '/path/of/results/figure3/gt_benchmark/denoise/plots/panel_h_agreement_biological_measurements.pdf'
)

DATASETS = ("L929", "NRK", "4T1")
METHODS = ("GT", "MigFinder")

# Experiment calibration: 174 pixels correspond to 50 micrometers.
# Override this value with --pixel-size-um if another acquisition uses a
# different calibration.
DEFAULT_PIXEL_SIZE_UM = 50.0 / 174.0

DATASET_STYLE = {
    "L929": {"color": "#B279D6", "marker": "o"},
    "NRK": {"color": "#4C78C4", "marker": "s"},
    "4T1": {"color": "#E34A9B", "marker": "^"},
}

METRICS = (
    {
        "column": "number_per_image",
        "title": "Number per image",
        "factor": lambda pixel_size_um: 1.0,
        "fixed_limits": None,
    },
    {
        "column": "average_area",
        "title": "Average area (µm²)",
        "factor": lambda pixel_size_um: pixel_size_um ** 2,
        "fixed_limits": None,
    },
    {
        "column": "average_intensity",
        "title": "Average intensity (a.u.)",
        "factor": lambda pixel_size_um: 1.0,
        "fixed_limits": None,
    },
    {
        "column": "average_diameter",
        "title": "Average diameter (µm)",
        "factor": lambda pixel_size_um: pixel_size_um,
        "fixed_limits": None,
    },
)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def load_paired_measurements(
        csv_path: Path,
        datasets: Tuple[str, ...],
        variant: str) -> pd.DataFrame:
    """Load and strictly pair GT/MigFinder measurements by dataset and image."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    metric_columns = [metric["column"] for metric in METRICS]
    required_columns = {"dataset", "image", "method", *metric_columns}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {missing_columns}")

    if "variant" in df.columns and variant:
        df = df[df["variant"] == variant].copy()
    df = df[df["dataset"].isin(datasets) & df["method"].isin(METHODS)].copy()

    duplicate_mask = df.duplicated(["dataset", "image", "method"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask, ["dataset", "image", "method"]
        ].sort_values(["dataset", "image", "method"])
        raise ValueError(
            "Duplicate dataset/image/method rows found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    for column in metric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    wide = df.set_index(["dataset", "image", "method"])[metric_columns].unstack("method")
    missing_methods = [
        method
        for method in METHODS
        if not all((column, method) in wide.columns for column in metric_columns)
    ]
    if missing_methods:
        raise ValueError(f"Missing measurement methods: {missing_methods}")

    records: List[Dict[str, object]] = []
    for (dataset, image), row in wide.iterrows():
        record: Dict[str, object] = {"dataset": dataset, "image": image}
        for column in metric_columns:
            record[f"{column}_gt"] = row[(column, "GT")]
            record[f"{column}_migfinder"] = row[(column, "MigFinder")]
        records.append(record)

    paired = pd.DataFrame(records)
    if paired.empty:
        raise ValueError("No paired GT/MigFinder measurements were found.")

    for dataset in datasets:
        count = int((paired["dataset"] == dataset).sum())
        if count == 0:
            raise ValueError(f"No paired measurements found for dataset {dataset}.")
    return paired


def pooled_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Return squared Pearson correlation for finite paired observations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return float("nan")
    correlation = float(np.corrcoef(x, y)[0, 1])
    return correlation ** 2


def nice_shared_limits(
        x: np.ndarray,
        y: np.ndarray,
        fixed_limits: Tuple[float, float] = None) -> Tuple[float, float]:
    """Choose identical, zero-based x/y limits with publication-friendly ticks."""
    if fixed_limits is not None:
        return fixed_limits

    pooled = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    pooled = pooled[np.isfinite(pooled)]
    if not len(pooled):
        return 0.0, 1.0

    data_max = max(float(pooled.max()), 1e-9)
    rough_step = data_max * 1.04 / 5.0
    magnitude = 10.0 ** np.floor(np.log10(rough_step))
    fraction = rough_step / magnitude
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    step = nice_fraction * magnitude
    upper = float(np.ceil(data_max * 1.04 / step) * step)
    return 0.0, upper


def plot_panel_h(
        paired_df: pd.DataFrame,
        datasets: Tuple[str, ...],
        pixel_size_um: float,
        output_path: Path) -> Dict[str, float]:
    """Create and save the compact 2x2 agreement panel."""
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be a positive finite value.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.9), dpi=300)
    r_squared_values: Dict[str, float] = {}

    for ax, metric in zip(axes.flat, METRICS):
        column = metric["column"]
        factor = float(metric["factor"](pixel_size_um))
        gt_column = f"{column}_gt"
        migfinder_column = f"{column}_migfinder"

        all_x: List[float] = []
        all_y: List[float] = []
        for dataset in datasets:
            dataset_df = paired_df[paired_df["dataset"] == dataset]
            x = dataset_df[gt_column].to_numpy(dtype=float) * factor
            y = dataset_df[migfinder_column].to_numpy(dtype=float) * factor
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            all_x.extend(x.tolist())
            all_y.extend(y.tolist())

            style = DATASET_STYLE[dataset]
            ax.scatter(
                x,
                y,
                s=25,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor=style["color"],
                linewidth=0.55,
                alpha=0.78,
                zorder=3,
                rasterized=False,
            )

        pooled_x = np.asarray(all_x, dtype=float)
        pooled_y = np.asarray(all_y, dtype=float)
        r_squared = pooled_r_squared(pooled_x, pooled_y)
        r_squared_values[column] = r_squared

        limits = nice_shared_limits(
            pooled_x, pooled_y, fixed_limits=metric["fixed_limits"]
        )
        ax.plot(
            limits,
            limits,
            linestyle=(0, (3.2, 2.2)),
            color="0.30",
            linewidth=0.9,
            zorder=1,
        )
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(metric["title"], pad=5.0)
        ax.text(
            0.055,
            0.945,
            rf"$R^2 = {r_squared:.2f}$" if np.isfinite(r_squared) else r"$R^2 = n/a$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=DATASET_STYLE[dataset]["marker"],
            markersize=6.0,
            markerfacecolor=DATASET_STYLE[dataset]["color"],
            markeredgecolor=DATASET_STYLE[dataset]["color"],
            alpha=0.85,
            label=dataset,
        )
        for dataset in datasets
    ]

    fig.suptitle(
        "Agreement of biological measurements\n(MigFinder vs. Consensus GT)",
        x=0.52,
        y=0.985,
        fontsize=10.5,
        fontweight="bold",
        linespacing=1.08,
    )
    fig.supxlabel("Consensus GT", x=0.52, y=0.075, fontsize=9)
    fig.supylabel("MigFinder", x=0.035, y=0.52, fontsize=9)
    fig.text(0.012, 0.985, "h", ha="left", va="top", fontsize=12, fontweight="bold")
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.005),
        ncol=len(datasets),
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.6,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.155,
        top=0.865,
        wspace=0.30,
        hspace=0.38,
    )
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Panel h: Agreement of biological measurements",
            "Subject": "MigFinder versus Consensus GT paired measurements",
        },
    )
    plt.close(fig)
    return r_squared_values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Fig. 3 Panel h paired biological-measurement scatter plots."
    )
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="denoise",
        help="Variant to select when the input CSV contains a variant column.",
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=DEFAULT_PIXEL_SIZE_UM,
        help=(
            "Image calibration in micrometers per pixel. Area is multiplied by "
            "this value squared and diameter by this value."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = tuple(args.datasets)
    paired_df = load_paired_measurements(args.input, datasets, args.variant)
    r_squared_values = plot_panel_h(
        paired_df,
        datasets,
        args.pixel_size_um,
        args.output,
    )

    print(f"Panel h saved to: {args.output}")
    print(f"Pixel calibration: {args.pixel_size_um:g} µm/pixel")
    print(f"Total paired images: {len(paired_df)}")
    for metric in METRICS:
        column = metric["column"]
        print(f"{metric['title']}: R^2={r_squared_values[column]:.6f}")


if __name__ == "__main__":
    main()
