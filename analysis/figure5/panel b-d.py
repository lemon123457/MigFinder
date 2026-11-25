#!/usr/bin/env python3
"""Reproduce Figure 5 track-level S1/S2 labels from the frozen feature table.

Input: the ``track_features`` sheet in ``figure5_source_data.xlsx``. That sheet
contains Figure 4 track features; no
detection, tracking, image access, or previously assigned labels are needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "output" / "figure5_source_data.xlsx"
FEATURES = [
    "median_area_um2",
    "median_circularity",
    "median_solidity",
    "median_aspect_ratio",
    "median_eccentricity",
    "median_intensity_au",
    "median_local_density_per_um2",
    "median_speed_um_h",
    "growth_rate_um2_min",
    "decay_rate_um2_min",
    "area_fluctuation_cv",
    "observed_duration_min",
]
SEED = 20261105
MIN_FRAMES = 8
CORRELATION_CUTOFF = 0.93


def load_tracks(path: Path) -> pd.DataFrame:
    tracks = pd.read_excel(path, sheet_name="track_features")
    required = {"global_track_id", "sequence_id", "track_frame_count", "data_split", *FEATURES}
    missing = required.difference(tracks.columns)
    if missing:
        raise ValueError(f"Missing track_features columns: {sorted(missing)}")
    tracks = tracks.loc[tracks["track_frame_count"] >= MIN_FRAMES].copy().reset_index(drop=True)
    if tracks.empty or tracks["global_track_id"].duplicated().any():
        raise ValueError("Expected nonempty data with one row per global_track_id")
    if set(tracks["data_split"]) != {"Discovery", "Held-out"}:
        raise ValueError("data_split must contain Discovery and Held-out")
    if tracks.groupby("sequence_id")["data_split"].nunique().gt(1).any():
        raise ValueError("One FOV appears in both Discovery and Held-out")
    return tracks


def prepare_features(tracks: pd.DataFrame, discovery: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit all preprocessing on Discovery tracks, as in the Figure 5 analysis."""
    raw = tracks[FEATURES].apply(pd.to_numeric, errors="coerce")
    available = [name for name in FEATURES if raw.loc[discovery, name].isna().mean() <= 0.30]
    values = raw[available].to_numpy(float)
    medians = np.nanmedian(values[discovery], axis=0)
    values = np.where(np.isfinite(values), values, medians)
    lower = np.nanpercentile(values[discovery], 1, axis=0)
    upper = np.nanpercentile(values[discovery], 99, axis=0)
    values = np.clip(values, lower, upper)

    variable = np.isfinite(np.nanstd(values[discovery], axis=0)) & (
        np.nanstd(values[discovery], axis=0) > 1e-10
    )
    values = values[:, variable]
    available = [name for name, keep in zip(available, variable) if keep]
    medians, lower, upper = medians[variable], lower[variable], upper[variable]

    correlation = pd.DataFrame(values[discovery], columns=available).corr(method="spearman").abs()
    retained = []
    for name in available:
        if all(
            not np.isfinite(correlation.loc[name, previous])
            or correlation.loc[name, previous] < CORRELATION_CUTOFF
            for previous in retained
        ):
            retained.append(name)
    indices = [available.index(name) for name in retained]
    values = values[:, indices]
    scaler = StandardScaler().fit(values[discovery])
    scaled = scaler.transform(values)
    parameters = {
        "features": retained,
        "impute_median": medians[indices].tolist(),
        "winsor_1_percent": lower[indices].tolist(),
        "winsor_99_percent": upper[indices].tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    return scaled, parameters


def assign_subtypes(scaled: np.ndarray, discovery: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fitted = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(scaled[discovery])
    centroids = np.vstack([scaled[discovery][fitted == label].mean(axis=0) for label in (0, 1)])
    # Match the Figure 5 numbering rule: sort centroids on their one-dimensional PCA axis.
    axis = PCA(n_components=1, random_state=SEED).fit_transform(centroids).ravel()
    order = np.argsort(axis)
    mapping = {int(old): new for new, old in enumerate(order)}
    discovery_labels = np.asarray([mapping[int(label)] for label in fitted], dtype=int)
    ordered_centroids = np.vstack(
        [scaled[discovery][discovery_labels == label].mean(axis=0) for label in (0, 1)]
    )
    all_labels = np.argmin(cdist(scaled, ordered_centroids), axis=1)
    all_labels[discovery] = discovery_labels
    return all_labels + 1, ordered_centroids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "release_output")
    args = parser.parse_args()

    tracks = load_tracks(args.input)
    discovery = tracks["data_split"].eq("Discovery").to_numpy()
    scaled, parameters = prepare_features(tracks, discovery)
    labels, centroids = assign_subtypes(scaled, discovery)
    assignments = tracks[["sequence_id", "global_track_id", "data_split"]].copy()
    assignments["subtype"] = [f"S{label}" for label in labels]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output_dir / "figure5_s1_s2_assignments.csv", index=False)
    metadata = {
        "source": str(args.input),
        "minimum_track_frames": MIN_FRAMES,
        "seed": SEED,
        "method": "Discovery-only preprocessing; Ward k=2; Held-out nearest centroid",
        "preprocessing": parameters,
        "centroids": centroids.tolist(),
        "counts": {key: int(value) for key, value in assignments["subtype"].value_counts().sort_index().items()},
    }
    (args.output_dir / "figure5_s1_s2_model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(assignments)} tracks: {metadata['counts']}")


if __name__ == "__main__":
    main()
