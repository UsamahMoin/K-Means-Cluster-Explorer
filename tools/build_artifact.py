#!/usr/bin/env python3
"""Build deterministic browser data for the k-means cluster explorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "UCI_datasets"
OUTPUT_PATH = ROOT / "cluster-artifact.json"
K_VALUES = range(2, 11)
RANDOM_STATE = 42
MAX_SCATTER_POINTS = 1400

DATASETS = {
    "pendigits": {
        "file": "pendigits_training.txt",
        "title": "Pen Digits",
        "shortDescription": "Pen trajectories resampled into eight x-y coordinate pairs.",
        "domain": "Handwritten digit recognition",
        "sourceUrl": "https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits",
        "sourceDoi": "10.24432/C5MG6K",
    },
    "satellite": {
        "file": "satellite_training.txt",
        "title": "Landsat Satellite",
        "shortDescription": "Multi-spectral values from 3x3 pixel neighborhoods.",
        "domain": "Remote sensing and land-cover pixels",
        "sourceUrl": "https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite",
        "sourceDoi": "10.24432/C55887",
    },
    "yeast": {
        "file": "yeast_training.txt",
        "title": "Yeast Proteins",
        "shortDescription": "Eight biochemical signals related to protein localization.",
        "domain": "Cellular localization of proteins",
        "sourceUrl": "https://archive.ics.uci.edu/dataset/110/yeast",
        "sourceDoi": "10.24432/C5KG68",
    },
}


def squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def initialize_kmeans_plus_plus(
    data: np.ndarray,
    cluster_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = [data[rng.integers(0, len(data))].copy()]
    closest_squared = np.sum((data - centers[0]) ** 2, axis=1)

    for _ in range(1, cluster_count):
        total = closest_squared.sum()
        if total <= 0:
            centers.append(data[rng.integers(0, len(data))].copy())
            continue
        probabilities = closest_squared / total
        next_index = rng.choice(len(data), p=probabilities)
        centers.append(data[next_index].copy())
        candidate_squared = np.sum((data - centers[-1]) ** 2, axis=1)
        closest_squared = np.minimum(closest_squared, candidate_squared)

    return np.asarray(centers)


def run_kmeans(
    data: np.ndarray,
    cluster_count: int,
    seed: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> dict:
    rng = np.random.default_rng(seed)
    centers = initialize_kmeans_plus_plus(data, cluster_count, rng)
    history = []
    labels = np.zeros(len(data), dtype=int)

    for iteration in range(max_iterations):
        distances = squared_distances(data, centers)
        labels = distances.argmin(axis=1)
        inertia = float(distances[np.arange(len(data)), labels].sum())
        history.append(inertia)

        updated = centers.copy()
        for cluster_index in range(cluster_count):
            members = data[labels == cluster_index]
            if len(members):
                updated[cluster_index] = members.mean(axis=0)
            else:
                farthest_index = np.argmax(
                    distances[np.arange(len(data)), labels]
                )
                updated[cluster_index] = data[farthest_index]

        center_shift = float(np.linalg.norm(updated - centers))
        centers = updated
        if center_shift <= tolerance:
            break

    distances = squared_distances(data, centers)
    labels = distances.argmin(axis=1)
    inertia = float(distances[np.arange(len(data)), labels].sum())
    if not history or inertia != history[-1]:
        history.append(inertia)

    return {
        "centers": centers,
        "labels": labels,
        "inertia": inertia,
        "iterations": iteration + 1,
        "history": history,
    }


def best_of_restarts(
    data: np.ndarray,
    cluster_count: int,
    dataset_offset: int,
    restart_count: int = 6,
) -> dict:
    runs = [
        run_kmeans(
            data,
            cluster_count,
            RANDOM_STATE + dataset_offset * 1000 + cluster_count * 50 + restart,
        )
        for restart in range(restart_count)
    ]
    return min(runs, key=lambda run: run["inertia"])


def choose_scatter_indices(row_count: int, dataset_offset: int) -> np.ndarray:
    if row_count <= MAX_SCATTER_POINTS:
        return np.arange(row_count)
    rng = np.random.default_rng(RANDOM_STATE + dataset_offset)
    return np.sort(
        rng.choice(row_count, size=MAX_SCATTER_POINTS, replace=False)
    )


def round_list(values: np.ndarray, digits: int = 6) -> list:
    return np.round(values.astype(float), digits).tolist()


def build_dataset(key: str, metadata: dict, dataset_offset: int) -> dict:
    frame = pd.read_csv(DATA_DIR / metadata["file"], sep=r"\s+", header=None)
    data = frame.iloc[:, :-1].to_numpy(dtype=float)
    true_labels = frame.iloc[:, -1].astype(str).to_numpy()
    scatter_indices = choose_scatter_indices(len(data), dataset_offset)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projection = pca.fit_transform(data)
    projection_scale = np.max(np.abs(projection), axis=0)
    projection_scale[projection_scale == 0] = 1
    normalized_projection = projection / projection_scale

    runs = {}
    for cluster_count in K_VALUES:
        result = best_of_restarts(data, cluster_count, dataset_offset)
        sample_size = min(1800, len(data))
        silhouette = silhouette_score(
            data,
            result["labels"],
            sample_size=sample_size,
            random_state=RANDOM_STATE,
        )
        projected_centers = pca.transform(result["centers"]) / projection_scale
        runs[str(cluster_count)] = {
            "inertia": round(result["inertia"], 6),
            "inertiaPerPoint": round(result["inertia"] / len(data), 6),
            "silhouette": round(float(silhouette), 6),
            "adjustedRand": round(
                float(adjusted_rand_score(true_labels, result["labels"])),
                6,
            ),
            "iterations": result["iterations"],
            "history": [
                round(value / len(data), 6)
                for value in result["history"]
            ],
            "assignments": result["labels"][scatter_indices].astype(int).tolist(),
            "centers": [
                round_list(center, 6)
                for center in projected_centers
            ],
            "clusterSizes": np.bincount(
                result["labels"],
                minlength=cluster_count,
            ).astype(int).tolist(),
        }

    best_silhouette_k = max(
        K_VALUES,
        key=lambda value: runs[str(value)]["silhouette"],
    )
    known_class_count = len(np.unique(true_labels))
    known_class_k = min(max(known_class_count, min(K_VALUES)), max(K_VALUES))

    return {
        **metadata,
        "rows": len(data),
        "features": data.shape[1],
        "knownClasses": known_class_count,
        "knownClassLabels": sorted(np.unique(true_labels).tolist(), key=lambda value: int(value)),
        "classCounts": {
            label: int((true_labels == label).sum())
            for label in sorted(np.unique(true_labels).tolist(), key=lambda value: int(value))
        },
        "missingValues": int(frame.isna().sum().sum()),
        "duplicateRows": int(frame.duplicated().sum()),
        "featureRange": [
            round(float(data.min()), 6),
            round(float(data.max()), 6),
        ],
        "pcaExplainedVariance": round_list(
            pca.explained_variance_ratio_,
            6,
        ),
        "scatter": {
            "points": [
                {
                    "x": round(float(normalized_projection[index, 0]), 6),
                    "y": round(float(normalized_projection[index, 1]), 6),
                    "label": str(true_labels[index]),
                }
                for index in scatter_indices
            ],
            "sampleSize": len(scatter_indices),
        },
        "runs": runs,
        "bestSilhouetteK": best_silhouette_k,
        "knownClassK": known_class_k,
    }


def build_artifact() -> dict:
    datasets = {
        key: build_dataset(key, metadata, index)
        for index, (key, metadata) in enumerate(DATASETS.items())
    }
    return {
        "title": "K-Means Cluster Explorer",
        "method": {
            "objective": "Within-cluster sum of squared Euclidean distances",
            "initialization": "k-means++",
            "restarts": 6,
            "maxIterations": 100,
            "kValues": list(K_VALUES),
            "labelsUsedForTraining": False,
            "silhouetteSampleLimit": 1800,
            "randomState": RANDOM_STATE,
        },
        "datasets": datasets,
    }


if __name__ == "__main__":
    artifact = build_artifact()
    OUTPUT_PATH.write_text(
        json.dumps(artifact, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.name}")
    for key, dataset in artifact["datasets"].items():
        best_k = dataset["bestSilhouetteK"]
        expected_k = dataset["knownClassK"]
        print(
            f"{key}: silhouette best k={best_k} "
            f"({dataset['runs'][str(best_k)]['silhouette']:.3f}), "
            f"known-class k={expected_k}, "
            f"ARI={dataset['runs'][str(expected_k)]['adjustedRand']:.3f}"
        )
