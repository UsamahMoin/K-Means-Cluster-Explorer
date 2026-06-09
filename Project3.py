"""Run the reproducible k-means analysis for all included UCI datasets."""

import json

from tools.build_artifact import OUTPUT_PATH, build_artifact


def main() -> None:
    artifact = build_artifact()
    OUTPUT_PATH.write_text(
        json.dumps(artifact, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )

    print("K-Means Cluster Explorer")
    print("Objective: within-cluster sum of squared Euclidean distances")
    print("Initialization: k-means++ with 6 deterministic restarts")
    for key, dataset in artifact["datasets"].items():
        best_k = dataset["bestSilhouetteK"]
        known_k = dataset["knownClassK"]
        best_run = dataset["runs"][str(best_k)]
        known_run = dataset["runs"][str(known_k)]
        print(f"\n{dataset['title']} ({key})")
        print(
            f"  {dataset['rows']} rows, {dataset['features']} features, "
            f"{dataset['knownClasses']} known classes"
        )
        print(
            f"  Best silhouette: k={best_k}, "
            f"score={best_run['silhouette']:.3f}"
        )
        print(
            f"  At known class count k={known_k}: "
            f"ARI={known_run['adjustedRand']:.3f}, "
            f"inertia/point={known_run['inertiaPerPoint']:.3f}"
        )
    print(f"\nWrote {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
