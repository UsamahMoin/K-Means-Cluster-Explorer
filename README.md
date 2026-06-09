# K-Means Cluster Explorer

An interactive, research-backed visualization of a from-scratch k-means++
implementation across three UCI datasets: Pen Digits, Landsat Satellite, and
Yeast Proteins.

**Live site:** https://usamahmoin.github.io/K-Means-Cluster-Explorer/

## What the project shows

- A PCA projection of each dataset colored by cluster assignment or known label.
- Results for every `k` from 2 through 10.
- Standard k-means inertia, silhouette score, and adjusted Rand agreement.
- Cluster sizes and iteration-by-iteration convergence.
- A comparison between the most geometrically separated `k` and the known
  number of classes.

Known labels are never passed to k-means. They are used only after clustering
to calculate the adjusted Rand index and to provide an optional comparison
view.

## Method

The implementation uses k-means++ initialization, Lloyd updates, six
deterministic restarts, and the standard within-cluster sum of squared
Euclidean distances. The lowest-inertia restart is retained for each value of
`k`. Silhouette scores use a deterministic sample of at most 1,800 rows.

PCA is used only to draw the two-dimensional scatterplot. Clustering runs in
the original feature space.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 Project3.py
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Rebuild the visualization data

```bash
python3 tools/build_artifact.py
```

This regenerates `cluster-artifact.json` from the files in `UCI_datasets/`.
