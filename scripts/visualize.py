#!/usr/bin/env python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOISE_LABEL = -1
MAX_LEGEND_ENTRIES = 12


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot objects coloured by the cluster column.")
    parser.add_argument("path", help="CSV written by clustering_cli")
    parser.add_argument("-c", "--columns", nargs="+", required=True,
                        help="feature columns; two are plotted directly, more are projected with PCA")
    parser.add_argument("--cluster-column", default="cluster")
    parser.add_argument("--origin-column", default="origin")
    parser.add_argument("--no-centers", action="store_true")
    parser.add_argument("--save", metavar="FILE")
    return parser.parse_args()


def project_to_plane(features: np.ndarray, names: list[str]) -> tuple[np.ndarray, str, str]:
    if features.shape[1] == 2:
        return features, names[0], names[1]
    centered = features - features.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T, "PC1", "PC2"


def plot_clusters(points: np.ndarray, labels: np.ndarray, is_training: np.ndarray,
                  axis_names: tuple[str, str], draw_centers: bool):
    figure, axes = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    cluster_labels = [label for label in unique_labels if label != NOISE_LABEL]
    colormap = plt.get_cmap("tab10" if len(cluster_labels) <= 10 else "tab20")

    for label in unique_labels:
        members = points[labels == label]
        training_members = points[(labels == label) & is_training]
        if label == NOISE_LABEL:
            axes.scatter(members[:, 0], members[:, 1], s=24, color="gray", marker="x", label="noise")
            continue
        color = colormap(int(label) % colormap.N)
        axes.scatter(members[:, 0], members[:, 1], s=18, color=color, alpha=0.7,
                     label=f"cluster {label}")
        if len(training_members):
            axes.scatter(training_members[:, 0], training_members[:, 1], s=46, facecolor=color,
                         edgecolor="black", linewidths=1.3, label=f"training of cluster {label}")
        if draw_centers:
            center = members.mean(axis=0)
            axes.scatter(center[0], center[1], s=200, marker="X", color=color, edgecolor="black")

    axes.set_xlabel(axis_names[0])
    axes.set_ylabel(axis_names[1])
    axes.set_title(f"Clusters: {len(cluster_labels)}")
    if len(unique_labels) <= MAX_LEGEND_ENTRIES and not is_training.any():
        axes.legend()
    axes.grid(True, alpha=0.3)
    return figure


def main() -> int:
    arguments = parse_arguments()
    table = pd.read_csv(arguments.path, encoding="utf-8-sig")
    missing = [name for name in [*arguments.columns, arguments.cluster_column] if name not in table]
    if missing:
        print(f"Columns not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    table = table.dropna(subset=arguments.columns)
    points, x_name, y_name = project_to_plane(table[arguments.columns].to_numpy(float),
                                              arguments.columns)
    is_training = (table[arguments.origin_column] == "training").to_numpy() \
        if arguments.origin_column in table else np.zeros(len(table), dtype=bool)
    figure = plot_clusters(points, table[arguments.cluster_column].to_numpy(int), is_training,
                           (x_name, y_name), not arguments.no_centers)
    if arguments.save:
        figure.savefig(arguments.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {arguments.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
