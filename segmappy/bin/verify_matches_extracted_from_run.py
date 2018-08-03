from __future__ import print_function
from builtins import input
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
import random
from sklearn import metrics

import ensure_segmappy_is_installed
from segmappy import Dataset

DATASET_FOLDER = "for_generating_matches/drive18"

dataset = Dataset(folder=DATASET_FOLDER, use_matches=False, normalize_classes=False)
segments, _, ids, n_ids, features, matches = dataset.load()


# Find the convex hulls of the last view of each segment.
unique_ids = []
unique_segments = []
for i in range(ids.size):
    # skip if it's not the last duplicate
    if i + 1 < ids.size and ids[i] == ids[i + 1]:
        continue
    unique_ids.append(ids[i])
    unique_segments.append(segments[i])

for i in range(len(matches)):

    print("matches[i][0]: ", matches[i][0])
    print("matches[i][1]: ", matches[i][1])
    source_segment = unique_segments[unique_ids.index(matches[i][0])]
    target_segment = unique_segments[unique_ids.index(matches[i][1])]

    x_min = min(source_segment[:, 0].min(), target_segment[:, 0].min())
    x_max = max(source_segment[:, 0].max(), target_segment[:, 0].max())
    y_min = min(source_segment[:, 1].min(), target_segment[:, 1].min())
    y_max = max(source_segment[:, 1].max(), target_segment[:, 1].max())
    z_min = min(source_segment[:, 2].min(), target_segment[:, 2].min())
    z_max = max(source_segment[:, 2].max(), target_segment[:, 2].max())

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(
        target_segment[:, 0],
        target_segment[:, 1],
        target_segment[:, 2],
        color="blue",
        marker=".",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(source_segment[:, 0], source_segment[:, 1], source_segment[:, 2])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    plt.draw()
    plt.pause(0.001)
    input("Segment: ")
    plt.clf()
