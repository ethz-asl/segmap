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
from segmappy.tools.hull import point_in_hull, n_points_in_hull, are_in_hull

DATASET_FOLDER = "/home/andrei/.segmap/training_datasets/bosch_test/"
FILE_PATH = DATASET_FOLDER + "/matches_database.csv"
if os.path.isfile(FILE_PATH):
    os.remove(FILE_PATH)

dataset = Dataset(
    folder=DATASET_FOLDER, use_matches=False, normalize_classes=False,
    use_merges=False, require_relevance=0.05)
segments, _, ids, n_ids, _, _, _ = dataset.load()

# Find the convex hulls of the last view of each segment.
unique_ids = []
unique_segments = []
hulls = []
unique_centroids = []

for i in range(ids.size):
    segment = segments[i]

    # skip if it's not the last duplicate
    if i + 1 < ids.size and ids[i] == ids[i + 1]:
        continue

    unique_ids.append(ids[i])
    unique_segments.append(segment)
    hull = ConvexHull(segment)
    hulls.append(hull)
    unique_centroids.append(np.mean(segment, 0))

CENTROID_DISTANCE_THRESHOLD = 3.0
SUBSAMPLING_RATE = 5
HULL_VOLUME_THRESHOLD = 0.33
N_ID_TO_SKIP = 0
PLOT_3D = False
PLOT_MAP = False

matches = []

n_unique_ids = len(unique_ids)

for i in range(n_unique_ids):
    segment1 = unique_segments[i]
    hull1 = hulls[i]
    if i + N_ID_TO_SKIP > n_unique_ids:
        continue
    range_j = range(n_unique_ids - i - 1 - N_ID_TO_SKIP)
    range_j = [x + i + 1 + N_ID_TO_SKIP for x in range_j]
    print("i: ", i, " matches ", len(matches))
    for j in range_j:
        if (
            np.linalg.norm(unique_centroids[i] - unique_centroids[j])
            <= CENTROID_DISTANCE_THRESHOLD
        ):

            segment2 = unique_segments[j]
            hull2 = hulls[j]

            subsampled_segment1 = segment1[0 : segment1.shape[0] : SUBSAMPLING_RATE]
            subsampled_segment2 = segment2[0 : segment2.shape[0] : SUBSAMPLING_RATE]

            n_points_1_in_2 = n_points_in_hull(segment1, hull2)
            n_points_2_in_1 = n_points_in_hull(segment2, hull1)

            ratio_1_in_2 = float(n_points_1_in_2) / float(segment1.shape[0])
            ratio_2_in_1 = float(n_points_2_in_1) / float(segment2.shape[0])

            ins, outs = are_in_hull(segment1, hull2)
            if len(ins) < 5:
                continue
            points_1_in_2 = segment1[ins, :]

            ins, outs = are_in_hull(segment2, hull1)
            if len(ins) < 5:
                continue
            points_2_in_1 = segment2[ins, :]

            intersection_hull = ConvexHull(
                np.concatenate((points_1_in_2, points_2_in_1))
            )

            volume_ratio_1 = intersection_hull.volume / hull1.volume
            volume_ratio_2 = intersection_hull.volume / hull2.volume

            if (
                volume_ratio_1 > HULL_VOLUME_THRESHOLD
                and volume_ratio_2 > HULL_VOLUME_THRESHOLD
            ):

                matches.append([unique_ids[i], unique_ids[j]])

                if PLOT_3D:
                    x_min = min(source_segment[:, 0].min(), target_segment[:, 0].min())
                    x_max = max(source_segment[:, 0].max(), target_segment[:, 0].max())
                    y_min = min(source_segment[:, 1].min(), target_segment[:, 1].min())
                    y_max = max(source_segment[:, 1].max(), target_segment[:, 1].max())
                    z_min = min(source_segment[:, 2].min(), target_segment[:, 2].min())
                    z_max = max(source_segment[:, 2].max(), target_segment[:, 2].max())

                    print("Source volume ratio: ", source_volume_ratio)
                    print("Target volume ratio: ", target_volume_ratio)

                    fig = plt.figure(1)
                    ax = fig.add_subplot(1, 2, 1, projection="3d")
                    ins, outs = are_in_hull(target_segment, source_hull)
                    ax.scatter(
                        target_segment[ins, 0],
                        target_segment[ins, 1],
                        target_segment[ins, 2],
                        color="blue",
                        marker=".",
                    )
                    ax.scatter(
                        target_segment[outs, 0],
                        target_segment[outs, 1],
                        target_segment[outs, 2],
                        color="red",
                        marker=".",
                    )
                    for simplex in target_hulls[i].simplices:
                        plt.plot(
                            target_segment[simplex, 0],
                            target_segment[simplex, 1],
                            target_segment[simplex, 2],
                            "k-",
                        )
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_zlim(z_min, z_max)

                    ax = fig.add_subplot(1, 2, 2, projection="3d")
                    ax.scatter(
                        source_segment[:, 0], source_segment[:, 1], source_segment[:, 2]
                    )
                    for simplex in source_hull.simplices:
                        plt.plot(
                            source_segment[simplex, 0],
                            source_segment[simplex, 1],
                            source_segment[simplex, 2],
                            "k-",
                        )
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_zlim(z_min, z_max)
                    plt.draw()
                    plt.pause(0.001)
                    input("Segment: ")
                    plt.clf()
                elif PLOT_MAP:
                    fig = plt.figure(1)
                    ax = fig.add_subplot(1, 2, 1)
                    plt.plot(segment1[:, 0], segment1[:, 1], "o")
                    # for simplex in target_hulls[i].simplices:
                    #    plt.plot(target_segment[simplex, 0], target_segment[simplex, 1], 'k-')

                    ax = fig.add_subplot(1, 2, 2)
                    plt.plot(segment2[:, 0], segment2[:, 1], "o")
                    # plt.draw()
                    # plt.pause(0.001)
                    # input('Segment: ')

if PLOT_MAP:
    plt.draw()
    plt.pause(0.001)
    input("Segment: ")
    plt.clf()

print("Number of matches: ", len(matches))

matches = np.asarray(matches).astype(np.int)
np.savetxt(FILE_PATH, (matches), fmt='%d', delimiter=" ")
