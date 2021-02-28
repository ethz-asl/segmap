from __future__ import print_function
from builtins import input
import numpy as np
import os
from scipy.spatial import ConvexHull, Delaunay
import random

import ensure_segmappy_is_installed
from segmappy import Dataset
from segmappy.tools.hull import point_in_hull, n_points_in_hull, are_in_hull

DATASET_FOLDER = "/home/andrei/.segmap/training_datasets/bosch_train/"

dataset = Dataset(
    folder=DATASET_FOLDER, use_matches=False, normalize_classes=False,
    use_merges=False, require_relevance=0.05)
segments, _, classes, n_classes, _, _, _ = dataset.load()
duplicate_ids = dataset.duplicate_ids
timestamps = dataset.timestamps

# Find the convex hulls of the last view of each segment.
unique_ids = []
unique_segments = []
hulls = []
unique_centroids = []
unique_timestamps = []

for cls in np.unique(classes):
    sequence_ids = np.where(classes == cls)[0]
    last_id = np.argmax(duplicate_ids[sequence_ids])
    last_id = sequence_ids[last_id]
    segment = segments[last_id]
    timestamp = timestamps[last_id]

    unique_ids.append(last_id)
    unique_segments.append(segment)
    hull = ConvexHull(segment)
    hulls.append(hull)
    unique_centroids.append(np.mean(segment, 0))
    unique_timestamps.append(timestamp)

CENTROID_DISTANCE_THRESHOLD = 15.0
SUBSAMPLING_RATE = 5
HULL_VOLUME_THRESHOLD = 0.4
MIN_MATCHING_TIME = 60 * 10

matches = []
n_unique_ids = len(unique_ids)
for i in range(n_unique_ids):
    segment1 = unique_segments[i]
    hull1 = hulls[i]
    timestamp1 = unique_timestamps[i]

    print("#", i, "matches", len(matches))
    for j in range(i + 1, n_unique_ids):
        timestamp2 = unique_timestamps[j]

        deuclid = np.linalg.norm(unique_centroids[i] - unique_centroids[j])
        dtime = abs(timestamp1 - timestamp2)

        if deuclid <= CENTROID_DISTANCE_THRESHOLD and dtime >= MIN_MATCHING_TIME:
            segment2 = unique_segments[j]

            # precheck
            #x_min1 = np.min(segment1[:,0])
            #x_max1 = np.max(segment1[:,0])
            #x_min2 = np.min(segment2[:,0])
            #x_max2 = np.max(segment2[:,0])
            #if x_min1 > x_max2 or x_max1 < x_min2:
            #    continue

            #y_min1 = np.min(segment1[:,1])
            #y_max1 = np.max(segment1[:,1])
            #y_min2 = np.min(segment2[:,1])
            #y_max2 = np.max(segment2[:,1])
            #if y_min1 > y_max2 or y_max1 < y_min2:
            #    continue

            # volume check
            hull2 = hulls[j]

            subsampled_segment1 = segment1[0 : segment1.shape[0] : SUBSAMPLING_RATE]
            subsampled_segment2 = segment2[0 : segment2.shape[0] : SUBSAMPLING_RATE]

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

            inter = intersection_hull.volume
            union = hull1.volume + hull2.volume - inter
            iou = float(inter) / float(union)

            if (iou > HULL_VOLUME_THRESHOLD):
                matches.append([classes[unique_ids[i]], classes[unique_ids[j]]])
                print(classes[unique_ids[i]], classes[unique_ids[j]])

print("Number of matches: ", len(matches))
matches = np.asarray(matches).astype(np.int)
np.savetxt("matches_database.csv", (matches), fmt='%d', delimiter=" ")
