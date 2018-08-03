from __future__ import print_function
import numpy as np
import os


def voxel_grid_to_cloud_by_probability(voxel_grid, thresh):
    cloud = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                if voxel_grid[i, j, k] > thresh:
                    cloud.append([i, j, k])
    return np.asarray(cloud)


def voxel_grid_to_cloud_by_n_points(voxel_grid, n_points):
    cloud_all = []
    probs = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                cloud_all.append([i, j, k])
                probs.append(voxel_grid[i, j, k])

    cloud = []
    indices = np.argsort(probs)[::-1][:n_points]

    for i in range(len(indices)):
        print("prob", probs[indices[i]])
        cloud.append(cloud_all[indices[i]])

    return np.asarray(cloud)


def compute_jaccard_index(seg1, seg2):
    union = [seg >= 1 for seg in seg1 + seg2]
    intersection = [seg == 2 for seg in seg1 + seg2]
    return float(np.sum(intersection)) / float(np.sum(union))


def compute_jaccard_indices_on_voxels_lists(voxels1, voxels2):
    return [compute_jaccard_index(vox1, vox2) for vox1, vox2 in zip(voxels1, voxels2)]


def keep_n_most_probable_voxels(voxel_grid, n_points):
    cloud_all = []
    probs = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                cloud_all.append([i, j, k])
                probs.append(voxel_grid[i, j, k])

    indices = np.argsort(probs)[::-1][:n_points]

    filtered = np.zeros(voxel_grid.shape)
    for i in range(len(indices)):
        filtered[
            cloud_all[indices[i]][0], cloud_all[indices[i]][1], cloud_all[indices[i]][2]
        ] = 1
    return filtered


def keep_voxels_above_threshold(voxel_grid, threshold):
    filtered = np.zeros(voxel_grid.shape)
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                if voxel_grid[i, j, k] >= threshold:
                    filtered[i, j, k] = 1
    return filtered


def voxel_grid_to_cloud(voxel_grid):
    cloud = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                if voxel_grid[i, j, k] == 1:
                    cloud.append([i, j, k])
    return np.asarray(cloud)


def compute_accuracies(seg1, seg2):
    # assuming that seg1 is the true segments
    n_pos = np.sum(seg1)
    n_neg = np.sum([vox == 0 for vox in seg1])
    seg1 = seg1 * 2
    seg_sum = seg1 + seg2
    n_true_pos = np.sum([vox == 3 for vox in seg_sum])
    n_false_pos = np.sum([vox == 1 for vox in seg_sum])
    n_false_negative = np.sum([vox == 2 for vox in seg_sum])
    n_true_negative = np.sum([vox == 0 for vox in seg_sum])

    TPR = float(n_true_pos) / float(n_pos)
    FNR = float(n_false_negative) / float(n_pos)
    FPR = float(n_false_pos) / float(n_neg)
    TNR = float(n_true_negative) / float(n_neg)
    return [TPR, FNR, FPR, TNR]


def compute_accuracies_on_voxel_lists(voxels1, voxels2):
    return [compute_accuracies(vox1, vox2) for vox1, vox2 in zip(voxels1, voxels2)]
