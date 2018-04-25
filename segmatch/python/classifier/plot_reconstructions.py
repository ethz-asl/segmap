from __future__ import print_function
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import Dataset
from IPython import embed 
from scipy.spatial import ConvexHull, Delaunay
import random
from sklearn import metrics


from import_export import load_segments_no_duplicates

#run_2018-01-24T11:50:48.334206



originals, sids_originals = load_segments_no_duplicates(
    '/tmp/online_matcher/', 'run_2018-01-30T14:47:52.511763_segments.csv')
reconstructions, sids_reconstructions = load_segments_no_duplicates(
    '/tmp/online_matcher/', 'run_2018-01-30T14:47:52.511763_reconstructions.csv')

#order along z to solve rendering issue in matplotlib
reconstructions_ordered = []
for reconstruction in reconstructions:
    reconstructions_ordered.append(reconstruction[reconstruction[:,2].argsort()])
reconstructions = reconstructions_ordered

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
        print('prob', probs[indices[i]])
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
        filtered[cloud_all[indices[i]][0], cloud_all[indices[i]][1], cloud_all[indices[i]][2]] = 1
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
    #assuming that seg1 is the true segments
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


MIN_N_POINTS = 300
POINT_SIZE = 20

n_saved = 0

from mpl_toolkits.mplot3d import *

for i in range(len(originals)):
    art3d.zalpha = lambda *args:args[0]
    fig = plt.figure(1, frameon=False)
    plt.clf()
    
    
    
    original = originals[i]
    reconstruction = reconstructions[i]
    
    if original.shape[0] > MIN_N_POINTS:
        seg = original
        axes_min = np.array(np.min(seg, axis=0))
        axes_max = np.array(np.max(seg, axis=0))
        print('axes_min max before ', str(axes_min[2]), ' ',  str(axes_max[2]))
        X = seg[:,0]
        Y = seg[:,1]
        Z = seg[:,2]
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max() / 2.0
        
        y_before = axes_max[2]
        
        seg = reconstruction
        axes_min_temp = np.minimum(axes_min, np.min(seg, axis=0))
        axes_max_temp = np.maximum(axes_max, np.max(seg, axis=0))
        
        if (axes_max_temp[2] > y_before + 1):
            seg[:,2] = seg[:,2] - (axes_max_temp[2] - y_before)
            axes_min = np.minimum(axes_min, np.min(seg, axis=0))
            axes_max = np.maximum(axes_max, np.max(seg, axis=0))
        else:
               axes_min = axes_min_temp 
               axes_max = axes_max_temp
               
        print('axes_min max after ', str(axes_min[2]), ' ',  str(axes_max[2]))
        X = seg[:,0]
        Y = seg[:,1]
        Z = seg[:,2]
        max_range = max(max_range,np.array([X.max()-X.min(), Y.max()-Y.min()]).max() / 2.0)
        
        ax = fig.add_subplot(121, projection='3d')
        seg = original
        #marker='.', lw = 0,
        ax.scatter(seg[:,0], seg[:,1], seg[:,2], s=POINT_SIZE  / max_range,c = seg[:,2], edgecolors='none',  depthshade=False, cmap="jet_r",
                   vmin = axes_min[2], vmax = axes_max[2])
        ax.set_xlim(axes_min[0], axes_max[0])
        ax.set_ylim(axes_min[1], axes_max[1])
        ax.set_zlim(axes_min[2], axes_max[2])
        mid_x = (seg[:,0].max()+seg[:,0].min()) * 0.5
        mid_y = (seg[:,1].max()+seg[:,1].min()) * 0.5
        mid_z = (seg[:,2].max()+seg[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_aspect(1)
        ax.grid(b=False)
	ax.patch.set_facecolor('black')
        ax.set_axis_off()
        
        plt.style.use('dark_background')
        plt.setp(ax.get_xmajorticklabels(), visible=False)
        plt.setp(ax.get_ymajorticklabels(), visible=False)
        plt.setp(ax.get_zmajorticklabels(), visible=False)
        
        ax_rec = fig.add_subplot(122, projection='3d')
        seg = reconstruction
        ax_rec.scatter(seg[:,0], seg[:,1], seg[:,2], s=POINT_SIZE / max_range,c = seg[:,2], edgecolors='none', depthshade=True, cmap="jet_r",
                   vmin = axes_min[2], vmax = axes_max[2])
        ax_rec.set_xlim(axes_min[0], axes_max[0])
        ax_rec.set_ylim(axes_min[1], axes_max[1])
        ax_rec.set_zlim(axes_min[2], axes_max[2])
        mid_x = (seg[:,0].max()+seg[:,0].min()) * 0.5
        mid_y = (seg[:,1].max()+seg[:,1].min()) * 0.5
        mid_z = (seg[:,2].max()+seg[:,2].min()) * 0.5
        ax_rec.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_rec.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_rec.set_zlim(mid_z - max_range, mid_z + max_range)
        ax_rec.set_aspect(1)
        ax_rec.grid(b=False)
	ax_rec.patch.set_facecolor('black')
        ax_rec.set_axis_off()
        
        plt.setp(ax_rec.get_xmajorticklabels(), visible=False)
        plt.setp(ax_rec.get_ymajorticklabels(), visible=False)
        plt.setp(ax_rec.get_zmajorticklabels(), visible=False)
        plt.style.use('dark_background')
        
        plt.draw()
        plt.pause(0.001)
        
                
        command = raw_input('Cmd: ')
        
        while command != '':
            if command == 'c':
                ax.azim = ax_rec.azim
                ax.elev = ax_rec.elev
                plt.draw()
                plt.pause(0.001)
            
            if command == 's':
                plt.savefig('reconstructions/seg_'+ str(n_saved) +'.pdf')
                plt.savefig('reconstructions/seg_'+ str(n_saved) +'.png')
                n_saved = n_saved + 1
            command = raw_input('Cmd: ')
    
    #        ax.set_xlim(0, 35)
    #    ax.set_ylim(0, 35)
    #    ax.set_zlim(0, 15)
    