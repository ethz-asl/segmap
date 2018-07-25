from __future__ import print_function
from builtins import input
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn import metrics

import ensure_segmappy_is_installed
from segmappy import Dataset
from segmappy.tools.import_export import load_segments_no_duplicates
from segmappy.tools.voxeltools import *
from mpl_toolkits.mplot3d import *

# run_2018-01-24T11:50:48.334206


originals, sids_originals = load_segments_no_duplicates(
    "/tmp/online_matcher/", "run_2018-01-30T14:47:52.511763_segments.csv"
)
reconstructions, sids_reconstructions = load_segments_no_duplicates(
    "/tmp/online_matcher/", "run_2018-01-30T14:47:52.511763_reconstructions.csv"
)

# order along z to solve rendering issue in matplotlib
reconstructions_ordered = []
for reconstruction in reconstructions:
    reconstructions_ordered.append(reconstruction[reconstruction[:, 2].argsort()])
reconstructions = reconstructions_ordered


MIN_N_POINTS = 300
POINT_SIZE = 20

n_saved = 0


for i in range(len(originals)):
    art3d.zalpha = lambda *args: args[0]
    fig = plt.figure(1, frameon=False)
    plt.clf()

    original = originals[i]
    reconstruction = reconstructions[i]

    if original.shape[0] > MIN_N_POINTS:
        seg = original
        axes_min = np.array(np.min(seg, axis=0))
        axes_max = np.array(np.max(seg, axis=0))
        print("axes_min max before ", str(axes_min[2]), " ", str(axes_max[2]))
        X = seg[:, 0]
        Y = seg[:, 1]
        Z = seg[:, 2]
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min()]).max() / 2.0

        y_before = axes_max[2]

        seg = reconstruction
        axes_min_temp = np.minimum(axes_min, np.min(seg, axis=0))
        axes_max_temp = np.maximum(axes_max, np.max(seg, axis=0))

        if axes_max_temp[2] > y_before + 1:
            seg[:, 2] = seg[:, 2] - (axes_max_temp[2] - y_before)
            axes_min = np.minimum(axes_min, np.min(seg, axis=0))
            axes_max = np.maximum(axes_max, np.max(seg, axis=0))
        else:
            axes_min = axes_min_temp
            axes_max = axes_max_temp

        print("axes_min max after ", str(axes_min[2]), " ", str(axes_max[2]))
        X = seg[:, 0]
        Y = seg[:, 1]
        Z = seg[:, 2]
        max_range = max(
            max_range, np.array([X.max() - X.min(), Y.max() - Y.min()]).max() / 2.0
        )

        ax = fig.add_subplot(121, projection="3d")
        seg = original
        # marker='.', lw = 0,
        ax.scatter(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            s=POINT_SIZE / max_range,
            c=seg[:, 2],
            edgecolors="none",
            depthshade=False,
            cmap="jet_r",
            vmin=axes_min[2],
            vmax=axes_max[2],
        )
        ax.set_xlim(axes_min[0], axes_max[0])
        ax.set_ylim(axes_min[1], axes_max[1])
        ax.set_zlim(axes_min[2], axes_max[2])
        mid_x = (seg[:, 0].max() + seg[:, 0].min()) * 0.5
        mid_y = (seg[:, 1].max() + seg[:, 1].min()) * 0.5
        mid_z = (seg[:, 2].max() + seg[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_aspect(1)
        ax.grid(b=False)
        ax.patch.set_facecolor("black")
        ax.set_axis_off()

        plt.style.use("dark_background")
        plt.setp(ax.get_xmajorticklabels(), visible=False)
        plt.setp(ax.get_ymajorticklabels(), visible=False)
        plt.setp(ax.get_zmajorticklabels(), visible=False)

        ax_rec = fig.add_subplot(122, projection="3d")
        seg = reconstruction
        ax_rec.scatter(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            s=POINT_SIZE / max_range,
            c=seg[:, 2],
            edgecolors="none",
            depthshade=True,
            cmap="jet_r",
            vmin=axes_min[2],
            vmax=axes_max[2],
        )
        ax_rec.set_xlim(axes_min[0], axes_max[0])
        ax_rec.set_ylim(axes_min[1], axes_max[1])
        ax_rec.set_zlim(axes_min[2], axes_max[2])
        mid_x = (seg[:, 0].max() + seg[:, 0].min()) * 0.5
        mid_y = (seg[:, 1].max() + seg[:, 1].min()) * 0.5
        mid_z = (seg[:, 2].max() + seg[:, 2].min()) * 0.5
        ax_rec.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_rec.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_rec.set_zlim(mid_z - max_range, mid_z + max_range)
        ax_rec.set_aspect(1)
        ax_rec.grid(b=False)
        ax_rec.patch.set_facecolor("black")
        ax_rec.set_axis_off()

        plt.setp(ax_rec.get_xmajorticklabels(), visible=False)
        plt.setp(ax_rec.get_ymajorticklabels(), visible=False)
        plt.setp(ax_rec.get_zmajorticklabels(), visible=False)
        plt.style.use("dark_background")

        plt.draw()
        plt.pause(0.001)

        command = input("Cmd: ")

        while command != "":
            if command == "c":
                ax.azim = ax_rec.azim
                ax.elev = ax_rec.elev
                plt.draw()
                plt.pause(0.001)

            if command == "s":
                plt.savefig("reconstructions/seg_" + str(n_saved) + ".pdf")
                plt.savefig("reconstructions/seg_" + str(n_saved) + ".png")
                n_saved = n_saved + 1
            command = input("Cmd: ")

    #        ax.set_xlim(0, 35)
    #    ax.set_ylim(0, 35)
    #    ax.set_zlim(0, 15)
