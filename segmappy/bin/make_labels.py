from __future__ import print_function
from builtins import input
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ensure_segmappy_is_installed
from segmappy import Dataset
from segmappy import Config
from segmappy.tools.classifiertools import get_default_dataset

configfile = "default_training.ini"
config = Config(configfile)

# tweak config parameters
config.folder = "dataset18"
config.use_matches = False

# False: will go through unlabeled segments and append to the csv file
# True: will go through the segments labeled as CLASS and
#       rewrite the database at the end
RELABEL = True
AUTOWALLS = False
CLASS = 1
CLASSES = ["other", "car", "building"]

# load dataset
dataset = get_default_dataset(config)

segments, _, ids, n_ids, features, matches, _ = dataset.load()

lids = dataset.lids
lids_lookup = dict()
for i, lid in enumerate(lids):
    lids_lookup[lid] = i
labels = dataset.labels

if RELABEL:
    default_label = CLASS
    n_objects = np.sum(labels == CLASS)
    print("There are " + str(n_objects) + " " + CLASSES[CLASS] + "(s).")
else:
    default_label = 0
    fp_labels = open(os.path.join(config.folder, "labels_database.csv"), "a")

print("Default is " + str(default_label) + ":" + CLASSES[default_label] + ".")
print("Type q and then ENTER to quit.")
for i in range(ids.size):
    # skip if it's not the last duplicate
    if i + 1 < ids.size and ids[i] == ids[i + 1]:
        continue

    if RELABEL:
        if not ids[i] in lids_lookup or labels[lids_lookup[ids[i]]] != CLASS:
            continue
    else:
        if ids[i] in lids_lookup:
            continue

    fig = plt.figure(1)
    plt.clf()

    ax = fig.add_subplot(121, projection="3d")

    segment = segments[i]
    segment = segment - np.min(segment, axis=0)

    # Maintain aspect ratio on xy scale
    ax.set_xlim(0, np.max(segment[:, :]))
    ax.set_ylim(0, np.max(segment[:, :]))
    ax.set_zlim(0, np.max(segment[:, :]))

    x, y, z = np.hsplit(segment, segment.shape[1])
    ax.scatter(x, y, z, c=z)

    ax = fig.add_subplot(122)
    ax.scatter(x, y)
    ax.set_xlim(0, np.max(segment[:, :]))
    ax.set_ylim(0, np.max(segment[:, :]))

    plt.draw()
    plt.pause(0.001)

    while True:
        # autolabel
        if AUTOWALLS and not RELABEL:
            max_x = max(segment[:, 0])
            min_x = min(segment[:, 0])
            max_y = max(segment[:, 1])
            min_y = min(segment[:, 1])

            dist = np.linalg.norm([max_x - min_x, max_y - min_y])

            if dist > 6:
                print(str(ids[i]) + " autolabeled as wall")
                label = 2
                break

        # consider user input
        label = input(str(ids[i]) + " label: ")

        if not label:
            label = default_label
            break
        if label in ["0", "1", "2", "q"]:
            break

    if label == "q":
        break

    if RELABEL:
        labels[lids_lookup[ids[i]]] = label
    else:
        fp_labels.write(str(ids[i]) + " " + str(label) + "\n")
        fp_labels.flush()

if RELABEL:
    fp_labels = open(os.path.join(config.folder, "labels_database.csv"), "w")
    for lid, label in zip(lids, labels):
        fp_labels.write(str(lid) + " " + str(label) + "\n")
