from __future__ import print_function
from builtins import input
import numpy as np

import ensure_segmappy_is_installed
from segmappy import Config
from segmappy.tools.classifiertools import (
    get_default_dataset,
    get_default_preprocessor,
    visualize,
)

configfile = "default_training.ini"
config = Config(configfile)

# load dataset
dataset = get_default_dataset(config)
preprocessor = get_default_preprocessor(config)

segments, positions, classes, n_classes, _, _, _ = dataset.load(
    preprocessor=preprocessor
)
duplicate_classes = dataset.duplicate_classes
duplicate_ids = dataset.duplicate_ids

# different visualization commands
VOXELIZED = True
CMD_VOXELIZED = ["v", "voxelize"]
CMD_CLASS = ["c", "class"]
CMD_LAST = ["l", "last"]
CMD_CLASS_INFO = ["i", "info"]
CMD_DUPLICATE_INFO = ["d", "duplicate"]
CMD_QUIT = ["q", "quit"]


def get_class(cls, classes, duplicate_classes):
    class_ids = np.where(classes == cls)[0]
    class_duplicates = duplicate_classes[class_ids]

    return class_ids, class_duplicates


def print_class_info(cls, class_ids, class_duplicates):
    unique_duplicates = np.unique(class_duplicates)
    print(
        "Class %d contains %d sequences and %d segments"
        % (cls, len(unique_duplicates), len(class_ids))
    )
    for unique_duplicate in unique_duplicates:
        print(
            "Duplicate sequence %d has %d segments"
            % (unique_duplicate, np.sum(class_duplicates == unique_duplicate))
        )


while True:
    cmd = input(">> ")
    cmd = cmd.strip()

    if not cmd:
        continue

    cmd = cmd.split(" ")

    if cmd[0] in CMD_CLASS:
        cls = int(cmd[1])
        class_ids, class_duplicates = get_class(cls, classes, duplicate_classes)
        print_class_info(cls, class_ids, class_duplicates)

        info_segments = [
            ("view %d/%d" % (duplicate_classes[segment_id], duplicate_ids[segment_id]))
            for segment_id in class_ids
        ]
        show_all = False

        segments_show = [segments[i] for i in class_ids]
    elif cmd[0] in CMD_LAST:
        segments_show = []
        info_segments = []
        show_all = True

        for cls in map(int, cmd[1:]):
            class_ids, class_duplicates = get_class(cls, classes, duplicate_classes)
            for class_duplicate in np.unique(class_duplicates):
                last_id = np.max(class_ids[class_duplicates == class_duplicate])
                segments_show.append(segments[last_id])
                info_segments.append(
                    "class %d view %d/%d"
                    % (cls, duplicate_classes[last_id], duplicate_ids[last_id])
                )
    elif cmd[0] in CMD_CLASS_INFO:
        cls = int(cmd[1])
        class_ids, class_duplicates = get_class(cls, classes, duplicate_classes)
        print_class_info(cls, class_ids, class_duplicates)
        continue
    elif cmd[0] in CMD_DUPLICATE_INFO:
        duplicate = int(cmd[1])
        class_ids = np.where(duplicate_classes == duplicate)[0]
        if class_ids.size > 0:
            cls = np.unique(classes[class_ids])[0]
            print("Duplicate belongs to class %d" % cls)
        continue
    elif cmd[0] in CMD_VOXELIZED:
        VOXELIZED = not VOXELIZED
        print("Voxelization is set to %s" % VOXELIZED)
        continue
    elif cmd[0] in CMD_QUIT:
        break
    else:
        print("Invalid command")
        continue

    if not len(segments_show):
        print("Class is empty or does not exist")
        continue

    if VOXELIZED:
        segments_show = preprocessor._rescale_coordinates(segments_show)
        voxelized_segments = preprocessor._voxelize(segments_show)

        segments_show = []
        for voxelized_segment in voxelized_segments:
            x, y, z = np.where(voxelized_segment > 0)
            segments_show.append(np.stack((x, y, z), axis=1))

    visualize(segments_show, info_segments, show_all=show_all)
