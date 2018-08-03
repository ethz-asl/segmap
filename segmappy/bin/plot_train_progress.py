import numpy as np
import os
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder")
parser.add_argument("--epoch")
parser.add_argument("--train", action="store_true")
args = parser.parse_args()

# size in pixels of each block in the image
PX = 5

import json

with open(os.path.join(args.folder, "centers.json"), "r") as fp:
    centers = np.array(json.load(fp))

with open(os.path.join(args.folder, "%s.json" % args.epoch), "r") as fp:
    debug = json.load(fp)

h = 0
w = 0
for sequence_debug in debug:
    if sequence_debug:
        train = sequence_debug[0][0]
        if train == args.train:
            h = max(h, len(sequence_debug))
            w += 1

img = np.zeros((h * PX, w * PX, 3), dtype=np.uint8)

col = 0
for sequence_debug in debug:
    if sequence_debug:
        train = sequence_debug[0][0]
        if train != args.train:
            continue

        for i, segment_debug in enumerate(sequence_debug):
            test, cls, cls_prob, top5_classes, _ = segment_debug

            if cls == top5_classes[0]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            img[i * PX : (i + 1) * PX, col * PX : (col + 1) * PX] = color

        col += 1

img = img[::-1, :, :]

from skimage.io import imsave

imsave("pred%s.png" % args.epoch, img)
