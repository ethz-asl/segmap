from __future__ import print_function

import numpy as np

## LOAD DATA ##
###############
print("Loading segments.")
from import_export import load_segments
segments, ids = load_segments(folder="/tmp/segment_matcher/")
from import_export import load_matches
matches = load_matches(folder="/tmp/segment_matcher/")

visualize=True
print("q<Enter> to quit")
for match in matches:
  if visualize:
    segment1 = segments[match[0]]
    segment2 = segments[match[1]]

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    X = segment1[:,0]
    Y = segment1[:,1]
    Z = segment1[:,2]
    fig = plt.figure(1)
    plt.cla()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(X, Y, Z)
    plt.pause(0.05)

    X = segment2[:,0]
    Y = segment2[:,1]
    Z = segment2[:,2]
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(X, Y, Z)
    plt.pause(0.05)

    ## Get User Input ##
    ####################
    keys = raw_input(">")
    if keys == 'q':
        visualize = False

plt.close("all")
