from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

import ensure_segmappy_is_installed
from segmappy import Config
from segmappy.tools.classifiertools import get_default_dataset, get_default_preprocessor

# read config file
configfile = "default_training.ini"
config = Config(configfile)

# load dataset and preprocessor
dataset = get_default_dataset(config)
preprocessor = get_default_preprocessor(config)

segments, positions, classes, n_classes, _, _, _ = dataset.load(
    preprocessor=preprocessor
)


# Class size distribution
class_histogram, _ = np.histogram(classes, bins=range(n_classes + 1))
class_size_histogram, bins = np.histogram(
    class_histogram, bins=range(np.max(class_histogram) + 2)
)

plt.figure(1)
plt.bar(bins[:-1], class_size_histogram)

# Segment size distribution
segments_size = []
for segment in segments:
    segments_size.append(np.max(segment, axis=0) - np.min(segment, axis=0))
segments_size = np.array(segments_size)

plt.figure(2)
segments_size = np.ceil(segments_size)
for axis in range(segments_size.shape[1]):
    axis_sizes = segments_size[:, axis]
    sizes_histogram, bins = np.histogram(
        axis_sizes, bins=np.arange(np.min(axis_sizes), np.max(axis_sizes) + 2)
    )
    sizes_cdf = np.cumsum(sizes_histogram) / float(np.sum(sizes_histogram))

    plt.subplot(3, 1, axis + 1)
    plt.plot(bins[:-1], sizes_cdf)

plt.show()
