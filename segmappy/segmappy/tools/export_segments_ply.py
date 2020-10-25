# Export the exported segments (from SegMap) as .ply.
# To be used as input into 3DSmoothNet.
import numpy as np
import os
import open3d as o3d

from segmappy import Config
from segmappy import Dataset

# read config file
configfile = "default_training.ini"
config = Config(configfile)


def main():
    """Extract segment point clouds and save in ply format.
    """
    # Path path to segments_database.csv (what is exported from SegMap).

    # Load Dataset.
    dataset = Dataset(
        folder=config.cnn_test_folder,
        base_dir=config.base_dir,
        keep_match_thresh=0.7,
        require_change=0.0,
        use_merges=False,
        require_relevance=0.0,
        require_diff_points=0.0,
        use_matches=False,
        min_class_size=0.0,
    )

    segments, positions, classes, n_classes, _, _, _ = dataset.load()

    print('--------------------')
    print('Loaded Dataset')
    print('Number of Segments: '+str(len(segments)))
    print('Number of Positions: '+str(len(positions)))
    print('Number of Descriptors: '+str(len(dataset.features)))


    # Convert each segment to .ply point cloud.

    # Save (viewID, segID).


    return

if __name__ == '__main__':
    main()