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
    base_dir = config.base_dir
    folder = config.cnn_test_folder
    print("Data Folder: "+base_dir+folder)

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
        normalize_classes=False
    )

    segments, positions, classes, n_classes, _, _, _ = dataset.load()
    duplicate_ids = dataset.duplicate_ids
    descriptors = dataset.features

    assert str(len(segments))==str(len(classes))==str(len(duplicate_ids))==str(len(positions))==str(len(descriptors))

    print('--------------------')
    print('Loaded Dataset')
    print('Number of Segments: '+str(len(segments))+" of type: "+str(type(segments)))
    print('Number of Segment IDs: '+str(len(classes))+" of type: "+str(type(classes)))
    print('Number of View IDs: '+str(len(duplicate_ids))+" of type: "+str(type(duplicate_ids)))
    print('Number of Positions: '+str(len(positions))+" of type: "+str(type(positions)))
    print('Number of Descriptors: '+str(len(descriptors))+" of type: "+str(type(descriptors)))
    print(".....................")

    # Convert each segment and save as .ply point cloud.
    folder_path = base_dir + folder + "/segment_clouds/"
    for idx in range(100):#len(segments)):
        seg_id = classes[idx]
        view_id = duplicate_ids[idx]
        cloud_xyz = segments[idx][:,:3]     
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(cloud_xyz)
        o3d.io.write_point_cloud(folder_path+"segment_"+str(seg_id)+"_view_"+str(view_id)+".ply", pc_ply)
        # o3d.io.write_point_cloud("TestData/sync"+str(idx)+".pcd", pc_ply)
    return

if __name__ == '__main__':
    main()