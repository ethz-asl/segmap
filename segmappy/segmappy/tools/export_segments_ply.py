# Export segments (already exported from SegMap) as .ply.
# To be used as input into 3DSmoothNet.
import numpy as np
import os
import open3d as o3d
import plyfile as pf

import tf.transformations as tf

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
    segment_save_dir = '/home/nikhilesh/Documents/3DSmoothNet/testing/' # base_dir + folder + "/3DSmoothNet/segment_clouds/"
    print("Data Folder: "+ base_dir + folder)

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
    
    for idx in range(len(segments)):
        if idx%50==0:
            print('Processing cloud ' + str(idx) + ' out of ' + str(len(segments)))
        # Identification.
        seg_id = classes[idx]
        view_id = duplicate_ids[idx]
        cloud_xyz = segments[idx][:,:3]

        # Compute centroid and add.
        cent_x = np.mean(cloud_xyz[:,0])
        cent_y = np.mean(cloud_xyz[:,1])
        cent_z = np.mean(cloud_xyz[:,2])
        cloud_xyz = np.insert(cloud_xyz, 0, [cent_x, cent_y, cent_z], axis=0)

        '''
        # Add a fake satelite cluster (3DSmoothNet related hacking..).
        # Compute radius of actual segment.
        seg_radius = np.sqrt(np.max(
            np.square(cloud_xyz[:,0] - cent_x) 
            + np.square(cloud_xyz[:,1] - cent_y) 
            + np.square(cloud_xyz[:,2] - cent_z)))

        # Add a fake keypoint with fake neighbours (far away from actual segment).
        fake_keypt = np.array([cent_x, cent_y, cent_z]) + 3.0*seg_radius
        fake_nn = np.vstack((fake_keypt,
            fake_keypt + np.array([0,0,1])*seg_radius,
            fake_keypt + np.array([1,0,1])*seg_radius,
            fake_keypt + np.array([1,1,0])*seg_radius,
            fake_keypt + np.array([0,1,1])*seg_radius,
            fake_keypt + np.array([1,0,0])*seg_radius,
            fake_keypt + np.array([1,1,1])*seg_radius,
            fake_keypt + np.array([0,1,0])*seg_radius))
        cloud_xyz = np.insert(cloud_xyz, 0, fake_nn, axis=0)
        '''
        
        # Just for some testing: Add random noise transformation.
        '''
        T_rand = tf.random_rotation_matrix(np.random.rand(3))
        T_rand[:3,3] = np.random.uniform(low=-2.0, high=2.0, size=(3,))
        noisy_cloud = np.matmul(T_rand, np.vstack((cloud_xyz.transpose(),np.ones(cloud_xyz.shape[0]))))
        cloud_xyz = noisy_cloud[:3,:].transpose()
        '''

        # Create structured array (plyfile).
        dt = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        cloud_xyz_ply = np.zeros(cloud_xyz.shape[0], dtype=dt)
        cloud_xyz_ply['x'] = cloud_xyz[:,0]
        cloud_xyz_ply['y'] = cloud_xyz[:,1]
        cloud_xyz_ply['z'] = cloud_xyz[:,2]


        # Write to file.
        vertex = cloud_xyz_ply
        filename = segment_save_dir + "segment_" +str(seg_id) + "_view_" + str(view_id) + ".ply"
        el = pf.PlyElement.describe(vertex, 'vertex')
        pf.PlyData([el]).write(filename)

        # Write ply with open3D (DON'T USE: open3D uses PLY_DOUBLE which will break PCL!).
        # pc_ply = o3d.geometry.PointCloud()
        # pc_ply.points = o3d.utility.Vector3dVector(cloud_xyz)
        # o3d.io.write_point_cloud(segment_save_dir+"open3D_segment_"+str(seg_id)+"_view_"+str(view_id)+".ply", pc_ply)
        # o3d.io.write_point_cloud(segment_save_dir+"pcd/"+"segment_"+str(seg_id)+"_view_"+str(view_id)+".pcd", pc_ply)

    return

if __name__ == '__main__':
    main()