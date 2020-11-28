# Script to split augmented cloud into a fully colored one and and full
# semantic colors with Alpha=1.
import argparse
import numpy as np
import struct
import sys


import rosbag
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image

from sklearn import linear_model


def main():
    # BGR semantic colors.
    segmentation_id_color = {1:[42,174,203], 
      2:[224,172,177], 
      3:[145,183,160],  # Bicycle/Motorcycle
      4:[137,241,224], 
      5:[132,224,232],  # Fence
      6:[105,64,153],
      7:[227,217,179], 
      8:[91,214,208], 
      9:[219,213,192], 
      10:[229,90,95], 
      11:[248,71,170], 
      12:[199,173,249],
      13:[205,228,85], 
      14:[208,160,121], 
      15:[180,238,141], 
      16:[53,246,59], 
      17:[50,96,227],
      18:[190,247,227], 
      19:[0,0,0], 
      31:[142,190,77], 
      32:[190,247,227], # Terrain -> Landscape
      33:[216,254,163], 
      34:[158,253,220]}
    parser = argparse.ArgumentParser(description="Augment Point Cloud")
    parser.add_argument('input_bag', metavar='input_bag', type=str, help='bag file with LiDAR, image, labels and TF')
    parser.add_argument('output_bag', metavar='output_bag', type=str, help='bag file with augmented cloud')
    parser.add_argument('ids_to_remove', metavar='id2del', type=str, help='comma separated list of semantic IDs to remove')
    parser.add_argument('potential_ids_to_remove', metavar='pot_id2del', type=str, help='comma separated list of potential semantic IDs to remove')
    args = parser.parse_args()

    print(args.ids_to_remove)
    in_bag = rosbag.Bag(args.input_bag)
    out_bag = rosbag.Bag(args.output_bag, 'w')
    deletion_lbl_ids = map(int, args.ids_to_remove.split(','))
    potential_lbl_ids = map(int, args.potential_ids_to_remove.split(','))
    print('Points with the following semantic labels will be removed!')
    print(deletion_lbl_ids)
    print(44 in deletion_lbl_ids)
    i=0
    for topic, pcl, t in in_bag.read_messages(topics=['/augmented_cloud']):
        i+=1
        print(i)
        print(t)
        points = point_cloud2.read_points(pcl)
        color_points = [] # Points with real BGR
        sem_points = [] # Points with semantic BGR
        aug_point_gs = [] # Points without labels representing ground (road, landscape,...)

        # GS v2.
        obvious_ground_pts = [] # Points that have obvious ground labels (roads, terrain,...)
        potential_ground_pts = [] # Points that have labels that are known to be wrongly assigned to ground (vegetation, small house, unknown).
        ground_points = [] # Just for visualization: The fitted ground plane
        aug_point_gs2 = [] # The resulting augmented cloud WITHOUT ground.

        for point in points:
          # Extract BGR and Alpha (semantic label).
          bgra = struct.unpack('BBBB', struct.pack('I', point[3]))  # Why did we call the field rgba, even though it's bgra?
          
          # Color cloud.
          color_points.append([point[0],
            point[1],
            point[2],
            float(bgra[0] /255.0),
            float(bgra[1] /255.0),
            float(bgra[2] /255.0)])

          # Lookup semantic label.
          label = int(bgra[3]/7.0)

          # Semantic cloud.
          bgr_sem = segmentation_id_color.get(label,[0,0,0])
          sem_points.append([point[0],
            point[1],
            point[2],
            float(bgr_sem[0]),  # We should scale down by 255.0 because RVIZ wants values in (0,1), but we only care about values and not correct viz for debugging...
            float(bgr_sem[1]),
            float(bgr_sem[2])])

          # Filtered cloud without certain labels.
          if not(label in deletion_lbl_ids):
            aug_point_gs.append(point)
          

          # GS v2##############################
          if label in deletion_lbl_ids:
            obvious_ground_pts.append(point)
          elif label in potential_lbl_ids:
            potential_ground_pts.append(point)
          else:
            aug_point_gs2.append(point)
          #####################################

        # GS v2################################
        print('Number of obvious ground_points: ' + str(len(obvious_ground_pts)))
        print('Number of potential ground points: ' + str(len(potential_ground_pts)))
        print('Number of safe points: ' + str(len(aug_point_gs2)))

        # Fit RANSAC plane.
        obvious_ground_pts = np.asarray(obvious_ground_pts)
        print('obvious_ground_pts: ' + str(obvious_ground_pts.shape))
        XY_ground = obvious_ground_pts[:,:2]
        Z_ground = obvious_ground_pts[:,2]
        ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        ransac.fit(XY_ground, Z_ground)
        print('Fitted plane!')

        # Second pass for potential ground points.
        del_cand = np.asarray(potential_ground_pts)
        print('del_cand' + str(del_cand.shape))
        Z_cand = ransac.predict(del_cand[:,:2])
        print('Z_cand: ' + str(Z_cand.shape))
        keep = np.square((Z_cand - del_cand[:,2])) > np.square(0.7) # Any deletion candidate that is within 70cm of the ground plane is considered ground as well.
        print('keep: ' + str(keep.shape))
        keep_second = del_cand[keep]
        print('keep_second: ' + str(keep_second.shape))
        keep_second = keep_second.tolist()
        print('keep_second: ' + str(len(keep_second)))
        print(keep_second[0])
        
        # Final selection of good points.
        print('safe points: ' + str(len(aug_point_gs2)))
        print(aug_point_gs2[0])
        print(np.array(aug_point_gs2).shape)
        aug_point_gs2.append(keep_second)   # Something fucks up here....
        print(np.array(aug_point_gs2).shape)
        print('Final safe points: ' + str(len(aug_point_gs2)))
        print('Fun: ' + str(np.array(aug_point_gs2).shape))
        ########################################

        fields_xyzbgr = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('b', 12, PointField.FLOAT32, 1),
          PointField('g', 16, PointField.FLOAT32, 1),
          PointField('r', 20, PointField.FLOAT32, 1)]

        fields_xyzbgra = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1)]


        header = pcl.header
        color_cloud = point_cloud2.create_cloud(header, fields_xyzbgr, color_points)
        sem_cloud = point_cloud2.create_cloud(header, fields_xyzbgr, sem_points)
        seg_cloud = point_cloud2.create_cloud(header, fields_xyzbgra, aug_point_gs) # Input cloud but with certain labels segmented out.
        seg_cloud2 = point_cloud2.create_cloud(header, fields_xyzbgra, aug_point_gs2)

        out_bag.write('/rgb_cloud', color_cloud, color_cloud.header.stamp, False)
        out_bag.write('/sem_cloud', sem_cloud, sem_cloud.header.stamp, False)
        out_bag.write('/augmented_cloud_no_ground', seg_cloud, seg_cloud.header.stamp, False)
        out_bag.write('/augmented_cloud_no_ground2', seg_cloud2, seg_cloud2.header.stamp, False)

    print('Finished!')
    out_bag.close()
   

if __name__ == '__main__':
    main()
