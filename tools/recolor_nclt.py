# Script to split augmented cloud into a fully colored one and and full
# semantic colors with Alpha=1.
import argparse
import struct
import sys

import rosbag
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image


def main():
    # BGR semantic colors.
    segmentation_id_color = {1:[42,174,203], 
      2:[224,172,177], 
      3:[145,183,160], 
      4:[137,241,224], 
      5:[132,224,232], 
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
      32:[190,247,227], 
      33:[216,254,163], 
      34:[158,253,220]}
    parser = argparse.ArgumentParser(description="Augment Point Cloud")
    parser.add_argument('input_bag', metavar='input_bag', type=str, help='bag file with LiDAR, image, labels and TF')
    parser.add_argument('output_bag', metavar='output_bag', type=str, help='bag file with augmented cloud')
    args = parser.parse_args()

    in_bag = rosbag.Bag(args.input_bag)
    out_bag = rosbag.Bag(args.output_bag, 'w')
    i=0
    for topic, pcl, t in in_bag.read_messages(topics=['/augmented_cloud']):
        i+=1
        print(i)
        print(t)
        points = point_cloud2.read_points(pcl)
        color_points = []
        sem_points = []

        for point in points:
          bgra = struct.unpack('BBBB', struct.pack('I', point[3]))  # Why did we call the field rgba, even though it's bgra?
          color_points.append([point[0],
            point[1],
            point[2],
            float(bgra[0] /255.0),
            float(bgra[1] /255.0),
            float(bgra[2] /255.0)])

          # Lookup semantic color.
          label = int(bgra[3]/7.0)

          bgr_sem = segmentation_id_color.get(label,[0,0,0])
          sem_points.append([point[0],
            point[1],
            point[2],
            float(bgr_sem[0]),  # We should scale down by 255.0 because RVIZ wants values in (0,1), but we only care about values and not correct viz for debugging...
            float(bgr_sem[1]),
            float(bgr_sem[2])])

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('b', 12, PointField.FLOAT32, 1),
          PointField('g', 16, PointField.FLOAT32, 1),
          PointField('r', 20, PointField.FLOAT32, 1)]

        header = pcl.header
        color_cloud = point_cloud2.create_cloud(header, fields, color_points)
        sem_cloud = point_cloud2.create_cloud(header, fields, sem_points)

        out_bag.write('/rgb_cloud', color_cloud, color_cloud.header.stamp, False)
        out_bag.write('/sem_cloud', sem_cloud, sem_cloud.header.stamp, False)
   

if __name__ == '__main__':
    main()
