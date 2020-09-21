#!/usr/bin/env python
import numpy
import os
import math
import tf.transformations as transformations
import rosbag
import rospy
import sys
import caffe
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


def main():
    bag_file = '/home/marius/.segmap/bosch/augmented_bosch1.bag'
    bag = rosbag.Bag(bag_file)

    max_distance = 200
    image_width = 640
    image_height = 480
    bucket_count = 20

    i = 0
    for topic, pcl, t in bag.read_messages(topics=['/augmented_cloud']):
        points = point_cloud2.read_points(pcl)
        j = 0
        azimuth_index = 0
        ring_index = 0
        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
            r = math.sqrt(x*x + y*y + z*z)
            # if r > max_distance:
            #     continue


            phi = math.atan2(y, x)
            theta = math.acos(z/r)

            print(theta)
            print(azimuth_index)
            print(ring_index)
            print("......")


            azimuth_index += 1
            if azimuth_index == image_width:
                ring_index += 1
                azimuth_index = 0


            # j += 1
            # if j == 1000:
            #     break

        i += 1
        print("-------")
        if i == 1:
            break
    # for topic, tf, t in bag.read_messages(topics=['/tf']):

    # for topic, tf_static, t in bag.read_messages(topics=['/tf_static']):


if __name__ == '__main__':
    main()
