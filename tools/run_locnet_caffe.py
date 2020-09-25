#!/usr/bin/env python
import numpy
from numpy import zeros
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
    model_file = '/home/marius/.segmap/locnet/models/kitti_delta_range.caffemodel'
    config_file = '/home/marius/.segmap/locnet/cfg/kitti_delta_range.prototxt'
    bag = rosbag.Bag(bag_file)
    caffe.set_mode_cpu()
    net = caffe.Net(config_file, model_file, 0)
    max_distance = 200
    d_min = 0.0
    d_max = 1.5
    image_width = 640
    image_height = 480
    bucket_count = 80       # given from network
    network_input_size = 64 # given from network
    delta_i_b = (1.0 / bucket_count) * (d_max - d_min)

    ii = 0
    for topic, pcl, t in bag.read_messages(topics=['/augmented_cloud']):
        points = point_cloud2.read_points(pcl)
        azimuth_index = 0
        line_index = 0
        ring_index = 0
        histogram = zeros([1, 3, bucket_count, network_input_size])
        is_new_line = True
        point_valid = True
        last_point_valid = True
        last_x = 0
        last_y = 0
        dist = 0

        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
            r = math.sqrt(x*x + y*y + z*z)

            if r > max_distance:
                point_valid = False
            else:
                point_valid = True

            if point_valid and last_point_valid and not(is_new_line):

                dist = math.sqrt((x - last_x)**2 + (y-last_y)**2)
                for i in range(0, bucket_count):

                    if d_min + i * delta_i_b > dist:
                        histogram[0, 0, i, ring_index] += 1
                        break

            # update for next point
            last_point_valid = point_valid
            is_new_line = False
            last_x = x
            last_y = y
            azimuth_index += 1
            if azimuth_index == image_width:
                line_index += 1
                azimuth_index = 0
                is_new_line = True    
            ring_index = math.trunc(line_index * network_input_size / image_height)

        output = net.forward_all(**{"data": histogram})
        ii += 1

        if ii == 100:
            break


if __name__ == '__main__':
    main()
