# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology
import numpy
import os
import yaml
import tf.transformations as transformations
import rosbag
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image


def main():

    with open(r'/home/marius/segmap_ws/src/segmap/segmapper/launch/bosch/segmentation_id_color.yaml') as file:
        segmentation_ids = yaml.load(file, Loader=yaml.FullLoader)

    bag_file = '/home/marius/.segmap/bosch/bosch.bag'
    # bag_file = '/media/scratch1/bosch/bosch.bag'
    out_bag_file = '/home/marius/.segmap/bosch/augmented_bosch.bag'
    # out_bag_file = '/media/scratch1/bosch/augmented_bosch.bag'
    bag = rosbag.Bag(bag_file)
    out_bag = rosbag.Bag(out_bag_file, 'w')
    image_width = 640
    image_height = 480
    f_x = image_width/2
    f_y = image_width/2
    c_x = image_width/2
    c_y = image_height/2

    tf_drone_cam = transformations.quaternion_matrix(numpy.array(
        [-0.5, 0.5, -0.5, 0.5]))
    tf_drone_cam[0, 3] = 0.3
    tf_drone_cam[1, 3] = 0.0
    tf_drone_cam[2, 3] = 0.0

    tf_drone_lidar = transformations.quaternion_matrix(numpy.array(
        [0.0, 0.0, 0.0, 1.0]))
    tf_drone_lidar[0, 3] = 0.0
    tf_drone_lidar[1, 3] = 0.0
    tf_drone_lidar[2, 3] = 0.2

    tf_cam_drone = transformations.inverse_matrix(tf_drone_cam)
    tf_cam_lidar = numpy.dot(tf_cam_drone, tf_drone_lidar)

    images = []

    i = 0

    for topic, image, t in bag.read_messages(topics='/airsim_drone/Depth_cam'):
        images.append(image)

        i += 1
        if i == 40:
            break

    image_iterator = 0
    i = 0
    for topic, lidar_pcl, t in bag.read_messages(topics=['/airsim_drone/Lidar']):
        while(images[image_iterator].header.stamp < lidar_pcl.header.stamp and image_iterator < len(images)-1):
            image_iterator += 1
        current_image = images[image_iterator]
        points = point_cloud2.read_points(lidar_pcl)
        j = 0
        for point in points:
            label = point[3]
            point_exp = (point[0], point[1], point[2], 1)
            tf_point_exp = numpy.dot(tf_cam_lidar, point_exp)
            tf_point = (tf_point_exp[0], tf_point_exp[1], tf_point_exp[2])
            print(point)
            print(tf_point)
            print(label)
            j += 1
            if j == 5:
                break



        i += 1
        if i == 5:
            break

        print('-------------')
    return


if __name__ == '__main__':
    main()
