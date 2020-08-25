# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology
import numpy
import os
import yaml
import struct
import tf.transformations as transformations
import rosbag
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():

    # bag_file = '/home/marius/.segmap/bosch/bosch.bag'
    bag_file = '/media/scratch1/mariusbr/bosch.bag'
    # out_bag_file = '/home/marius/.segmap/bosch/augmented_bosch.bag'
    out_bag_file = '/media/scratch1/mariusbr/augmented_bosch.bag'
    bag = rosbag.Bag(bag_file)
    out_bag = rosbag.Bag(out_bag_file, 'w')
    image_width = 640.0
    image_height = 480.0
    f_x = image_width / 2.0
    f_y = image_width / 2.0
    c_x = image_width / 2.0
    c_y = image_height / 2.0
    camera_intrinsics = [[f_x, 0.0, c_x, 0.0], [
        0.0, f_y, c_y, 0.0], [0.0, 0.0, 1.0, 0.0]]

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

    for topic, image, t in bag.read_messages(topics='/airsim_drone/Scene_cam'):

        images.append(image)

        i += 1
        print('Image: ' + str(i))

    image_iterator = 0
    i = 0
    for topic, lidar_pcl, t in bag.read_messages(topics=['/airsim_drone/Lidar']):

        augmented_points = []

        while(images[image_iterator].header.stamp < lidar_pcl.header.stamp and image_iterator < len(images)-1):
            image_iterator += 1
        current_image = images[image_iterator]
        points = point_cloud2.read_points(lidar_pcl)
        j = 0
        for point in points:
            label = point[3]
            projected_point = numpy.dot(
                tf_cam_lidar, (point[0], point[1], point[2], 1))
            r = 255
            g = 255
            b = 0

            if (projected_point[2] > 0):
                camera_point = numpy.dot(camera_intrinsics, projected_point)
                image_coordinates = [
                    camera_point[0] / camera_point[2], camera_point[1] / camera_point[2]]

                if image_coordinates[0] > 0 and image_coordinates[0] < image_width - 1:
                    if image_coordinates[1] > 0 and image_coordinates[1] < image_height - 1:
                        u = int(round(image_coordinates[0]))
                        v = int(round(image_coordinates[1]))
                        bridge = CvBridge()
                        cv_image = bridge.imgmsg_to_cv2(
                            current_image, desired_encoding='passthrough')
                        b = cv_image[v, u, 0]
                        g = cv_image[v, u, 1]
                        r = cv_image[v, u, 2]

            rgba = struct.unpack('I', struct.pack(
                'BBBB', b, g, r, int(label) + 200))[0]
            augmented_points.append([point[0], point[1], point[2], rgba])

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1),
                  ]
        header = lidar_pcl.header
        augmented_cloud = point_cloud2.create_cloud(
            header, fields, augmented_points)

        out_bag.write('/augmented_cloud', augmented_cloud,
                      augmented_cloud.header.stamp, False)
        out_bag.write('/airsim_drone/Lidar', lidar_pcl,
                      lidar_pcl.header.stamp, False)

        i += 1
        print('Pointcloud: ' + str(i))
    i = 0
    for topic, tf, t in bag.read_messages(topics=['/tf']):
        out_bag.write('/tf', tf, tf.transforms[0].header.stamp, False)
        i += 1
        print('TF: ' + str(i))

    for topic, tf_static, t in bag.read_messages(topics=['/tf_static']):
        out_bag.write('/tf_static', tf_static,
                      tf_static.transforms[0].header.stamp, False)

    i = 0
    for topic, msg, t in bag.read_messages(topics=['/airsim_drone/RGBD_cam']):
        out_bag.write('/airsim_drone/RGBD_cam', msg, msg.header.stamp, False)
        i += 1
        print('Depth Cam: ' + str(i))

    out_bag.close()


if __name__ == '__main__':
    main()
