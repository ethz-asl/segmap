# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology
import numpy
import os
import yaml
import struct
import tf.transformations as transformations
import rosbag
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    # with open('/home/marius/segmap_ws/src/segmap/segmapper/launch/bosch/segmentation_id_color.yaml', 'r') as stream:
    with open('/home/mariusbr/segmap_ws/src/segmap/segmapper/launch/bosch/segmentation_id_color.yaml', 'r') as stream:
        segmentation_id_color = yaml.load(stream)
    # bag_file = '/home/marius/.segmap/bosch/bosch.bag'
    bag_file = '/media/scratch1/mariusbr/bosch.bag'
    # out_bag_file = '/home/marius/.segmap/bosch/augmented_bosch.bag'
    out_bag_file = '/media/scratch1/mariusbr/augmented_bosch.bag'
    bag = rosbag.Bag(bag_file)
    out_bag = rosbag.Bag(out_bag_file, 'w')
    image_width = 640
    image_height = 480
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

    tf_lidar_drone = transformations.inverse_matrix(tf_drone_lidar)
    tf_lidar_cam = numpy.dot(tf_lidar_drone, tf_drone_cam)

    images = []

    i = 0

    for topic, image, t in bag.read_messages(topics='/airsim_drone/Seg_cam'):
        images.append(image)
        i += 1
        print('Image: ' + str(i))

    image_iterator = 0
    i = 0
    for topic, depth_cam_pcl, t in bag.read_messages(topics=['/airsim_drone/RGBD_cam']):
        augmented_points = []
        while(images[image_iterator].header.stamp < depth_cam_pcl.header.stamp and image_iterator < len(images)-1):
            image_iterator += 1
        current_image = images[image_iterator]
        points = point_cloud2.read_points(depth_cam_pcl)
        for point in points:
            j += 1
            projected_point = numpy.dot(
                tf_lidar_cam, (point[0], point[1], point[2], 1))

            rgb = bytearray(struct.pack("f", point[3]))
            rgb = struct.unpack('<i', str(rgb))[0]
            r = (rgb >> 16) & 0xff
            g = (rgb >> 8) & 0xff
            b = rgb & 0xff

            camera_point = numpy.dot(camera_intrinsics, point)
            image_coordinates = [
                camera_point[0] / camera_point[2], camera_point[1] / camera_point[2]]
            u = int(round(image_coordinates[0]))
            v = int(round(image_coordinates[1]))
            b_sem = current_image.data[3*(u + v * image_width)]
            b_sem = struct.unpack('B', str(b_sem))[0]
            g_sem = current_image.data[3*(u + v * image_width) + 1]
            g_sem = struct.unpack('B', str(g_sem))[0]
            r_sem = current_image.data[3*(u + v * image_width) + 2]
            r_sem = struct.unpack('B', str(r_sem))[0]
            label = 0
            for key, value in segmentation_id_color['dict_class_color'].items():
                if [b_sem, g_sem, r_sem] == value:
                    label = key

            rgba = struct.unpack('I', struct.pack(
                        'BBBB', b, g, r, int(label) * 7))[0]
            augmented_points.append(
                [projected_point[0], projected_point[1], projected_point[2], rgba])


        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1),
                  ]
        header = depth_cam_pcl.header
        header.frame_id = 'airsim_drone/Lidar'
        augmented_cloud = point_cloud2.create_cloud(
            header, fields, augmented_points)

        out_bag.write('/augmented_cloud', augmented_cloud,
                      augmented_cloud.header.stamp, False)

        i += 1
        print('Pointcloud: ' + str(i))
    i = 0
    for topic, tf, t in bag.read_messages(topics=['/tf']):
        out_bag.write('/tf', tf, tf.transforms[0].header.stamp, False)
        i += 1
        if i == 1:
            time_hack = tf.transforms[0].header.stamp
        print('TF: ' + str(i))

    for topic, tf_static, t in bag.read_messages(topics=['/tf_static']):
        out_bag.write('/tf_static', tf_static,
                      time_hack, False)
    out_bag.close()


if __name__ == '__main__':
    main()
