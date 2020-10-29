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
    segmentation_id_color = {1:[42,174,203], 2:[224,172,177], 3:[145,183,160], 4:[137,241,224], 5:[132,224,232], 6:[105,64,153],
               7:[227,217,179], 8:[91,214,208], 9:[219,213,192], 10:[229,90,95], 11:[248,71,170], 12:[199,173,249],
               13:[205,228,85], 14:[208,160,121], 15:[180,238,141], 16:[53,246,59], 17:[50,96,227],
               18:[190,247,227], 19:[0,0,0], 31:[142,190,77], 32:[190,247,227], 33:[216,254,163], 34:[158,253,220]}
    lookup_id_color = numpy.zeros((256, 256, 256))
    for key, value in segmentation_id_color.items():
        lookup_id_color[value[0], value[1], value[2]] = key

    bag_file = '/mnt/scratch1/bosch/2020-10-14/base-map.bag'
    out_bag_file = '/mnt/scratch1/bosch/2020-10-14/base-map-augmented.bag'

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
    subsample_locations = numpy.linspace(50, image_height - 50, 64).astype(int)
    lookup_subsample_locations = numpy.zeros(image_height)
    lookup_subsample_locations[subsample_locations] = 1


    tf_lidar_cam = transformations.quaternion_matrix(numpy.array(
        [0.5, -0.5, 0.5, -0.5]))
    tf_lidar_cam[0, 3] = 0.0
    tf_lidar_cam[1, 3] = 0.0
    tf_lidar_cam[2, 3] = 0.0

    images = []
    for topic, image, t in bag.read_messages(topics='/airsim_drone/Seg_cam'):
        images.append(image)

    image_iterator = 0
    i = 0
    for topic, depth_cam_pcl, t in bag.read_messages(topics=['/airsim_drone/RGBD_cam']):
        augmented_points = []

        while(images[image_iterator].header.stamp < depth_cam_pcl.header.stamp and image_iterator < len(images)-1):
            image_iterator += 1
        current_image = images[image_iterator]

        points = point_cloud2.read_points(depth_cam_pcl)
        for point in points:
            # distance filters
            dist = point[0]*point[0] + point[1]*point[1] + point[2]*point[2]
            if dist > 2500 or dist < 6:
                continue

            camera_point = numpy.dot(camera_intrinsics, point)
            image_coordinates = [
                camera_point[0] / camera_point[2],
                camera_point[1] / camera_point[2]]
            u = int(round(image_coordinates[0]))
            v = int(round(image_coordinates[1]))

            if not lookup_subsample_locations[v]:
                continue

            projected_point = numpy.dot(
                tf_lidar_cam, (point[0], point[1], point[2], 1))

            rgb = bytearray(struct.pack("f", point[3]))
            rgb = struct.unpack('<i', str(rgb))[0]
            r = (rgb >> 16) & 0xff
            g = (rgb >> 8) & 0xff
            b = rgb & 0xff

            b_sem = current_image.data[3*(u + v * image_width)]
            b_sem = struct.unpack('B', str(b_sem))[0]
            g_sem = current_image.data[3*(u + v * image_width) + 1]
            g_sem = struct.unpack('B', str(g_sem))[0]
            r_sem = current_image.data[3*(u + v * image_width) + 2]
            r_sem = struct.unpack('B', str(r_sem))[0]

            label = lookup_id_color[b_sem, g_sem, r_sem]

            rgba = struct.unpack('I', struct.pack(
                        'BBBB', b, g, r, int(label) * 7))[0]
            augmented_points.append(
                [projected_point[0], projected_point[1], projected_point[2], rgba])


        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1)]

        header = depth_cam_pcl.header
        header.frame_id = '/airsim_drone'
        augmented_cloud = point_cloud2.create_cloud(
            header, fields, augmented_points)

        out_bag.write('/augmented_cloud', augmented_cloud,
                      augmented_cloud.header.stamp, False)

        i += 1
        if i == 200:
            break
        print('Pointcloud: ' + str(i))

    for topic, tf, t in bag.read_messages(topics=['/tf']):
        out_bag.write('/tf', tf, tf.transforms[0].header.stamp, False)

    out_bag.close()

if __name__ == '__main__':
    main()
