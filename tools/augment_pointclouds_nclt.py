# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology
import argparse
import numpy as np
import os
import yaml
import struct
import sys
import tf.transformations as transformations
import rosbag
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# from NCLT scripts. ToDo(alaturn) Rather import original NCLT script as module.
def ssc_to_homo(ssc):

    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H


def main():
    parser = argparse.ArgumentParser(description="Augment Point Cloud")
    parser.add_argument('input_bag', metavar='input_bag', type=str, help='bag file with LiDAR, image, labels and TF')
    # parser.add_argument('output_bag', metavar='output_bag', type=str, help='bag file with augmented cloud')
    args = parser.parse_args()

    bridge = CvBridge()

    # ToDO(alaturn) Adapt this to the colors of nvidia and ids of SegMap
    # segmentation_id_color = {1:[42,174,203], 2:[224,172,177], 3:[145,183,160], 4:[137,241,224], 5:[132,224,232], 6:[105,64,153],
               # 7:[227,217,179], 8:[91,214,208], 9:[219,213,192], 10:[229,90,95], 11:[248,71,170], 12:[199,173,249],
               # 13:[205,228,85], 14:[208,160,121], 15:[180,238,141], 16:[53,246,59], 17:[50,96,227],
               # 18:[190,247,227], 19:[0,0,0], 31:[142,190,77], 32:[190,247,227], 33:[216,254,163], 34:[158,253,220]}
    # lookup_id_color = numpy.zeros((256, 256, 256))
    # for key, value in segmentation_id_color.items():
        # lookup_id_color[value[0], value[1], value[2]] = key

    in_bag = rosbag.Bag(args.input_bag)
    # out_bag = rosbag.Bag(args.output_bag, 'w')

    # ToDo(alaturn) Read these in from file. Just for LB3-Cam5 for now...
    # Intrinsics (given by NCLT).
    image_width = 1616 #u, x #646  
    image_height = 1232 #492
    f_x = 399.433184
    f_y = 399.433184
    c_x = 826.361952 #621.668624
    c_y = 621.668624 #826.361952
    camera_intrinsics = [[f_x, 0.0, c_x, 0.0], [
        0.0, f_y, c_y, 0.0], [0.0, 0.0, 1.0, 0.0]]

    # Scaled intrinsics (because the images on the bag are scaled down).
    scale_img = 0.4
    image_width_sc = int(scale_img*1616) #*1616) #646  
    image_height_sc = int(scale_img*1232) #*1232) #492
    f_x_sc = scale_img*f_x
    f_y_sc = scale_img*f_y
    c_x_sc = scale_img*c_x
    c_y_sc = scale_img*c_y
    camera_intrinsics_sc = [[f_x_sc, 0.0, c_x_sc, 0.0], [
        0.0, f_y_sc, c_y_sc, 0.0], [0.0, 0.0, 1.0, 0.0]]    

    # Extrinsics (given by NCLT).
    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50] # (x, y, z, phi, theta, psi).
    tf_body_lb3 = ssc_to_homo(x_body_lb3)    # a 4x4 homogenous transformation matrix.
    x_lb3_c5 = [0.041862, -0.001905, -0.000212, 160.868615, 89.9114152, 160.619894]
    tf_lb3_c5 = ssc_to_homo(x_lb3_c5)
    x_body_vel = [0.002, -0.004, -0.957, 0.807, 0.166, -90.703]
    tf_body_vel = ssc_to_homo(x_body_vel)

    tf_c5_vel = np.dot(np.linalg.inv(tf_lb3_c5), np.dot(np.linalg.inv(tf_body_lb3), tf_body_vel))
    tf_c5_body = np.linalg.inv(np.dot(tf_body_lb3, tf_lb3_c5))
    print('scale')
    print(camera_intrinsics_sc)
    print('tf_c5_body')
    print(tf_c5_body)

    # Print out the transform.
    print('tf_body_lb3')
    rpy = transformations.euler_from_matrix(tf_body_lb3, 'szyx')   
    rpy = np.array(rpy)
    print(rpy)

    # subsample_locations = numpy.linspace(50, image_height - 50, 64).astype(int)
    # lookup_subsample_locations = numpy.zeros(image_height)
    # lookup_subsample_locations[subsample_locations] = 1

    # Get images and semantic labels, together with timestamps.
    images = []
    img_ts = []
    for topic, image, t in in_bag.read_messages(topics='/images/raw5'):
        # ToDo(alaturn) Rotate back images.
        images.append(image)
        img_ts.append(t)    # Cause the images are not stamped. ToDo(alaturn) Fix inside nclt2ros.

    labels = []
    lab_ts = []
    for topic, label, t in in_bag.read_messages(topics='/images/prediction5'):
        labels.append(label)
        lab_ts.append(t)
    assert(len(img_ts)==len(lab_ts))

    image_iterator = 0
    for topic, lidar_pcl, t in in_bag.read_messages(topics=['/velodyne_points']):
        augmented_points = []
        # Forward search for getting img<->cloud correspondence. NCLT has already synced lidar and image...
        while(img_ts[image_iterator] < lidar_pcl.header.stamp and image_iterator < len(images)-1):
            image_iterator += 1
        current_image = images[image_iterator]
        current_label = labels[image_iterator]
        cv_image = bridge.imgmsg_to_cv2(current_image, desired_encoding='bgr8')
        cv_label = bridge.imgmsg_to_cv2(current_label, desired_encoding='bgr8')
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_label = cv2.rotate(cv_label, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Process pointcloud.
        points = point_cloud2.read_points(lidar_pcl)
        im_pts = np.array([0, 0, 0, 0, 0])
        for point in points:
            point_nclt = np.array(point[:3])
            point_nclt[1] = -point_nclt[1]  # Massive hack because nclt2ros rotates body to get base_link, but the given extrinsics are for body.
            point_nclt[2] = -point_nclt[2]
            # Convert into camera frame.
            point_hmg = np.append(np.array(point_nclt), [1])
            point_cam = np.dot(tf_c5_body, point_hmg)   # Note: nclt2ros stamps LiDAR in 'velodyne' frame, but actually they are still in base_link frame (vel_synced).

        #     # distance filters
        #     dist = point[0]*point[0] + point[1]*point[1] + point[2]*point[2]
        #     if dist > 2500 or dist < 6:
        #         continue

            # Project onto image.
            camera_point = np.dot(camera_intrinsics_sc, point_cam)
            image_coordinates = [
                camera_point[0] / camera_point[2],
                camera_point[1] / camera_point[2]]
            u = int(round(image_coordinates[0]))
            v = int(round(image_coordinates[1]))

            # Check if projection lies on image.
            if camera_point[2] > 0 and u > 0 and u < image_width_sc and v > 0 and v < image_height_sc:
                pt = np.array([u,v])

                # Get color at projected position.
                bgr = cv_label[v, u]
                pt = np.append(pt, bgr)
                # print(bgr.shape)
                # print(pt.shape)
                # print(im_pts.shape)
                im_pts = np.vstack((im_pts, pt))
                # print(im_pts.shape)
                # print(im_pts.shape)
                # Get 'label-color' at projected position.

                # Convert to SegMap label convention.

                # Create PointXYZRGBA (need to signswap y, z again to fit nclt2ros convention of 'base_link').

        # Draw on image.
        red = [0,0,255]
        for pt in im_pts:
            # print(pt[2:].shape)
            cv_image[pt[1], pt[0]] = pt[2:]
            cv_label[pt[1], pt[0]] = red

        # Show overlaid image.
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        cv_label = cv2.rotate(cv_label, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('lol2', cv_image)
        cv2.imshow('lol', cv_label)
        cv2.waitKey(5)

        # Create out cloud.
        # header = depth_cam_pcl.header
        # header.frame_id = '/airsim_drone'
        # augmented_cloud = point_cloud2.create_cloud(
        # header, fields, augmented_points)

        # out_bag.write('/augmented_cloud', augmented_cloud,
        #               augmented_cloud.header.stamp, False)



    #         if not lookup_subsample_locations[v]:
    #             continue

    #         projected_point = numpy.dot(
    #             tf_lidar_cam, (point[0], point[1], point[2], 1))

    #         rgb = bytearray(struct.pack("f", point[3]))
    #         rgb = struct.unpack('<i', str(rgb))[0]
    #         r = (rgb >> 16) & 0xff
    #         g = (rgb >> 8) & 0xff
    #         b = rgb & 0xff

    #         b_sem = current_image.data[3*(u + v * image_width)]
    #         b_sem = struct.unpack('B', str(b_sem))[0]
    #         g_sem = current_image.data[3*(u + v * image_width) + 1]
    #         g_sem = struct.unpack('B', str(g_sem))[0]
    #         r_sem = current_image.data[3*(u + v * image_width) + 2]
    #         r_sem = struct.unpack('B', str(r_sem))[0]

    #         label = lookup_id_color[b_sem, g_sem, r_sem]

    #         rgba = struct.unpack('I', struct.pack(
    #                     'BBBB', b, g, r, int(label) * 7))[0]
    #         augmented_points.append(
    #             [projected_point[0], projected_point[1], projected_point[2], rgba])


    #     fields = [PointField('x', 0, PointField.FLOAT32, 1),
    #               PointField('y', 4, PointField.FLOAT32, 1),
    #               PointField('z', 8, PointField.FLOAT32, 1),
    #               PointField('rgba', 12, PointField.UINT32, 1)]

    #     header = depth_cam_pcl.header
    #     header.frame_id = '/airsim_drone'
    #     augmented_cloud = point_cloud2.create_cloud(
    #         header, fields, augmented_points)

    #     out_bag.write('/augmented_cloud', augmented_cloud,
    #                   augmented_cloud.header.stamp, False)

    #     i += 1
    #     if i == 200:
    #         break
    #     print('Pointcloud: ' + str(i))

    # for topic, tf, t in bag.read_messages(topics=['/tf']):
    #     out_bag.write('/tf', tf, tf.transforms[0].header.stamp, False)

    # out_bag.close()

if __name__ == '__main__':
    main()
