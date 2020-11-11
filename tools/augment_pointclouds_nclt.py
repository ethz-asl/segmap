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
    parser.add_argument('output_bag', metavar='output_bag', type=str, help='bag file with augmented cloud')
    parser.add_argument('camera_param_dir', metavar='cam_par_dir', type=str, help='directory with camera intrinsics/extrinsics')
    args = parser.parse_args()

    bridge = CvBridge()

    # LookUp BGR (cityscapes) -> ID (segmap).
    segmentation_id_color = {
        33:[128,64,128],      # Road -> Road
        33:[232,35,244],      # Sidewalk -> Road
        34:[70,70,70],       # Building -> Small house
        34:[156,102,102],    # Wall -> Small house
        5:[153,153,190],     # Fence -> Fence
        10:[153,153,153],    # Pole -> LightGenerator
        19:[30,170,250],     # Traffic Light -> SignBoard
        19:[0,220,220],      # Traffic Sign -> SignBoard
        2:[35,142,107],      # Vegetation -> Tree
        32:[152,251,152],    # Terrain -> Landscape
        31:[180,130,70],     # Sky -> Sky
        12:[60,20,220],      # Person -> Portapotty 
        12:[0,0,255],        # Rider -> Portapotty
        1:[142,0,0],         # Car -> Car
        7:[70,0,0],          # Truck -> Truck
        7:[100,60,0],        # Bus -> Truck
        7:[100,80,0],        # Train -> Truck
        3:[230,0,0],         # Motorcycle -> Bench
        3:[32,11,119]       # Bicycle -> Bench 
    }
    lookup_id_color = np.zeros((256, 256, 256))
    for key, value in segmentation_id_color.items():
        lookup_id_color[value[0], value[1], value[2]] = key

    in_bag = rosbag.Bag(args.input_bag)
    out_bag = rosbag.Bag(args.output_bag, 'w')

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
    # f_x_sc = scale_img*f_x
    # f_y_sc = scale_img*f_y
    # c_x_sc = scale_img*c_x
    # c_y_sc = scale_img*c_y
    # camera_intrinsics_sc = [[f_x_sc, 0.0, c_x_sc, 0.0], [
    #     0.0, f_y_sc, c_y_sc, 0.0], [0.0, 0.0, 1.0, 0.0]]    

    # Extrinsics (given by NCLT).
    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50] # (x, y, z, phi, theta, psi).
    tf_body_lb3 = ssc_to_homo(x_body_lb3)    # a 4x4 homogenous transformation matrix.
    # x_lb3_c5 = [0.041862, -0.001905, -0.000212, 160.868615, 89.9114152, 160.619894]
    # tf_lb3_c5 = ssc_to_homo(x_lb3_c5)
    # x_body_vel = [0.002, -0.004, -0.957, 0.807, 0.166, -90.703]
    # tf_body_vel = ssc_to_homo(x_body_vel)

    # tf_c5_vel = np.dot(np.linalg.inv(tf_lb3_c5), np.dot(np.linalg.inv(tf_body_lb3), tf_body_vel))
    # tf_c5_body = np.linalg.inv(np.dot(tf_body_lb3, tf_lb3_c5))
    # print('scale')
    # print(camera_intrinsics_sc)
    # print('tf_c5_body')
    # print(tf_c5_body)

    # # Print out the transform.
    # print('tf_body_lb3')
    # rpy = transformations.euler_from_matrix(tf_body_lb3, 'szyx')   
    # rpy = np.array(rpy)
    # print(rpy)
    #######################################################################
    K_cam1 = scale_img*np.loadtxt(camera_param_dir + 'K_cam1.csv', delimiter=',')
    K_cam2 = scale_img*np.loadtxt(camera_param_dir + 'K_cam2.csv', delimiter=',')
    K_cam3 = scale_img*np.loadtxt(camera_param_dir + 'K_cam3.csv', delimiter=',')
    K_cam4 = scale_img*np.loadtxt(camera_param_dir + 'K_cam4.csv', delimiter=',')
    K_cam5 = scale_img*np.loadtxt(camera_param_dir + 'K_cam5.csv', delimiter=',')
    K_cam1[2,2] = 1.0
    K_cam2[2,2] = 1.0
    K_cam3[2,2] = 1.0
    K_cam4[2,2] = 1.0
    K_cam5[2,2] = 1.0

    x_lb3_c1 = np.loadtxt(camera_param_dir + 'x_lb3_c1.csv', delimiter=',')
    x_lb3_c2 = np.loadtxt(camera_param_dir + 'x_lb3_c2.csv', delimiter=',')
    x_lb3_c3 = np.loadtxt(camera_param_dir + 'x_lb3_c3.csv', delimiter=',')
    x_lb3_c4 = np.loadtxt(camera_param_dir + 'x_lb3_c4.csv', delimiter=',')
    x_lb3_c5 = np.loadtxt(camera_param_dir + 'x_lb3_c5.csv', delimiter=',')

    tf_lb3_c1 = ssc_to_homo(x_lb3_c1)
    tf_lb3_c2 = ssc_to_homo(x_lb3_c2)
    tf_lb3_c3 = ssc_to_homo(x_lb3_c3)
    tf_lb3_c4 = ssc_to_homo(x_lb3_c4)
    tf_lb3_c5 = ssc_to_homo(x_lb3_c5)

    tf_c1_body = np.linalg.inv(np.dot(tf_body_lb3, tf_lb3_c1))
    tf_c2_body = np.linalg.inv(np.dot(tf_body_lb3, tf_lb3_c2))
    tf_c3_body = np.linalg.inv(np.dot(tf_body_lb3, tf_lb3_c3))
    tf_c4_body = np.linalg.inv(np.dot(tf_body_lb3, tf_lb3_c4))
    tf_c5_body = np.linalg.inv(np.dot(tf_body_lb3, tf_lb3_c5))
    ########################################################################

    # subsample_locations = numpy.linspace(50, image_height - 50, 64).astype(int)
    # lookup_subsample_locations = numpy.zeros(image_height)
    # lookup_subsample_locations[subsample_locations] = 1

    # Get images and semantic labels, together with timestamps.
    images1 = []
    img_ts = []
    for topic, image, t in in_bag.read_messages(topics='/images/raw1'):
        images1.append(image)
        img_ts.append(t)    # Cause the images are not stamped. ToDo(alaturn) Fix inside nclt2ros.
    images2 = []
    for topic, image, t in in_bag.read_messages(topics='/images/raw2'):
        images2.append(image)
    images3 = []
    for topic, image, t in in_bag.read_messages(topics='/images/raw3'):
        images3.append(image)
    images4 = []
    for topic, image, t in in_bag.read_messages(topics='/images/raw4'):
        images4.append(image)
    images5 = []
    for topic, image, t in in_bag.read_messages(topics='/images/raw5'):
        images5.append(image)

    # labels = []
    # lab_ts = []
    # for topic, label, t in in_bag.read_messages(topics='/images/prediction5'):
    #     labels.append(label)
    #     lab_ts.append(t)
    # assert(len(img_ts)==len(lab_ts))
    labels1 = []
    for topic, label, t in in_bag.read_messages(topics='/images/prediction1'):
        labels1.append(label)
    labels2 = []
    for topic, label, t in in_bag.read_messages(topics='/images/prediction2'):
        labels2.append(label)
    labels3 = []
    for topic, label, t in in_bag.read_messages(topics='/images/prediction3'):
        labels3.append(label)
    labels4 = []
    for topic, label, t in in_bag.read_messages(topics='/images/prediction4'):
        labels4.append(label)
    labels5 = []
    for topic, label, t in in_bag.read_messages(topics='/images/prediction5'):
        labels5.append(label)

    image_iterator = 0
    for topic, lidar_pcl, t in in_bag.read_messages(topics=['/velodyne_points']):
        augmented_points = []
        # Forward search for getting img<->cloud correspondence. NCLT has already synced lidar and image...
        while(img_ts[image_iterator] < lidar_pcl.header.stamp and image_iterator < len(images)-1):
            image_iterator += 1
        current_image1 = images1[image_iterator]
        current_image2 = images2[image_iterator]
        current_image3 = images3[image_iterator]
        current_image4 = images4[image_iterator]
        current_image5 = images5[image_iterator]

        current_label1 = labels1[image_iterator]
        current_label2 = labels2[image_iterator]
        current_label3 = labels3[image_iterator]
        current_label4 = labels4[image_iterator]
        current_label5 = labels5[image_iterator]

        cv_image1 = bridge.imgmsg_to_cv2(current_image1, desired_encoding='bgr8')
        cv_image2 = bridge.imgmsg_to_cv2(current_image2, desired_encoding='bgr8')
        cv_image3 = bridge.imgmsg_to_cv2(current_image3, desired_encoding='bgr8')
        cv_image4 = bridge.imgmsg_to_cv2(current_image4, desired_encoding='bgr8')
        cv_image5 = bridge.imgmsg_to_cv2(current_image5, desired_encoding='bgr8')
        
        cv_label1 = bridge.imgmsg_to_cv2(current_label1, desired_encoding='bgr8')
        cv_label2 = bridge.imgmsg_to_cv2(current_label2, desired_encoding='bgr8')
        cv_label3 = bridge.imgmsg_to_cv2(current_label3, desired_encoding='bgr8')
        cv_label4 = bridge.imgmsg_to_cv2(current_label4, desired_encoding='bgr8')
        cv_label5 = bridge.imgmsg_to_cv2(current_label5, desired_encoding='bgr8')
        
        cv_image1 = cv2.rotate(cv_image1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_image2 = cv2.rotate(cv_image2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_image3 = cv2.rotate(cv_image3, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_image4 = cv2.rotate(cv_image4, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_image5 = cv2.rotate(cv_image5, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv_label1 = cv2.rotate(cv_label1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_label2 = cv2.rotate(cv_label2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_label3 = cv2.rotate(cv_label3, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_label4 = cv2.rotate(cv_label4, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv_label5 = cv2.rotate(cv_label5, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Process pointcloud.
        points = point_cloud2.read_points(lidar_pcl)
        im_pts = np.array([0, 0, 0, 0, 0])
        for point in points:
        #     # distance filters
        #     dist = point[0]*point[0] + point[1]*point[1] + point[2]*point[2]
        #     if dist > 2500 or dist < 6:
        #         continue
            point_nclt = np.array(point[:3])
            point_nclt[1] = -point_nclt[1]  # Massive hack because nclt2ros rotates body to get base_link, but the given extrinsics are for body.
            point_nclt[2] = -point_nclt[2]
            
            # Convert into camera frame.
            point_hmg = np.append(np.array(point_nclt), [1])
            # point_cam = np.dot(tf_c5_body, point_hmg)   # Note: nclt2ros stamps LiDAR in 'velodyne' frame, but actually they are still in base_link frame (vel_synced).
            point_c1 = np.dot(tf_c1_body, point_hmg)
            point_c2 = np.dot(tf_c2_body, point_hmg)
            point_c3 = np.dot(tf_c3_body, point_hmg)
            point_c4 = np.dot(tf_c4_body, point_hmg)
            point_c5 = np.dot(tf_c5_body, point_hmg)
            
            # Project onto image.
            # camera_point = np.dot(camera_intrinsics_sc, point_cam)
            # image_coordinates = [
                # camera_point[0] / camera_point[2],
                # camera_point[1] / camera_point[2]]
            # u = int(round(image_coordinates[0]))
            # v = int(round(image_coordinates[1]))
            camera1_point = np.dot(K_cam1, point_c1)
            camera2_point = np.dot(K_cam2, point_c2)
            camera3_point = np.dot(K_cam3, point_c3)
            camera4_point = np.dot(K_cam4, point_c4)
            camera5_point = np.dot(K_cam5, point_c5)

            im_coordinates1 = [
                camera1_point[0] / camera1_point[2],
                camera1_point[1] / camera1_point[2]]
            im_coordinates2 = [
                camera2_point[0] / camera2_point[2],
                camera2_point[1] / camera2_point[2]]
            im_coordinates3 = [
                camera3_point[0] / camera3_point[2],
                camera3_point[1] / camera3_point[2]]
            im_coordinates4 = [
                camera4_point[0] / camera4_point[2],
                camera4_point[1] / camera4_point[2]]
            im_coordinates5 = [
                camera5_point[0] / camera5_point[2],
                camera5_point[1] / camera5_point[2]]

            u1 = int(round(im_coordinates1[0]))
            v1 = int(round(im_coordinates1[1]))
            u2 = int(round(im_coordinates2[0]))
            v2 = int(round(im_coordinates2[1]))
            u3 = int(round(im_coordinates3[0]))
            v3 = int(round(im_coordinates3[1]))
            u4 = int(round(im_coordinates4[0]))
            v4 = int(round(im_coordinates4[1]))
            u5 = int(round(im_coordinates5[0]))
            v5 = int(round(im_coordinates5[1]))

            # Check if projection lies on image.
            bgr = []
            bgr_sem = []
            if im_coordinates1[2] > 0 and u1 > 0 and u1 < image_width_sc and v1 > 0 and v1 < image_height_sc:
                # Image 1.
                bgr = cv_image1[v1, u1]
                bgr_sem = cv_label1[v1, u1]
            else if im_coordinates2[2] > 0 and u2 > 0 and u2 < image_width_sc and v2 > 0 and v2 < image_height_sc:
                # Image 2.
                bgr = cv_image2[v2, u2]
                bgr_sem = cv_label2[v2, u2]
            else if im_coordinates3[2] > 0 and u3 > 0 and u3 < image_width_sc and v3 > 0 and v3 < image_height_sc:
                # Image 3.
                bgr = cv_image3[v3, u3]
                bgr_sem = cv_label3[v3, u3]
            else if im_coordinates4[2] > 0 and u4 > 0 and u4 < image_width_sc and v4 > 0 and v4 < image_height_sc:
                # Image 4.
                bgr = cv_image4[v4, u4]
                bgr_sem = cv_label4[v4, u4]
            else if im_coordinates5[2] > 0 and u5 > 0 and u5 < image_width_sc and v5 > 0 and v5 < image_height_sc:
                # Image 5.
                bgr = cv_image5[v5, u5]
                bgr_sem = cv_label5[v5, u5]
            else:
                continue

            # Create PointXYZRGBA
            label = lookup_id_color[bgr_sem[0], bgr_sem[1], bgr_sem[2]]
            rgba = struct.unpack('I', struct.pack(
                    'BBBB', bgr[0], bgr[1], bgr[2], int(label) * 7))[0]
            aug_pt = [float(point_nclt[0]), float(-point_nclt[1]), float(-point_nclt[2]), rgba]
            augmented_points.append(aug_pt)

            # if camera_point[2] > 0 and u > 0 and u < image_width_sc and v > 0 and v < image_height_sc:
            #     pt = np.array([u,v])

            #     # Get color and label 'color' at projected position.
            #     bgr = cv_image[v, u]
            #     bgr_sem = cv_label[v, u]

            #     # For viz
            #     pt = np.append(pt, bgr_sem)
            #     im_pts = np.vstack((im_pts, pt))               

            #     # Create PointXYZRGBA (need to signswap y, z again to fit nclt2ros convention of 'base_link').
            #     label = lookup_id_color[bgr_sem[0], bgr_sem[1], bgr_sem[2]]
            #     rgba = struct.unpack('I', struct.pack(
            #             'BBBB', bgr[0], bgr[1], bgr[2], int(label) * 7))[0]
            #     aug_pt = [float(point_nclt[0]), float(-point_nclt[1]), float(-point_nclt[2]), rgba]
            #     augmented_points.append(aug_pt)

        # Draw on image.
        # red = [0,0,255]
        # for pt in im_pts:
        #     # print(pt[2:].shape)
        #     cv_image[pt[1], pt[0]] = pt[2:]
        #     cv_label[pt[1], pt[0]] = red

        # # Show overlaid image.
        # cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        # cv_label = cv2.rotate(cv_label, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imshow('cv_image', cv_image)
        # cv2.imshow('cv_label', cv_label)
        # cv2.waitKey(1)

        # Create augmented cloud.
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1)]
        header = lidar_pcl.header
        header.frame_id = '/base_link'  #'/airsim_drone'    # ToDo(alaturn) Use naming from BOSCH
        augmented_cloud = point_cloud2.create_cloud(header, fields, augmented_points)

        out_bag.write('/augmented_cloud', augmented_cloud,
                      augmented_cloud.header.stamp, False)

        print('Wrote cloud!')

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

    out_bag.close()

if __name__ == '__main__':
    main()
