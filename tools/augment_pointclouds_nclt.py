# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology
import argparse
import numpy as np
import os
import yaml
import struct
import sys
import tf.transformations as transformations
import time
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
    parser.add_argument('scale', metavar='scale', type=float, help='how much images were scaled down from raw NCLT dimensions')
    args = parser.parse_args()

    bridge = CvBridge()

    lookup_id_color = np.zeros((256, 256, 256))
    lookup_id_color[0,0,0] = 0              # Undefined   
    lookup_id_color[128,64,128] = 33        # Road -> Road
    lookup_id_color[232,35,244] = 33        # Sidewalk -> Road
    lookup_id_color[70,70,70] = 34          # Building -> Small house
    lookup_id_color[156,102,102] = 34       # Wall -> Small house
    lookup_id_color[153,153,190] = 5        # Fence -> Fence
    lookup_id_color[153,153,153] = 10       # Pole -> LightGenerator
    lookup_id_color[30,170,250] = 19        # Traffic Light -> SignBoard
    lookup_id_color[0,220,220] = 19         # Traffic Sign -> SignBoard
    lookup_id_color[35,142,107] =  2        # Vegetation -> Tree
    lookup_id_color[152,251,152] = 32       # Terrain -> Landscape
    lookup_id_color[180,130,70] = 31        # Sky -> Sky
    lookup_id_color[60,20,220] = 12         # Person -> Portapotty 
    lookup_id_color[0,0,255] = 12           # Rider -> Portapotty
    lookup_id_color[142,0,0] = 1            # Car -> Car
    lookup_id_color[70,0,0] = 7             # Truck -> Truck
    lookup_id_color[100,60,0] = 7           # Bus -> Truck
    lookup_id_color[100,80,0] = 7           # Train -> Truck
    lookup_id_color[230,0,0] = 3            # Motorcycle -> Bench
    lookup_id_color[32,11,119] = 3          # Bicycle -> Bench 

    in_bag = rosbag.Bag(args.input_bag)
    out_bag = rosbag.Bag(args.output_bag, 'w')

    # Re-Write TFs.
    for topic, tf, t in in_bag.read_messages(topics=['/tf', '/tf_static']):
        out_bag.write('/tf', tf, tf.transforms[0].header.stamp, False)  
    print('Wrote all TFs!')
    
    # Intrinsics (given by NCLT).
    image_width = 1616
    image_height = 1232

    # Scaled intrinsics (because the images on the bag are scaled down during postprocessing).
    scale_img = args.scale #0.4
    image_width_sc = int(scale_img*1616) #*1616) #646  
    image_height_sc = int(scale_img*1232) #*1232) #492

    # Extrinsics (given by NCLT).
    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50] # (x, y, z, phi, theta, psi).
    tf_body_lb3 = ssc_to_homo(x_body_lb3)    # a 4x4 homogenous transformation matrix.

    #######################################################################
    K_cam1 = scale_img*np.loadtxt(args.camera_param_dir + 'K_cam1.csv', delimiter=',')
    K_cam2 = scale_img*np.loadtxt(args.camera_param_dir + 'K_cam2.csv', delimiter=',')
    K_cam3 = scale_img*np.loadtxt(args.camera_param_dir + 'K_cam3.csv', delimiter=',')
    K_cam4 = scale_img*np.loadtxt(args.camera_param_dir + 'K_cam4.csv', delimiter=',')
    K_cam5 = scale_img*np.loadtxt(args.camera_param_dir + 'K_cam5.csv', delimiter=',')
    K_cam1[2,2] = 1.0
    K_cam2[2,2] = 1.0
    K_cam3[2,2] = 1.0
    K_cam4[2,2] = 1.0
    K_cam5[2,2] = 1.0

    x_lb3_c1 = np.loadtxt(args.camera_param_dir + 'x_lb3_c1.csv', delimiter=',')
    x_lb3_c2 = np.loadtxt(args.camera_param_dir + 'x_lb3_c2.csv', delimiter=',')
    x_lb3_c3 = np.loadtxt(args.camera_param_dir + 'x_lb3_c3.csv', delimiter=',')
    x_lb3_c4 = np.loadtxt(args.camera_param_dir + 'x_lb3_c4.csv', delimiter=',')
    x_lb3_c5 = np.loadtxt(args.camera_param_dir + 'x_lb3_c5.csv', delimiter=',')

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

    # The actual image boundaries (zero padding during postprocessing) -> Applies to the unrotated image BEFORE scaling down!!!.
    xc, yc = 804, 617
    height_crop, width_crop = 800, 1238
    top = int(yc-height_crop/2)
    bot = int(yc+height_crop/2)
    left = int(xc-width_crop/2)
    right = int(xc+width_crop/2)
    top_sc = scale_img*top
    bot_sc = bot*scale_img
    left_sc = left*scale_img
    right_sc = right*scale_img

    # Generators
    img1_gen = in_bag.read_messages(topics='/images/raw1')
    img2_gen = in_bag.read_messages(topics='/images/raw2')
    img3_gen = in_bag.read_messages(topics='/images/raw3')
    img4_gen = in_bag.read_messages(topics='/images/raw4')
    img5_gen = in_bag.read_messages(topics='/images/raw5')
    lab1_gen = in_bag.read_messages(topics='/images/prediction1')
    lab2_gen = in_bag.read_messages(topics='/images/prediction2')
    lab3_gen = in_bag.read_messages(topics='/images/prediction3')
    lab4_gen = in_bag.read_messages(topics='/images/prediction4')
    lab5_gen = in_bag.read_messages(topics='/images/prediction5')

    topic, im1, ti1 = img1_gen.next()
    topic, im2, ti2 = img2_gen.next()
    topic, im3, ti3 = img3_gen.next()
    topic, im4, ti4 = img4_gen.next()
    topic, im5, ti5 = img5_gen.next()
    topic, lab1, tl1 = lab1_gen.next()
    topic, lab2, tl2 = lab2_gen.next()
    topic, lab3, tl3 = lab3_gen.next()
    topic, lab4, tl4 = lab4_gen.next()
    topic, lab5, tl5 = lab5_gen.next()

    first_sync = True
    skip_img_incr = False
    i=0
    skip_pcl1 = 0
    skip_pcl2 = 0
    start_t = 0
    time_st = time.time()
    for topic, lidar_pcl, t in in_bag.read_messages(topics=['/velodyne_points']):
        augmented_points = []

        # Sync up lidar and images: Assume lidar started earlier.
        if not(t==ti1) and first_sync:
            if t < ti1:
                # LiDAR before images.
                skip_pcl1+=1
                print('Bad luck, try next pcl msg!')
                print(t)
                print(ti1)
                print(skip_pcl1)
                continue
            if t > ti1:
                # Images before LiDAR.
                while not (t==ti1):
                    topic, im1, ti1 = img1_gen.next()
                    topic, im2, ti2 = img2_gen.next()
                    topic, im3, ti3 = img3_gen.next()
                    topic, im4, ti4 = img4_gen.next()
                    topic, im5, ti5 = img5_gen.next()
                    topic, lab1, tl1 = lab1_gen.next()
                    topic, lab2, tl2 = lab2_gen.next()
                    topic, lab3, tl3 = lab3_gen.next()
                    topic, lab4, tl4 = lab4_gen.next()
                    topic, lab5, tl5 = lab5_gen.next()
    
        # Now, should be synced and should be able to go to normal mode...
        if first_sync:
            first_sync = False
            start_t = t
            time_st = time.time()
            print('In-Sync!')
        else:
            if not skip_img_incr:
                try:
                    topic, im1, ti1 = img1_gen.next()
                    topic, im2, ti2 = img2_gen.next()
                    topic, im3, ti3 = img3_gen.next()
                    topic, im4, ti4 = img4_gen.next()
                    topic, im5, ti5 = img5_gen.next()
                    topic, lab1, tl1 = lab1_gen.next()
                    topic, lab2, tl2 = lab2_gen.next()
                    topic, lab3, tl3 = lab3_gen.next()
                    topic, lab4, tl4 = lab4_gen.next()
                    topic, lab5, tl5 = lab5_gen.next()
                except StopIteration:
                    print('No images left!')
                    break

        # Very rarely, there is an image without a corresponding point cloud! Assume next one will be synced again.
        if not(t==ti1):
            print('Out of Sync again, very weird...')
            if(t<ti1):
                print('PCL lagging behind')
                skip_img_incr = True # Will increase PCL without increasing image.
                skip_pcl2+=1
                continue
            elif(t>ti1):
                print('Image lagging behind!')
                while not (t==ti1): # Increase images till in-sync again...
                    topic, im1, ti1 = img1_gen.next()
                    topic, im2, ti2 = img2_gen.next()
                    topic, im3, ti3 = img3_gen.next()
                    topic, im4, ti4 = img4_gen.next()
                    topic, im5, ti5 = img5_gen.next()
                    topic, lab1, tl1 = lab1_gen.next()
                    topic, lab2, tl2 = lab2_gen.next()
                    topic, lab3, tl3 = lab3_gen.next()
                    topic, lab4, tl4 = lab4_gen.next()
                    topic, lab5, tl5 = lab5_gen.next()
        else:
            skip_img_incr = False

        # Last line of defense.
        assert (t==ti1) and (t==ti2) and (t==ti3) and (t==ti4) and (t==ti5) and (t==tl1) and (t==tl2) and (t==tl3) and (t==tl4) and (t==tl5), "Timestamp not synced!"
        i+=1

        # Rotate it into native LB3 orientation.
        cv_image1 = bridge.imgmsg_to_cv2(im1, desired_encoding='bgr8')
        cv_image2 = bridge.imgmsg_to_cv2(im2, desired_encoding='bgr8')
        cv_image3 = bridge.imgmsg_to_cv2(im3, desired_encoding='bgr8')
        cv_image4 = bridge.imgmsg_to_cv2(im4, desired_encoding='bgr8')
        cv_image5 = bridge.imgmsg_to_cv2(im5, desired_encoding='bgr8')
        
        cv_label1 = bridge.imgmsg_to_cv2(lab1, desired_encoding='bgr8')
        cv_label2 = bridge.imgmsg_to_cv2(lab2, desired_encoding='bgr8')
        cv_label3 = bridge.imgmsg_to_cv2(lab3, desired_encoding='bgr8')
        cv_label4 = bridge.imgmsg_to_cv2(lab4, desired_encoding='bgr8')
        cv_label5 = bridge.imgmsg_to_cv2(lab5, desired_encoding='bgr8')
        
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
            point_c1 = np.dot(tf_c1_body, point_hmg)[:3]
            point_c2 = np.dot(tf_c2_body, point_hmg)[:3]
            point_c3 = np.dot(tf_c3_body, point_hmg)[:3]
            point_c4 = np.dot(tf_c4_body, point_hmg)[:3]
            point_c5 = np.dot(tf_c5_body, point_hmg)[:3]
            
            # Project onto image.
            # camera_point = np.dot(camera_intrinsics_sc, point_cam)
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
            if camera1_point[2] > 0 and u1 > left_sc and u1 < right_sc and v1 > top_sc and v1 < bot_sc:
                # Image 1.
                bgr = cv_image1[v1, u1]
                bgr_sem = cv_label1[v1, u1]
            elif camera2_point[2] > 0 and u2 > left_sc and u2 < right_sc and v2 > top_sc and v2 < bot_sc:
                # Image 2.
                bgr = cv_image2[v2, u2]
                bgr_sem = cv_label2[v2, u2]
            elif camera3_point[2] > 0 and u3 > left_sc and u3 < right_sc and v3 > top_sc and v3 < bot_sc:
                # Image 3.
                bgr = cv_image3[v3, u3]
                bgr_sem = cv_label3[v3, u3]
            elif camera4_point[2] > 0 and u4 > left_sc and u4 < right_sc and v4 > top_sc and v4 < bot_sc:
                # Image 4.
                bgr = cv_image4[v4, u4]
                bgr_sem = cv_label4[v4, u4]
            elif camera5_point[2] > 0 and u5 > left_sc and u5 < right_sc and v5 > top_sc and v5 < bot_sc:
                # Image 5.
                bgr = cv_image5[v5, u5]
                bgr_sem = cv_label5[v5, u5]
            else:
                bgr = [255,255,255]
                bgr_sem = [0,0,0]

            # Create PointXYZRGBA
            label = lookup_id_color[bgr_sem[0], bgr_sem[1], bgr_sem[2]]
            rgba = struct.unpack('I', struct.pack(
                    'BBBB', bgr[0], bgr[1], bgr[2], int(label) * 7))[0]
            
            aug_pt = [float(point_nclt[0]), float(-point_nclt[1]), float(-point_nclt[2]), rgba] # Swap back from NCLT to ROS frames.
            augmented_points.append(aug_pt)

        # Create augmented cloud.
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1)]

        header = lidar_pcl.header
        header.frame_id = '/base_link'
        augmented_cloud = point_cloud2.create_cloud(header, fields, augmented_points)

        out_bag.write('/augmented_cloud', augmented_cloud,
                      augmented_cloud.header.stamp, False)
        
        if i%20==0:
            print('Written ' + str(i) + ' clouds!')
            print('Written ' + str(t.secs - start_t.secs) + ' seconds')
            print('Took ' + str(time.time() - time_st) + ' seconds so far to process.')

    out_bag.close()
    print('Bag closed!')
    print(i)
    print(skip_pcl1)
    print(skip_pcl2)

if __name__ == '__main__':
    main()
