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
from std_msgs.msg import Float64MultiArray

import cv2
import cv_bridge as cvb


def main():
    # File paths.
    # ToDo(alaturn) read these in as argument.
    bag_file = '/media/nikhilesh/Nikhilesh/SemSegMap/Bags/BOSCH/bosch_augmented_1_cam.bag' 
    out_bag_file = '/media/nikhilesh/Nikhilesh/SemSegMap/Bags/BOSCH/locnet.bag' 
    model_file = '/home/nikhilesh/segmap_ws/src/LocNet_caffe/models/kitti_range.caffemodel'
    config_file = '/home/nikhilesh/segmap_ws/src/LocNet_caffe/cfg/kitti_range_deploy.prototxt'
    bag = rosbag.Bag(bag_file)
    out_bag = rosbag.Bag(out_bag_file, 'w')
    caffe.set_mode_cpu()
    net = caffe.Net(config_file, model_file, caffe.TEST)
    
    # Parameters of the handcrafted LocNet histogram (input to CNN).
    # ToDo(alaturn) read these in as argument.
    max_distance = 200  # ToDo(alatur) isn't this what d_max is for??
    d_min = 2.5 # There are some body returns.
    d_max = 80 #beyond it's not nice data.
    z_min = -2.0
    theta_deg_min = -40 # positive theta = above horizont
    theta_deg_max = 40 
    image_width = 640
    image_height = 480
    bucket_count = 80       # given from network
    network_input_size = 64  # given from network
    delta_i_b = (1.0 / bucket_count) * (d_max - d_min)
    delta_theta_deg = (1.0/network_input_size)*(theta_deg_max - theta_deg_min)
    i = 0
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

        # print len(list(points))
        l = 0
        maxx = - 1000
        minn = 1000 
        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
            r = math.sqrt(x*x + y*y + z*z)
            r2d = math.sqrt(x*x + y*y)

            if  (r < d_min or r > d_max):# or (z < z_min):
                continue

            # 1. Compute vertical angle.
            theta_deg = numpy.sign(z)*math.atan(abs(z)/r2d)/math.pi*180.0 + abs(theta_deg_min)

            # 2. Assign to scan line.
            scan_line = int(math.floor(theta_deg/delta_theta_deg))

            # 3. Assign to distance bucket.
            bucket = int(math.floor((r-d_min)/delta_i_b))

            # 4. Increase counter.
            histogram[0,0,bucket,scan_line] +=1

            # if r > max_distance:
            #     point_valid = False
            # else:
            #     point_valid = True

            # if point_valid and last_point_valid and not(is_new_line):

            #     # ToDo(alaturn) Switch to R mode (better than deltaR)
            #     #dist = math.sqrt((x - last_x)**2 + (y-last_y)**2)
            #     for n in range(0, bucket_count):

            #         if d_min + n * delta_i_b > dist:
            #             histogram[0, 0, n, ring_index] += 1
            #             break

            # # update for next point
            # last_point_valid = point_valid
            # is_new_line = False
            # last_x = x
            # last_y = y
            # azimuth_index += 1
            # if azimuth_index == image_width:
            #     line_index += 1
            #     azimuth_index = 0
            #     is_new_line = True
            # ring_index = math.trunc(
            #     line_index * network_input_size / image_height)
        
        # ToDo(alaturn) Normalize count in each ring.
        k = 0
        for col in numpy.transpose(histogram[0,0,:,:]):
            ring_count = max(1,numpy.sum(col))
            histogram[0,0,:,k] = histogram[0,0,:,k]/ring_count
            k +=1

        # Now pass the histogram through the network.
        net.forward_all(**{"data": histogram})
        output = net.blobs['feat'].data

        # Save the computed metrics to new bag.
        bridge = cvb.CvBridge()

        # Convert Histogram to image & save to bag.
        print 'max val bef'
        print numpy.max(histogram)
        histogram_img = histogram[0,0,:,:]*254
        print 'max val aft'
        print numpy.max(histogram_img)
        histogram_img = histogram_img.astype(numpy.uint8)
        histogram_img_large = cv2.resize(histogram_img, None, fx = 7, fy = 7, interpolation = cv2.INTER_CUBIC)
        histogram_img_large = numpy.transpose(histogram_img_large)
        histogram_img_large = numpy.flipud(histogram_img_large)
        histogram_img_large = numpy.dstack((histogram_img_large,histogram_img_large,histogram_img_large))
        histogram_img_msg = bridge.cv2_to_imgmsg(histogram_img_large, encoding="bgr8")
        cv2.imshow("image", histogram_img_large)
        cv2.waitKey(100)

        # Feature vector.
        output_msg = Float64MultiArray(data=output[0])

        # Convert feature vector to image & save to bag.
        feature_vec = output[0]
        norm = numpy.linalg.norm(feature_vec)
        feature_vec = (feature_vec/norm)*254
        feature_vec = feature_vec.astype(numpy.uint8)
        feature_vec = numpy.transpose(feature_vec)
        feature_vec = cv2.rotate(feature_vec, cv2.ROTATE_90_CLOCKWISE)
        feature_img_large = cv2.resize(feature_vec, None, fx = 20, fy = 70, interpolation = cv2.INTER_CUBIC)
        feature_img_msg = bridge.cv2_to_imgmsg(feature_img_large, encoding="mono8")
        # cv2.imshow("image", feature_img_large)
        # cv2.waitKey()


        # Save to bag.
        # out_bag.write('/augmented_cloud', pcl,
        #               pcl.header.stamp, False)
        out_bag.write('/locnet_range_histogram', histogram_img_msg,
                      pcl.header.stamp, False)
        out_bag.write('/locnet_descriptor', output_msg,
                      pcl.header.stamp, False)
        out_bag.write('/locnet_descriptor_img', feature_img_msg, pcl.header.stamp, False)
        i += 1
        print i
        # if i == 100:
        #     break
        

    tf_it = 0
    for topic, tf, t in bag.read_messages(topics=['/tf']):
        out_bag.write('/tf', tf, tf.transforms[0].header.stamp, False)
        tf_it += 1
        if tf_it == 1:
            time_hack = tf.transforms[0].header.stamp
        print('TF: ' + str(tf_it))

    for topic, tf_static, t in bag.read_messages(topics=['/tf_static']):
        out_bag.write('/tf_static', tf_static,
                      time_hack, False)
    out_bag.close()
    

if __name__ == '__main__':
    main()
