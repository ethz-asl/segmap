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
    bag_file = '/media/nikhilesh/Nikhilesh/SemSegMap/Bags/BOSCH/bosch_augmented_1_cam.bag' 
    out_bag_file = '/media/nikhilesh/Nikhilesh/SemSegMap/Bags/BOSCH/locnet.bag' 
    model_file = '/home/nikhilesh/segmap_ws/src/LocNet_caffe/models/kitti_range.caffemodel'
    config_file = '/home/nikhilesh/segmap_ws/src/LocNet_caffe/cfg/kitti_range_deploy.prototxt'
    bag = rosbag.Bag(bag_file)
    out_bag = rosbag.Bag(out_bag_file, 'w')
    caffe.set_mode_cpu()
    net = caffe.Net(config_file, model_file, caffe.TEST)
    
    # Parameters of the handcrafted LocNet histogram (input to CNN).
    max_distance = 200
    d_min = 0.0
    d_max = 1.5
    image_width = 640
    image_height = 480
    bucket_count = 80       # given from network
    network_input_size = 64  # given from network
    delta_i_b = (1.0 / bucket_count) * (d_max - d_min)
    print 'Hallo'
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

                # ToDo(alaturn) Switch to R mode (better than deltaR)
                dist = math.sqrt((x - last_x)**2 + (y-last_y)**2)
                for n in range(0, bucket_count):

                    if d_min + n * delta_i_b > dist:
                        histogram[0, 0, n, ring_index] += 1
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
            ring_index = math.trunc(
                line_index * network_input_size / image_height)
        
        # Now pass the histogram through the network.
        net.forward_all(**{"data": histogram})
        output = net.blobs['feat'].data

        # Save the computed metrics to new bag.
        output_msg = Float64MultiArray(data=output[0])
        
        # ToDo(alatur) Convert Histogram to image & save to bag.
        histogram_img = numpy.zeros([bucket_count,network_input_size,3])
        histogram_img[:,:,0] = histogram[0,0,:,:]
        histogram_img[:,:,1] = histogram[0,0,:,:]
        histogram_img[:,:,2] = histogram[0,0,:,:]
        histogram_img_large = cv2.resize(histogram_img, None, fx = 7, fy = 7, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("image", histogram_img_large)
        cv2.waitKey()
        bridge = cvb.CvBridge()
        image_message = bridge.cv2_to_imgmsg(histogram_img_large, "bgr8")
        # print image_message.header
        image_message.header.stamp = pcl.header.stamp
        out_bag.write('/range_histogram', image_message, pcl.header.stamp, False)

        out_bag.write('/augmented_cloud', pcl,
                      pcl.header.stamp, False)

        out_bag.write('/locnet_descriptor', output_msg,
                      pcl.header.stamp, False)
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
