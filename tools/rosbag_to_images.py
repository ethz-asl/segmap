# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--image_topic", help="Image topic.", nargs='+')
    args = parser.parse_args()

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    topic_counter = {topic : 0 for topic in args.image_topic}
    for topic in args.image_topic:
        topic_subdir = os.path.join(args.output_dir, topic.replace('/', '-'))
        if not os.path.exists(topic_subdir):
            os.makedirs(topic_subdir)

    for topic, msg, t in bag.read_messages(topics=args.image_topic):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(args.output_dir, topic.replace('/', '-'), "frame%06i.png" % topic_counter[topic]), cv_img)
        print("Wrote image %i" % count)

        count += 1
        topic_counter[topic] += 1

    bag.close()

    return

if __name__ == '__main__':
    main()