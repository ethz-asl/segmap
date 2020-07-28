#!/usr/bin/env python
from __future__ import print_function
from array import *
import rospy
from segmatch.msg import batch_full_forward_pass_msg
from numpy import empty


def batch_full_forward_pass_callback(msg):
    rospy.loginfo('I heard %s', msg.timestamp)

    # ------------------------
    # Decoding of ROS message
    # ------------------------

    # Scales
    scales = empty([msg.scales.layout.dim[0].size,
                    msg.scales.layout.dim[1].size])

    for i in range(0, msg.scales.layout.dim[0].size):
        for j in range(0, msg.scales.layout.dim[1].size):
            scales[i][j] = msg.scales.data[msg.scales.layout.dim[1].stride * i + j]

    for r in scales:
        for c in r:
            print(c, end=" ")
        print()


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('tf_interface_topic/batch_full_forward_pass_topic',
                     batch_full_forward_pass_msg, batch_full_forward_pass_callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
