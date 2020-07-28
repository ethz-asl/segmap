#!/usr/bin/env python

import rospy
from segmatch.msg import batch_full_forward_pass_msg


def batch_full_forward_pass_callback(msg):
    rospy.loginfo('I heard %s', msg.timestamp)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('tf_interface_topic/batch_full_forward_pass_topic',
                     batch_full_forward_pass_msg, batch_full_forward_pass_callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
