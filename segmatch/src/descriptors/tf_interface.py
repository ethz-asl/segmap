#!/usr/bin/env python

import rospy
from segmatch.msg import tensorflow_msg


def callback(data):
    rospy.loginfo('I heard %s', data.timestamp)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('tf_interface_topic/tensorflow_msg', tensorflow_msg, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
