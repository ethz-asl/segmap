#!/usr/bin/env python

import rospy
from std_msgs.msg import String


def callback(data):
    rospy.loginfo('I heard %s', data.data)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('tensorflow_interface', String, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
