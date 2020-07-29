#!/usr/bin/env python
# from __future__ import print_function
from array import *
import rospy
from segmatch.msg import batch_full_forward_pass_msg
from numpy import empty


def batch_full_forward_pass_callback(msg):
    # rospy.loginfo('I heard %s', msg.timestamp)
    input_tensor_name = msg.input_tensor_name
    scales_tensor_name = msg.scales_tensor_name
    descriptor_values_name = msg.descriptor_values_name
    reconstruction_values_name = msg.reconstruction_values_name

    # Scales
    scales = empty([msg.scales.layout.dim[0].size,
                    msg.scales.layout.dim[1].size])

    for i in range(0, msg.scales.layout.dim[0].size):
        for j in range(0, msg.scales.layout.dim[1].size):
            scales[i][j] = msg.scales.data[msg.scales.layout.dim[1].stride * i + j]

        # Inputs
    inputs = empty([msg.inputs.layout.dim[0].size, msg.inputs.layout.dim[1].size,
                    msg.inputs.layout.dim[2].size, msg.inputs.layout.dim[3].size])

    stride_1 = msg.inputs.layout.dim[1].stride
    stride_2 = msg.inputs.layout.dim[2].stride
    stride_3 = msg.inputs.layout.dim[3].stride

    for i in range(0, msg.inputs.layout.dim[0].size):
        for j in range(0, msg.inputs.layout.dim[1].size):
            for k in range(0, msg.inputs.layout.dim[2].size):
                for l in range(0, msg.inputs.layout.dim[3].size):
                    inputs[i][j][k][l] = msg.inputs.data[i *
                                                         stride_1 + j*stride_2 + k * stride_3 + l]

    # for r in scales:
    #     for c in r:
    #         print(c, end=" ")
    #     print()
    batch_full_forward_pass_tf_interface(
        inputs, scales, input_tensor_name, scales_tensor_name, descriptor_values_name, reconstruction_values_name)


def batch_full_forward_pass_tf_interface(inputs, scales, input_tensor_name,
scales_tensor_name, descriptor_values_name, reconstruction_values_name):
    print('ola')


def listener():
    rospy.init_node('listener', anonymous = True)
    rospy.Subscriber('tf_interface_topic/batch_full_forward_pass_topic',
                     batch_full_forward_pass_msg, batch_full_forward_pass_callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
