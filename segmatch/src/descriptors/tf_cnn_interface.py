#!/usr/bin/env python3
from __future__ import print_function
from array import *
import rospy
from segmatch.msg import cnn_input_msg
from segmatch.msg import cnn_output_msg
from segmatch.msg import sem_input_msg
from segmatch.msg import sem_output_msg
from std_msgs.msg import MultiArrayDimension
from numpy import empty
from numpy import zeros
import os
import tensorflow as tf
import copy


class TensorflowCNNInterface:

    def cnn_input_callback(self, msg):

        # rospy.loginfo('Receiving at: %s', msg.timestamp)
        input_tensor_name = msg.input_tensor_name
        scales_tensor_name = msg.scales_tensor_name
        descriptor_values_name = msg.descriptor_values_name
        reconstruction_values_name = msg.reconstruction_values_name
        message_id = msg.timestamp
        # Scales
        scales = empty([msg.scales.layout.dim[0].size,
                        msg.scales.layout.dim[1].size])

        for i in range(0, msg.scales.layout.dim[0].size):
            for j in range(0, msg.scales.layout.dim[1].size):
                scales[i][j] = msg.scales.data[msg.scales.layout.dim[1].stride * i + j]

        # Inputs

        inputs = empty([msg.inputs.layout.dim[0].size, msg.inputs.layout.dim[1].size,
                        msg.inputs.layout.dim[2].size, msg.inputs.layout.dim[3].size, 1])

        stride_1 = msg.inputs.layout.dim[1].stride
        stride_2 = msg.inputs.layout.dim[2].stride
        stride_3 = msg.inputs.layout.dim[3].stride

        for i in range(0, msg.inputs.layout.dim[0].size):
            for j in range(0, msg.inputs.layout.dim[1].size):
                for k in range(0, msg.inputs.layout.dim[2].size):
                    for l in range(0, msg.inputs.layout.dim[3].size):
                        inputs[i][j][k][l][0] = msg.inputs.data[i *
                                                                stride_1 + j*stride_2 + k * stride_3 + l]

        self.tf_interface(
            inputs, scales, input_tensor_name, scales_tensor_name, descriptor_values_name, reconstruction_values_name, message_id)

    def tf_interface(self, inputs, scales, input_tensor_name,
                     scales_tensor_name, descriptor_values_name, reconstruction_values_name, message_id):

        descriptors = self.cnn_sess.run(self.cnn_descriptor, feed_dict={
            self.cnn_input: inputs, self.cnn_scales: scales},)
        reconstructions = self.cnn_sess.run(self.cnn_reconstruction, feed_dict={
            self.cnn_input: inputs, self.cnn_scales: scales},)
        self.publish_cnn_output(reconstructions, descriptors, message_id)

    def publish_cnn_output(self, reconstructions, descriptors, message_id):

        out_msg = cnn_output_msg()
        layout_dim = MultiArrayDimension()
        out_msg.timestamp = message_id

        # Descriptors

        layout_dim.size = descriptors.shape[0]
        layout_dim.stride = descriptors.shape[0] * descriptors.shape[1]
        out_msg.descriptors.layout.dim.append(copy.copy(layout_dim))

        layout_dim.size = descriptors.shape[1]
        layout_dim.stride = descriptors.shape[1]
        out_msg.descriptors.layout.dim.append(copy.copy(layout_dim))

        for i in range(0, out_msg.descriptors.layout.dim[0].size):
            for j in range(0, out_msg.descriptors.layout.dim[1].size):
                out_msg.descriptors.data.append(descriptors[i][j])

        # Reconstructions

        layout_dim.size = reconstructions.shape[0]
        layout_dim.stride = reconstructions.shape[0] * \
            reconstructions.shape[1] * \
            reconstructions.shape[2] * reconstructions.shape[3]
        out_msg.reconstructions.layout.dim.append(copy.copy(layout_dim))

        layout_dim.size = reconstructions.shape[1]
        layout_dim.stride = reconstructions.shape[1] * \
            reconstructions.shape[2] * reconstructions.shape[3]
        out_msg.reconstructions.layout.dim.append(copy.copy(layout_dim))

        layout_dim.size = reconstructions.shape[2]
        layout_dim.stride = reconstructions.shape[2] * reconstructions.shape[3]
        out_msg.reconstructions.layout.dim.append(copy.copy(layout_dim))

        layout_dim.size = reconstructions.shape[3]
        layout_dim.stride = reconstructions.shape[3]
        out_msg.reconstructions.layout.dim.append(copy.copy(layout_dim))

        for i in range(0, out_msg.reconstructions.layout.dim[0].size):
            for j in range(0, out_msg.reconstructions.layout.dim[1].size):
                for k in range(0, out_msg.reconstructions.layout.dim[2].size):
                    for l in range(0, out_msg.reconstructions.layout.dim[3].size):
                        out_msg.reconstructions.data.append(
                            reconstructions[i][j][k][l][0])

        self.cnn_output_publisher.publish(out_msg)

    def setup(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('tf_interface_topic/cnn_input_topic',
                         cnn_input_msg, self.cnn_input_callback)

        self.cnn_output_publisher = rospy.Publisher(
            'tf_interface_topic/cnn_output_topic', cnn_output_msg, queue_size=50)

        self.cnn_model_path = rospy.get_param(
            '/SegMapper/SegMatchWorker/SegMatch/Descriptors/cnn_model_path')

        # CNN features
        tf.compat.v1.get_default_graph()
        self.cnn_saver = tf.compat.v1.train.import_meta_graph(
            os.path.join(self.cnn_model_path, "model.ckpt.meta")
        )
        cnn_graph = tf.compat.v1.get_default_graph()
        self.cnn_input = cnn_graph.get_tensor_by_name('InputScope/input:0')
        self.cnn_scales = cnn_graph.get_tensor_by_name('scales:0')
        self.cnn_descriptor = cnn_graph.get_tensor_by_name(
            'OutputScope/descriptor_read:0')
        self.cnn_reconstruction = cnn_graph.get_tensor_by_name(
            'ReconstructionScopeAE/ae_reconstruction_read:0')
        self.cnn_sess = tf.compat.v1.Session()
        self.cnn_saver.restore(self.cnn_sess, tf.train.latest_checkpoint(
            self.cnn_model_path))
        rospy.spin()


if __name__ == '__main__':
    interface = TensorflowCNNInterface()
    interface.setup()
