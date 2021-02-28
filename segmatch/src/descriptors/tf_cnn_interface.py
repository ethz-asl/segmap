#!/usr/bin/env python
from __future__ import print_function
from array import *
import rospy
from segmatch.msg import cnn_input_msg
from segmatch.msg import cnn_output_msg
from segmatch.msg import sem_input_msg
from segmatch.msg import sem_output_msg
from std_msgs.msg import MultiArrayDimension
import numpy as np
import os
import tensorflow as tf
import copy
from threading import Lock

class TensorflowCNNInterface:
    def cnn_input_callback(self, msg):
        message_id = msg.timestamp

        # Scales
        scales = np.empty([msg.scales.layout.dim[0].size,
                           msg.scales.layout.dim[1].size])

        for i in range(0, msg.scales.layout.dim[0].size):
            for j in range(0, msg.scales.layout.dim[1].size):
                scales[i][j] = msg.scales.data[msg.scales.layout.dim[1].stride * i + j]

        # Inputs
        n_segments = msg.input_indexes.layout.dim[0].size
        stride = msg.inputs.layout.dim[1].stride

        voxels = np.array((32, 32, 16))
        inputs = np.zeros((n_segments,) + tuple(voxels) + (1,))
        if self.use_semantics:
            semantics = np.zeros(
                (n_segments, self.n_classes), dtype=np.float)
        else:
            semantics = []

        start_index = 0
        for s in range(n_segments):
            end_index = msg.input_indexes.data[s]

            for i in range(start_index, end_index):
                x = msg.inputs.data[i * stride]
                y = msg.inputs.data[i * stride + 1]
                z = msg.inputs.data[i * stride + 2]

                if self.use_semantics:
                    #r = msg.inputs.data[i * stride + 3] / 255.0
                    #g = msg.inputs.data[i * stride + 4] / 255.0
                    #b = msg.inputs.data[i * stride + 5] / 255.0
                    semantic_class = msg.inputs.data[i * stride + 6]
                    #semantics[s][semantic_class] += 1
                    semantics[s][0] += 1
                else:
                    inputs[s][x][y][z][0] = 1.0
            semantics[s] /= float(end_index - start_index)

            start_index = end_index

        self.mutex.acquire()
        self.tf_interface(
            inputs, scales, message_id, semantics=semantics)

    def tf_interface(self, inputs, scales, message_id, semantics=None):
        feed_dict = {
            self.cnn_input: inputs,
            self.cnn_scales: scales}
        if self.use_semantics:
            feed_dict[self.cnn_semantics] = semantics

        descriptors = self.cnn_sess.run(self.cnn_descriptor, feed_dict=feed_dict)
        #reconstructions = self.cnn_sess.run(self.cnn_reconstruction, feed_dict)
        reconstructions = np.array([])

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
        '''layout_dim.size = reconstructions.shape[0]
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
                            reconstructions[i][j][k][l][0])'''

        #print("Sending now!!", message_id)
        self.cnn_output_publisher.publish(out_msg)
        self.mutex.release()

    def setup(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('tf_interface_topic/cnn_input_topic',
                         cnn_input_msg, self.cnn_input_callback)

        self.cnn_output_publisher = rospy.Publisher(
            'tf_interface_topic/cnn_output_topic', cnn_output_msg, queue_size=50)

        self.cnn_model_path = rospy.get_param(
            '/SegMapper/SegMatchWorker/SegMatch/Descriptors/cnn_model_path')

        self.use_semantics = rospy.get_param(
            '/SegMapper/SegMatchWorker/SegMatch/Descriptors/use_semantics')

        self.mutex = Lock()

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
        if self.use_semantics:
            self.cnn_semantics = cnn_graph.get_tensor_by_name("semantics:0")
            self.n_classes = 35

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.cnn_sess = tf.compat.v1.Session(
            config=tf.ConfigProto(gpu_options=gpu_options))
        self.cnn_saver.restore(self.cnn_sess, tf.train.latest_checkpoint(
            self.cnn_model_path))
        print("[INFO] Descriptor CNN model loaded.")

        rospy.spin()


if __name__ == '__main__':
    interface = TensorflowCNNInterface()
    interface.setup()
