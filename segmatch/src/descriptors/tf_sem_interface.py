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


class TensorflowSemanticsInterface:
    def sem_input_callback(self, msg):

        input_tensor_name = msg.input_tensor_name
        output_tensor_name = msg.output_tensor_name
        message_id = msg.timestamp

        inputs = empty([msg.inputs.layout.dim[0].size,
                        msg.inputs.layout.dim[1].size])

        for i in range(0, msg.inputs.layout.dim[0].size):
            for j in range(0, msg.inputs.layout.dim[1].size):
                inputs[i][j] = msg.inputs.data[msg.inputs.layout.dim[1].stride * i + j]

        self.tf_interface_semantics(inputs, input_tensor_name,
                                    output_tensor_name, message_id)

    def tf_interface_semantics(self, inputs, input_tensor_name,
                               output_tensor_name, message_id):

        semantics = self.sem_session.run(self.sem_output, feed_dict={
            self.sem_input: inputs},)

        self.publish_sem_output(message_id, semantics)

    def publish_sem_output(self, message_id, semantics):
        out_msg = sem_output_msg()
        layout_dim = MultiArrayDimension()
        out_msg.timestamp = message_id

        layout_dim.size = semantics.shape[0]
        layout_dim.stride = semantics.shape[0] * semantics.shape[1]
        out_msg.semantics.layout.dim.append(copy.copy(layout_dim))

        layout_dim.size = semantics.shape[1]
        layout_dim.stride = semantics.shape[1]
        out_msg.semantics.layout.dim.append(copy.copy(layout_dim))

        for i in range(0, out_msg.semantics.layout.dim[0].size):
            for j in range(0, out_msg.semantics.layout.dim[1].size):
                out_msg.semantics.data.append(semantics[i][j])

        self.sem_output_publisher.publish(out_msg)

    def setup(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('tf_interface_topic/sem_input_topic',
                         sem_input_msg, self.sem_input_callback)
        self.sem_output_publisher = rospy.Publisher(
            'tf_interface_topic/sem_output_topic', sem_output_msg, queue_size=50)
        self.semantics_model_path = rospy.get_param(
            '/SegMapper/SegMatchWorker/SegMatch/Descriptors/semantics_nn_path')

        # Semantics
        tf.compat.v1.get_default_graph()
        self.saver = tf.compat.v1.train.import_meta_graph(
            os.path.join(self.semantics_model_path, "model.ckpt.meta")
        )

        sem_graph = tf.compat.v1.get_default_graph()
        self.sem_input = sem_graph.get_tensor_by_name('InputScope/input:0')
        self.sem_output = sem_graph.get_tensor_by_name(
            'OutputScope/output_read:0')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        self.cnn_sess = tf.compat.v1.Session(
            config=tf.ConfigProto(gpu_options=gpu_options))ons))
        self.sem_session = tf.compat.v1.Session()
        self.saver.restore(self.sem_session, tf.train.latest_checkpoint(
            self.semantics_model_path))
        rospy.spin()


if __name__ == '__main__':
    interface = TensorflowSemanticsInterface()
    interface.setup()
