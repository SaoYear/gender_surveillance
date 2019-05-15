from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.slim.python.slim.nets import resnet_utils, resnet_v2
# ResNet has the shape of 50, 101, 152, 200
import tensorflow as tf
import train_process as p
import test_process as t
import val_process as v
import random
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ResNet
# Args:
#     inputs: A tensor of size [batch, height_in, width_in, channels].
#     blocks: A list of length equal to the number of ResNet blocks. Each element
#       is a resnet_utils.Block object describing the units in the block.
#     num_classes: Number of predicted classes for classification tasks. If None
#       we return the features before the logit layer.
#     is_training: whether batch_norm layers are in training mode.
#     global_pool: If True, we perform global average pooling before computing the
#       logits. Set to True for image classification, False for dense prediction.
#     output_stride: If None, then the output will be computed at the nominal
#       network stride. If output_stride is not None, it specifies the requested
#       ratio of input to output spatial resolution.
#     include_root_block: If True, include the initial convolution followed by
#       max-pooling, if False excludes it. If excluded, `inputs` should be the
#       results of an activation-less convolution.
#     reuse: whether or not the network and its variables should be reused. To be
#       able to reuse 'scope' must be given.
#     scope: Optional variable_scope.
# Return:
#     net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
#       If global_pool is False, then height_out and width_out are reduced by a
#       factor of output_stride compared to the respective height_in and width_in,
#       else both height_out and width_out equal one. If num_classes is None, then
#       net is the output of the last ResNet block, potentially after global
#       average pooling. If num_classes is not None, net contains the pre-softmax
#       activations.
#     end_points: A dictionary from components of the network to the corresponding
#       activation.

class ResNet_Model(object):
    def __init__(self, lr=1e-4, batch_size=128, dropout_prob=0.8, epoch=100,
                 train_dir='C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/PETA dataset_19000/rescale/',
                 test_dir='C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/PETA dataset_19000/test/'):

        # 输入数据
        self.data = p.ImgInput(train_dir, batch_size)
        self.test = t.ImgInput(test_dir, batch_size)
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.epoch = epoch
        self.cross_entropy = 0
        self.train_step = 0
        self.init = 0
        self.build_model()

    def build_model(self):
        self.img = tf.placeholder(tf.float32, [None, 150, 100, 3])/255
        self.label = tf.placeholder(tf.float32, [None, 2])

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, _ = resnet_v2.resnet_v2_50(self.img, num_classes=2, is_training=True, global_pool=True)
            net = tf.reshape(net, [self.batch_size, 2])
            # dense_1 = tf.layers.dense(net, units=5, activation=tf.nn.relu)
            self.prediction_1 = net

            # self.prediction = tf.nn.softmax(tf.layers.dense(dense_1, units=2))

        with tf.variable_scope("ensemble_2"):
            with tf.variable_scope('convolution_1') as scope:
                W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 32], mean=0, stddev=0.01), name='conv1')
                b_conv1 = tf.Variable(tf.truncated_normal(shape=[32], mean=0, stddev=0.1), name='bias1')
                h_conv1 = tf.nn.dropout(
                    tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.img, W_conv1, strides=[1, 1, 1, 1], padding='VALID'),
                                              b_conv1)),
                    keep_prob=self.dropout_prob)  # output size 84x84x16
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='VALID')  # output size 42x42x16

            with tf.variable_scope('convolution_2') as scope:
                W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 10], mean=0, stddev=0.01), name='conv2')
                b_conv2 = tf.Variable(tf.truncated_normal(shape=[10], mean=0, stddev=0.05), name='bias2')
                h_conv2 = tf.nn.dropout(
                    tf.nn.relu(
                        tf.nn.bias_add(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID'), b_conv2)),
                    keep_prob=self.dropout_prob)
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                self.dense_input = tf.reshape(h_pool2, (-1, 34 * 22 * 10))

            with tf.variable_scope('dense') as scope:
                h_dense1 = tf.nn.dropout(
                    tf.layers.dense(self.dense_input, units=34 * 22 * 10, activation=tf.nn.relu),
                    keep_prob=self.dropout_prob)
                h_dense2 = tf.layers.dense(h_dense1, units=2, activation=None)
                self.prediction_2 = h_dense2

        # self.sum_prediction = tf.concat([self.prediction_1, self.prediction_2], axis=1)
        self.sum_prediction = (self.prediction_1 + self.prediction_2)/2
        print('sum_prediction:', self.sum_prediction)

        with tf.variable_scope('result') as scope:
            # weight = tf.reshape(tf.nn.softmax(tf.layers.dense(self.dense_input, units=2)), (self.batch_size, 1, 2))
            # self.pin = weight
            # self.prediction = tf.nn.softmax(tf.reshape(tf.matmul(weight, self.sum_prediction), (self.batch_size, 2)))
            self.prediction = tf.nn.softmax(self.sum_prediction)
            print("prediction:", self.prediction)
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))

            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(
                self.label * tf.log(tf.clip_by_value(self.prediction, 1e-10, 0.999999)), reduction_indices=[1]))

        global_step = tf.Variable(0, trainable=False)
        # self.learning = tf.train.exponential_decay(self.lr, global_step, 70, 0.8, staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy, global_step=global_step)

        # self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
        #     self.cross_entropy, global_step=global_step)
        # self.f_cross_entropy = tf.reduce_mean(
        #     -tf.reduce_sum(
        #         self.label * tf.log(tf.clip_by_value(self.prediction, 1e-10, 0.999999)),
        #         reduction_indices=[1]))


    def train(self):
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        _loop = self.data.total_batch // self.data.batch_size
        _loop_test = self.test.total_batch // self.data.batch_size
        for i in range(self.epoch):
            wrong_index = []

            for j in range(_loop):
                train_x, train_y = self.data.next_batch()

                #pre = sess.run(prediction, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
                #print(np.size(pre))
                #sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
                _, c, result = sess.run([self.train_step, self.cross_entropy, self.prediction],
                                              feed_dict={self.img: train_x, self.label: train_y})

                # if j % 20 == 0:

                # for index, value in enumerate(list(test)):
                #     if not value:
                #         wrong_index.append((index + j * self.batch_size) % 11122)
                # print('Learning rate:', test)
            # print("Pin sample:", test)
            total_accuracy = []
            for k in range(_loop_test):
                text_x, text_y = self.test.next_batch()
                accuracy = sess.run(self.accuracy, feed_dict={self.img: text_x, self.label: text_y})
                total_accuracy.append(accuracy)
            print(total_accuracy)
            print(" Accuracy:", np.mean(total_accuracy))
            print("Epoch times:" + str(i))
            print("cross_entropy = ", "{:.8f}".format(c))
            # print('Test loss:', loss)



model = ResNet_Model()
model.train()
