import tensorflow as tf
import NN_code.Image_Process.train_process as p
import NN_code.Image_Process.test_process as t
import random
import numpy as np


class Model(object):
    def __init__(self, lr=5e-5, batch_size=128, dropout_prob=0.9, epoch=150,
                 train_dir='C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/data_train/',
                 val_dir='C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/data_test/',
                 test_dir='C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/data_val/'):
        # 输入数据
        self.data = p.ImgInput(train_dir, batch_size)
        self.test = p.ImgInput(val_dir, batch_size)
        self.true_test = p.ImgInput(test_dir, batch_size)
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.epoch = epoch
        self.cross_entropy = 0
        self.train_step = 0
        self.init = 0
        self.build_model()

    def build_model(self):
        self.img = tf.placeholder(tf.float32, [None, 160, 96, 3])/255
        self.label = tf.placeholder(tf.float32, [None, 2])
        with tf.variable_scope('model_1'):
            prediction_1 = self.conv_model()
        with tf.variable_scope('model_2'):
            prediction_2 = self.conv_model()
        with tf.variable_scope('model_3'):
            prediction_3 = self.conv_model()
        # with tf.variable_scope('model_4'):
        #     prediction_4 = self.conv_model()
        # with tf.variable_scope('model_5'):
        #     prediction_5 = self.conv_model()
        # with tf.variable_scope('model_6'):
        #     prediction_6 = self.conv_model()
        # with tf.variable_scope('model_7'):
        #     prediction_7 = self.conv_model()
        # with tf.variable_scope('model_8'):
        #     prediction_8 = self.conv_model()
        # with tf.variable_scope('model_9'):
        #     prediction_9 = self.conv_model()

        self.prediction = (prediction_1
                           + prediction_2
                           + prediction_3
                           # + prediction_4
                           # + prediction_5
                           # + prediction_6
                           # + prediction_7
                           # + prediction_8
                           # + prediction_9
                           )/3
        with tf.variable_scope('result') as scope:
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(
                self.label * tf.log(tf.clip_by_value(self.prediction, 1e-10, 0.999999)),
                reduction_indices=[1]))

        global_step = tf.Variable(0, trainable=False)
        # self.learning = tf.train.exponential_decay(self.lr, global_step, 700, 0.1, staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy, global_step=global_step)
        self.f_cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(
                self.label * tf.log(tf.clip_by_value(self.prediction, 1e-10, 0.999999)),
                reduction_indices=[1]))


    def train(self):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        _loop = self.data.total_batch // self.batch_size
        _loop_test = self.test.total_batch // self.batch_size
        for i in range(self.epoch):
            random.shuffle(self.data.indices)
            for j in range(_loop):
                train_x, train_y = self.data.next_batch()

                #pre = sess.run(prediction, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
                #print(np.size(pre))
                #sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
                _, c = sess.run([self.train_step, self.cross_entropy], feed_dict={self.img: train_x, self.label: train_y})
            total_accuracy = []
            for k in range(_loop_test+1):
                text_x, text_y = self.test.next_batch()
                accuracy, loss = sess.run([self.accuracy, self.f_cross_entropy],
                                            feed_dict={self.img: text_x, self.label: text_y})
                total_accuracy.append(accuracy)

            print("Epoch times:", i+1)
            print("cross_entropy = ", "{:.8f}".format(c))
            # print("Accuracy list:", total_accuracy)
            print('Accuracy:', np.mean(total_accuracy))

        total_accuracy = []
        for k in range(_loop_test + 1):
            text_x, text_y = self.true_test.next_batch()
            accuracy, loss = sess.run([self.accuracy, self.f_cross_entropy],
                                      feed_dict={self.batch_size: self.batch_size, self.img: text_x, self.label: text_y})
            total_accuracy.append(accuracy)
        print("Test accuracy:", np.mean(total_accuracy))


    def conv_model(self):
        with tf.variable_scope('conv1') as scope:
            W_conv1 = tf.Variable(tf.truncated_normal(shape=[4, 4, 3, 10], mean=0, stddev=0.01), name='conv1')
            b_conv1 = tf.Variable(tf.truncated_normal(shape=[10], mean=0, stddev=0.1), name='bias1')
            h_conv1 = tf.nn.dropout(
                tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(self.img, W_conv1, strides=[1, 1, 1, 1], padding='VALID'), b_conv1)),
                keep_prob=self.dropout_prob)        # output size 84x84x16
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')  # output size 42x42x16

        with tf.variable_scope('conv2') as scope:
            W_conv2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 10, 32], mean=0, stddev=0.01), name='conv2')
            b_conv2 = tf.Variable(tf.truncated_normal(shape=[32], mean=0, stddev=0.1), name='bias2')
            h_conv2 = tf.nn.dropout(
                tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID'), b_conv2)),
                keep_prob=self.dropout_prob)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('conv3') as scope:
            W_conv3 = tf.Variable(tf.random_normal(shape=[4, 4, 32, 64], mean=0, stddev=0.01), name='conv3')
            b_conv3 = tf.Variable(tf.random_normal(shape=[64], mean=0, stddev=0.1), name='bias3')
            h_conv3 = tf.nn.leaky_relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
            h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            print(h_pool3)
        dense_input = tf.reshape(h_pool3, (self.batch_size, 9792))

        with tf.variable_scope('dense') as scope:
            h_dense1 = tf.nn.dropout(
                tf.layers.dense(dense_input, units=9792, activation=tf.nn.leaky_relu), keep_prob=self.dropout_prob)
            h_dense2 = tf.layers.dense(h_dense1, units=20)
            h_dense3 = tf.layers.dense(h_dense2, units=2, activation=None)
            prediction = tf.nn.softmax(h_dense3, axis=1)

        return prediction
model = Model()
model.train()
