import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import NN_code.Image_Process.train_process as p
import random
import numpy as np
import time


class Model(object):
    def __init__(self, lr=5e-5, batch_size=128, dropout_prob=0.9, epoch=100,
                 train_dir='DIR/Database/data_train/',
                 val_dir='DIR/Database/data_test/',
                 test_dir='DIR/Database/data_val/'):
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
        self.bs = tf.placeholder(tf.int32, None)
        self.img = tf.placeholder(tf.float32, [None, 160, 96, 3])/255
        self.label = tf.placeholder(tf.float32, [None, 2])
        print('Image label', self.img)
        print('Batch size:', self.bs)


        with tf.variable_scope('conv1') as scope:
            W_conv1 = tf.Variable(tf.truncated_normal(shape=[4, 4, 3, 15], mean=0, stddev=0.01), name='conv1')
            b_conv1 = tf.Variable(tf.truncated_normal(shape=[15], mean=0, stddev=0.1), name='bias1')
            h_conv1 = tf.nn.dropout(
                tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.img, W_conv1, strides=[1, 1, 1, 1], padding='VALID'), b_conv1)),
                rate=1 - self.dropout_prob)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')  # output size 42x42x16

        with tf.variable_scope('conv2') as scope:
            W_conv2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 15, 48], mean=0, stddev=0.01), name='conv2')
            b_conv2 = tf.Variable(tf.truncated_normal(shape=[48], mean=0, stddev=0.1), name='bias2')
            h_conv2 = tf.nn.dropout(
                tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID'), b_conv2)),
                rate=1 - self.dropout_prob)

            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            print(h_pool2)

        with tf.variable_scope('conv3') as scope:
            W_conv3 = tf.Variable(tf.random_normal(shape=[4, 4, 48, 96], mean=0, stddev=0.01), name='conv3')
            b_conv3 = tf.Variable(tf.random_normal(shape=[96], mean=0, stddev=0.1), name='bias3')
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)
            h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.pin = [h_conv1] + [h_pool1] + [h_conv2] + [h_pool2] + [h_conv3] + [h_pool3]
        # with tf.variable_scope('conv3_dim_change'):
        #     W_conv3_dim = tf.Variable(tf.random_normal(shape=[1, 1, 64, 10], mean=0, stddev=0.01), name='conv3')
        #     b_conv3_dim = tf.Variable(tf.random_normal(shape=[10], mean=0, stddev=0.1), name='bias3')
        #     h_conv3_dim = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv3_dim, strides=[1, 1, 1, 1], padding='VALID') + b_conv3_dim)

        # with tf.variable_scope('conv4'):
        #     W_conv4 = tf.Variable(tf.random_normal(shape=[2, 2, 64, 32], mean=0, stddev=0.01), name='conv4')
        #     b_conv4 = tf.Variable(tf.random_normal(shape=[32], mean=0, stddev=0.1), name='bias4')
        #     h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
        #     # h_pool_res = tf.nn.max_pool(h_conv4 + h_conv3_dim, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #     # h_pool_res = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # with tf.variable_scope('conv5'):
        #     W_conv5 = tf.Variable(tf.random_normal(shape=[2, 2, 32, 10], mean=0, stddev=0.01), name='conv5')
        #     b_conv5 = tf.Variable(tf.random_normal(shape=[10], mean=0, stddev=0.1), name='bias5')
        #     h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5)
        #     # h_pool_res = tf.nn.max_pool(h_conv4 + h_conv3_dim, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print(h_pool3)
        dense_input = tf.reshape(h_pool3, (self.bs, 12288))

        with tf.variable_scope('dense') as scope:
            h_dense1 = tf.nn.dropout(
                tf.layers.dense(dense_input, units=12288, activation=tf.nn.relu), keep_prob=self.dropout_prob)
            h_dense2 = tf.layers.dense(h_dense1, units=20)
            h_dense3 = tf.layers.dense(h_dense2, units=2, activation=None)
            self.prediction = tf.nn.softmax(h_dense3, axis=1)

        with tf.variable_scope('result') as scope:
            print("Final prediction:", self.prediction)
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(
                self.label * tf.log(tf.clip_by_value(self.prediction, 1e-10, 0.999999)),
                reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
        self.f_cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(
                self.label * tf.log(tf.clip_by_value(self.prediction, 1e-10, 0.999999)),
                reduction_indices=[1]))
        self.saver = tf.train.Saver()


    def train(self):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        _loop = self.data.total_batch // self.batch_size
        _loop_test = self.test.total_batch // self.batch_size
        start = time.time()
        accuracy_list = []
        cost_list = []
        slice = []
        for i in range(self.epoch):
            random.shuffle(self.data.indices)
            for j in range(_loop):
                train_x, train_y = self.data.next_batch()
                _, c = sess.run([self.train_step, self.cross_entropy], feed_dict={self.bs: self.batch_size, self.img: train_x, self.label: train_y})
            total_cost = []
            total_accuracy = []
            for k in range(_loop_test+1):
                text_x, text_y = self.test.next_batch()
                accuracy, loss, pin = sess.run([self.accuracy, self.f_cross_entropy, self.pin],
                                            feed_dict={self.bs: self.batch_size, self.img: text_x, self.label: text_y})
                total_accuracy.append(accuracy)
                total_cost.append(c)

            print("Epoch times:", i+1)
            print("cross_entropy = ", "{:.8f}".format(c))
            # print("Accuracy list:", total_accuracy)
            if (i % 4==0):
                cost_list.append(np.mean(total_cost))
                accuracy_list.append(np.mean(total_accuracy))
            if (i == 30):
                for item in pin:
                    p = np.mean(item, axis=3)
                    img = Image.fromarray(p[1, :, :])
                    plt.imshow(img)
                    plt.show()
                for item in pin:
                    p = np.mean(item, axis=3)
                    img = Image.fromarray(p[2, :, :])
                    plt.imshow(img)
                    plt.show()
                for item in pin:
                    p = np.mean(item, axis=3)
                    img = Image.fromarray(p[50, :, :])
                    plt.imshow(img)
                    plt.show()
                for item in pin:
                    p = np.mean(item, axis=3)
                    img = Image.fromarray(p[100, :, :])
                    plt.imshow(img)
                    plt.show()
            print('Accuracy:', np.mean(total_accuracy))
            if np.mean(total_accuracy) >= 0.853:
                save = self.saver.save(sess, './pre-model/model_accuracy_{:.3f}_epoch_{}_trainloss_{:.3f}_valloss_{:.3f}.ckpt'.format(
                    np.mean(total_accuracy), i+1, c, loss))
                print("Model saved in file:", save)

        total_accuracy = []
        for k in range(_loop_test + 1):
            text_x, text_y = self.true_test.next_batch()
            accuracy, loss = sess.run([self.accuracy, self.f_cross_entropy],
                                      feed_dict={self.bs: self.batch_size, self.img: text_x, self.label: text_y})

            total_accuracy.append(accuracy)
        print("Test accuracy:", np.mean(total_accuracy))
        stop = time.time()


        print(accuracy_list)
        print(cost_list)
        print("Time cost:", stop - start)

"""
            # with tf.variable_scope('convolution_4') as scope:
            #     W_conv4 = tf.Variable(tf.random_normal(shape=[4, 4, 64, 128], mean=0, stddev=0.1), name='conv4')
            #     b_conv4 = tf.Variable(tf.random_normal(shape=[128], mean=0, stddev=0.1), name='bias4')
            #     h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
            #     h_conv4_drop = tf.nn.dropout(h_conv4, self.dropout_prob)
            # 
            # with tf.variable_scope('convolution_5') as scope:
            #     W_conv5 = tf.Variable(tf.random_normal(shape=[10, 4, 128, 2], mean=0, stddev=0.1), name='conv5')  # patch 3x3, in size 128, out size 10
            #     b_conv5 = tf.Variable(tf.random_normal(shape=[2], mean=0, stddev=0.1), name='bias5')
            #     h_conv5 = tf.nn.conv2d(h_conv4_drop, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5  # output size 1x1x10
            # 
            # h_dense_mid = tf.layers.dense(h_dense2, units=10, activation=None)
            # h_dense_male = tf.layers.dense(h_dense_mid, units=10, activation=None)
            # h_dense_male_a = tf.nn.softmax(h_dense_male, axis=1)
            # h_dense_female = tf.layers.dense(h_dense_mid, units=10, activation=None)
            # h_dense_female_a = tf.nn.softmax(h_dense_female, axis=1)
            # sum_sc = tf.constant(1, dtype=tf.float32, shape=[10, 1])
            # male_score = tf.matmul(h_dense_mid * h_dense_male_a, sum_sc)
            # female_score = tf.matmul(h_dense_mid * h_dense_female_a, sum_sc)
            # self.result = tf.concat([female_score, male_score], axis=1)
            # print(self.result)
"""

model = Model()
model.train()
