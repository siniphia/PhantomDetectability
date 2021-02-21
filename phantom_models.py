from tensorflow.python.ops.init_ops import he_normal as he
import tensorflow as tf
import ops


class PhantomBasic:
    def __init__(self, name, img, channel, classes):
        with tf.variable_scope(name):
            # feature weights
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1 = tf.get_variable(name='res_w1', shape=[3, 3, channel, 64], initializer=he())

            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2 = tf.get_variable(name='res_w2', shape=[3, 3, 64, 128], initializer=he())

            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 256], initializer=he())
            self.res_w3 = tf.get_variable(name='res_w3', shape=[3, 3, 128, 256], initializer=he())

            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 256, 512], initializer=he())
            self.res_w4 = tf.get_variable(name='res_w4', shape=[3, 3, 256, 512], initializer=he())

            self.proj_w5 = tf.get_variable(name='proj_w5', shape=[1, 1, 512, 1024], initializer=he())
            self.res_w5 = tf.get_variable(name='res_w5', shape=[3, 3, 512, 1024], initializer=he())

            self.proj_w6 = tf.get_variable(name='proj_w6', shape=[1, 1, 1024, 2048], initializer=he())
            self.res_w6 = tf.get_variable(name='res_w6', shape=[3, 3, 1024, 2048], initializer=he())

            self.proj_w7 = tf.get_variable(name='proj_w7', shape=[1, 1, 2048, 4096], initializer=he())
            self.res_w7 = tf.get_variable(name='res_w7', shape=[3, 3, 2048, 4096], initializer=he())

            self.fc_w = tf.get_variable(name='fc_w', shape=[4096, classes], initializer=he())

            # common graphs
            self.res1_1 = ops.resblock_single('res1_1', img, self.res_w1, self.proj_w1)
            self.pool1 = tf.nn.max_pool(self.res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 40,40,64
            self.res2_1 = ops.resblock_single('res2_1', self.pool1, self.res_w2, self.proj_w2)
            self.pool2 = tf.nn.max_pool(self.res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 20,20,128
            self.res3_1 = ops.resblock_single('res3_1', self.pool2, self.res_w3, self.proj_w3)
            self.pool3 = tf.nn.max_pool(self.res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 10,10,256
            self.res4_1 = ops.resblock_single('res4_1', self.pool3, self.res_w4, self.proj_w4)
            self.pool4 = tf.nn.max_pool(self.res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 5,5,512
            self.res5_1 = ops.resblock_single('res5_1', self.pool4, self.res_w5, self.proj_w5)
            self.pool5 = tf.nn.max_pool(self.res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 3,3,1024
            self.res6_1 = ops.resblock_single('res6_1', self.pool5, self.res_w6, self.proj_w6)
            self.pool6 = tf.nn.max_pool(self.res6_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 2,2,2048
            self.res7_1 = ops.resblock_single('res7_1', self.pool6, self.res_w7, self.proj_w7)
            self.pool7 = tf.nn.avg_pool(self.res7_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 1,1,4096
            self.logits = tf.nn.softmax(tf.matmul(tf.squeeze(self.pool7), self.fc_w))
