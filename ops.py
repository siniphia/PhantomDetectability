import tensorflow as tf
import numpy as np
from tensorflow.python.ops.init_ops import he_normal as he
from tensorflow.contrib.layers import l2_regularizer as l2
"""
Description
    *_dual : dual conv layers without changing filter number 
    *_proj : same with *_dual but change filter number using 1x1 conv projection
    _resblock : pre-activated resblock (https://arxiv.org/pdf/1603.05027.pdf)
    _seresblock : resblock with se block (https://arxiv.org/pdf/1709.01507.pdf)
    conv : combination of conv + batchnorm + relu with downsampling option
"""


def group_norm(x, g=32, eps=1e-5, name='gn'):
    with tf.variable_scope(name):
        _, h, w, c = x.get_shape().as_list()
        g = min(g, c)
        x = tf.reshape(x, shape=[-1, h, w, g, c // g])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable(name + '_gamma', [1, 1, 1, c], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(name + '_beta', [1, 1, 1, c], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(x, [-1, h, w, c]) * gamma + beta

    return x


def seresblock_proj(name, input, weight1, weight2, weight_proj, filter_num, ratio=4, norm_layer='bn'):
    with tf.variable_scope(name):
        fc1_w = tf.get_variable(name=name + '_fc1_w', shape=[filter_num, int(filter_num / ratio)], initializer=he())
        fc2_w = tf.get_variable(name=name + '_fc2_w', shape=[int(filter_num / ratio), filter_num], initializer=he())

        # residual block
        if norm_layer == 'bn':
            conv = tf.nn.conv2d(input, weight1, strides=(1, 1, 1, 1), padding='SAME')
            bn = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
            af = tf.nn.leaky_relu(bn)
            conv = tf.nn.conv2d(af, weight2, strides=(1, 1, 1, 1), padding='SAME')
            bn = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
            proj = tf.nn.conv2d(input, weight_proj, strides=(1, 1, 1, 1), padding='SAME')
            bn_proj = tf.contrib.layers.batch_norm(proj, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        elif norm_layer == 'gn':
            conv = tf.nn.conv2d(input, weight1, strides=(1, 1, 1, 1), padding='SAME')
            bn = group_norm(conv, name=name + '_gn1')
            af = tf.nn.leaky_relu(bn)
            conv = tf.nn.conv2d(af, weight2, strides=(1, 1, 1, 1), padding='SAME')
            bn = group_norm(conv, name=name + '_gn2')
            proj = tf.nn.conv2d(input, weight_proj, strides=(1, 1, 1, 1), padding='SAME')
            bn_proj = group_norm(proj)

        # squeeze & exitation block
        gap = tf.reduce_mean(bn_proj, axis=[1, 2])  # (B x 1 x C)
        fc1 = tf.reshape(gap, shape=[-1, filter_num])  # (B x C)
        fc1 = tf.matmul(fc1, fc1_w)  # (B x C/R)
        relu = tf.nn.leaky_relu(fc1)  # (B x C/R)
        fc2 = tf.matmul(relu, fc2_w)  # (B x C)
        sig = tf.nn.sigmoid(fc2)  # (B x C)
        descriptor = tf.reshape(sig, shape=[-1, 1, 1, filter_num])

        out = tf.nn.leaky_relu(bn + bn_proj * descriptor)

    return out


def seresblock_dual(name, input, weight1, weight2, filter_num, ratio=4):
    with tf.variable_scope(name):
        fc1_w = tf.get_variable(name=name + '_fc1_w', shape=[filter_num, int(filter_num / ratio)], initializer=he())
        fc2_w = tf.get_variable(name=name + '_fc2_w', shape=[int(filter_num / ratio), filter_num], initializer=he())

        # residual block
        bn1 = tf.contrib.layers.batch_norm(input, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        relu1 = tf.nn.leaky_relu(bn1)
        conv1 = tf.nn.conv2d(relu1, weight1, strides=(1, 1, 1, 1), padding='SAME')
        bn2 = tf.contrib.layers.batch_norm(conv1, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        relu2 = tf.nn.leaky_relu(bn2)
        conv2 = tf.nn.conv2d(relu2, weight2, strides=(1, 1, 1, 1), padding='SAME')

        # squeeze & exitation block
        gap = tf.reduce_mean(conv2, axis=[1, 2])  # (B x 1 x C)
        fc1 = tf.reshape(gap, shape=[-1, filter_num])  # (B x C)
        fc1 = tf.matmul(fc1, fc1_w)  # (B x C/R)
        relu = tf.nn.leaky_relu(fc1)  # (B x C/R)
        fc2 = tf.matmul(relu, fc2_w)  # (B x C)
        sig = tf.nn.sigmoid(fc2)  # (B x C)
        descriptor = tf.reshape(sig, shape=[-1, 1, 1, filter_num])

        out = input + conv2 * descriptor

    return out


def resblock_single(name, input, weight, weight_proj):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(input, weight, strides=(1, 1, 1, 1), padding='SAME')
        bn = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        proj = tf.nn.conv2d(input, weight_proj, strides=(1, 1, 1, 1), padding='SAME')
        bn_proj = tf.contrib.layers.batch_norm(proj, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        out = tf.nn.leaky_relu(bn + bn_proj)
        
    return out


def resblock_proj(name, input, weight1, weight2, weight_proj):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(input, weight1, strides=(1, 1, 1, 1), padding='SAME')
        bn = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        af = tf.nn.leaky_relu(bn)
        conv = tf.nn.conv2d(af, weight2, strides=(1, 1, 1, 1), padding='SAME')
        bn = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        proj = tf.nn.conv2d(input, weight_proj, strides=(1, 1, 1, 1), padding='SAME')
        bn_proj = tf.contrib.layers.batch_norm(proj, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        out = tf.nn.leaky_relu(bn + bn_proj)

    return out


def resblock_dual(name, input, weight1, weight2):
    with tf.variable_scope(name):
        bn1 = tf.contrib.layers.batch_norm(input, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        relu1 = tf.nn.leaky_relu(bn1)
        conv1 = tf.nn.conv2d(relu1, weight1, strides=(1, 1, 1, 1), padding='SAME')
        bn2 = tf.contrib.layers.batch_norm(conv1, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        relu2 = tf.nn.leaky_relu(bn2)
        conv2 = tf.nn.conv2d(relu2, weight2, strides=(1, 1, 1, 1), padding='SAME')

    return conv2 + input


def conv(name, input, weight, downsample=False, norm_layer='bn'):
    with tf.variable_scope(name):
        if downsample:
            conv = tf.nn.conv2d(input, weight, strides=(1, 2, 2, 1), padding='SAME')
        else:
            conv = tf.nn.conv2d(input, weight, strides=(1, 1, 1, 1), padding='SAME')

        if norm_layer == 'bn':
            conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.9, zero_debias_moving_mean=True)
        elif norm_layer == 'gn':
            conv = group_norm(conv)
        conv = tf.nn.relu(conv)

    return conv


def grad_cam(loss, conv, image_size):
    grads = tf.gradients(loss, conv)[0]  # normalize the gradient for a single layer
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    conv_shape = conv.shape
    conv = tf.reshape(conv, [-1, np.prod(conv_shape[1:-1]), conv_shape[-1]])  # N H*W C
    weights = tf.expand_dims(tf.reduce_max(norm_grads, axis=[1, 2]), axis=-1)  # N C 1

    class_activation_map = tf.matmul(conv, weights)  # N H*W 1
    class_activation_map = tf.reshape(class_activation_map, [-1, conv_shape[1], conv_shape[2], 1])

    # zero-out using relu (to obtain positive effect only) and resizing heatmap
    class_activation_map = tf.nn.relu(class_activation_map)
    class_activation_map = tf.image.resize_bilinear(images=class_activation_map, size=image_size)

    return class_activation_map


# train : 102, val : 80, test : 102
# model for 80x80 images
class PhantomNet:
    def __init__(self, name, image, channel, classes):
        with tf.variable_scope(name):
            self.conv_w1 = tf.get_variable(name='conv_w1', shape=[3, 3, channel, 64], initializer=he())
            self.conv_w2 = tf.get_variable(name='conv_w2', shape=[3, 3, 64, 128], initializer=he())
            self.conv_w3 = tf.get_variable(name='conv_w3', shape=[3, 3, 128, 256], initializer=he())
            self.conv_w4 = tf.get_variable(name='conv_w4', shape=[3, 3, 256, 320], initializer=he())
            self.fc_w = tf.get_variable(name='fc_w', shape=[320*2*2, classes])

            self.conv1 = conv('conv1', image, self.conv_w1, False)
            self.pool1 = tf.nn.max_pool(self.conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 40,40,64
            self.conv2 = conv('conv2', self.pool1, self.conv_w2, False)
            self.pool2 = tf.nn.max_pool(self.conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 20,20,128
            self.conv3 = conv('conv3', self.pool2, self.conv_w3, False)
            self.pool3 = tf.nn.max_pool(self.conv3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 10,10,256
            self.conv4 = conv('conv4', self.pool3, self.conv_w4, False)
            self.avg_pool = tf.nn.avg_pool(self.conv4, ksize=(1, 5, 5, 1), strides=(1, 5, 5, 1), padding='SAME')  # 2,2,320
            self.flat = tf.reshape(self.avg_pool, shape=[-1, 320*2*2])
            self.logits = tf.matmul(self.flat, self.fc_w)


class SeResNet:
    def __init__(self, name, lt_img, rt_img, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channel, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 196], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 196], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 196, 196], initializer=he())
            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 196, 256], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 196, 256], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 256, 256], initializer=he())
            self.proj_w5 = tf.get_variable(name='proj_w5', shape=[1, 1, 256, 320], initializer=he())
            self.res_w5_1 = tf.get_variable(name='res_w5_1', shape=[3, 3, 256, 320], initializer=he())
            self.res_w5_2 = tf.get_variable(name='res_w5_2', shape=[3, 3, 320, 320], initializer=he())
            self.conv_w = tf.get_variable(name='conv_w', shape=[3, 3, 320, 320], initializer=he())
            self.fc_w = tf.get_variable(name='fc_w', shape=[1280, classes], initializer=he())

            # 2 - Graphs
            self.res1_1 = seresblock_proj('res1_1', lt_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.pool1 = tf.nn.max_pool(self.res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 128,128,64
            self.res2_1 = seresblock_proj('res2_1', self.pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.pool2 = tf.nn.max_pool(self.res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 64,64,128
            self.res3_1 = seresblock_proj('res3_1', self.pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196)
            self.pool3 = tf.nn.max_pool(self.res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 32,32,196
            self.res4_1 = seresblock_proj('res4_1', self.pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256)
            self.pool4 = tf.nn.max_pool(self.res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 16,16,256
            self.res5_1 = seresblock_proj('res5_1', self.pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 320)
            self.pool5 = tf.nn.max_pool(self.res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 8,8,320
            self.conv_last = tf.nn.conv2d(self.pool5, self.conv_w, strides=(1, 1, 1, 1), padding='SAME')
            self.bn_last = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.conv_last, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.avg = tf.nn.max_pool(self.bn_last, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')  # 2,2,320
            self.flat = tf.reshape(self.avg, shape=[-1, 1280])
            self.logits = tf.matmul(self.flat, self.fc_w)


class BCMNet:
    def __init__(self, name, src_img, dst_img, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channel, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 196], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 196], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 196, 196], initializer=he())
            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 196, 256], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 196, 256], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 256, 256], initializer=he())
            self.proj_w5 = tf.get_variable(name='proj_w5', shape=[1, 1, 256, 320], initializer=he())
            self.res_w5_1 = tf.get_variable(name='res_w5_1', shape=[3, 3, 256, 320], initializer=he())
            self.res_w5_2 = tf.get_variable(name='res_w5_2', shape=[3, 3, 320, 320], initializer=he())

            self.conv_w1 = tf.get_variable(name='conv_w1', shape=[3, 3, 64, 128], initializer=he())
            self.conv_w2 = tf.get_variable(name='conv_w2', shape=[3, 3, 128, 320], initializer=he())

            self.bcm_fc = tf.get_variable(name='bcm_fc', shape=[1280, classes], initializer=he())

            # 2 - Graphs
            self.lt_res1_1 = seresblock_proj('lt_res1_1', src_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.lt_pool1 = tf.nn.max_pool(self.lt_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 128,128,64
            self.lt_res2_1 = seresblock_proj('lt_res2_1', self.lt_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.lt_pool2 = tf.nn.max_pool(self.lt_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 64,64,128
            self.lt_res3_1 = seresblock_proj('lt_res3_1', self.lt_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196)
            self.lt_pool3 = tf.nn.max_pool(self.lt_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 32,32,196
            self.lt_res4_1 = seresblock_proj('lt_res4_1', self.lt_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256)
            self.lt_pool4 = tf.nn.max_pool(self.lt_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 16,16,256
            self.lt_res5_1 = seresblock_proj('lt_res5_1', self.lt_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 320)
            self.lt_pool5 = tf.nn.max_pool(self.lt_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 8,8,320

            # 2 - Graphs
            self.rt_res1_1 = seresblock_proj('rt_res1_1', dst_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.rt_pool1 = tf.nn.max_pool(self.rt_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 128,128,64
            self.rt_res2_1 = seresblock_proj('rt_res2_1', self.rt_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.rt_pool2 = tf.nn.max_pool(self.rt_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 64,64,128
            self.rt_res3_1 = seresblock_proj('rt_res3_1', self.rt_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196)
            self.rt_pool3 = tf.nn.max_pool(self.rt_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 32,32,196
            self.rt_res4_1 = seresblock_proj('rt_res4_1', self.rt_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256)
            self.rt_pool4 = tf.nn.max_pool(self.rt_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 16,16,256
            self.rt_res5_1 = seresblock_proj('rt_res5_1', self.rt_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 320)
            self.rt_pool5 = tf.nn.max_pool(self.rt_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 8,8,320

            self.bcm = self.get_bcm()  # (?,8,8,64)

            self.bcm_conv1 = tf.nn.conv2d(self.bcm, self.conv_w1, strides=(1, 1, 1, 1), padding='SAME')
            self.bcm_conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.bcm_conv1, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.bcm_pool1 = tf.nn.max_pool(self.bcm_conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME') # (?,4,4,128)
            self.bcm_conv2 = tf.nn.conv2d(self.bcm_pool1, self.conv_w2, strides=(1, 1, 1, 1), padding='SAME')
            self.bcm_conv2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.bcm_conv2, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.bcm_pool2 = tf.nn.max_pool(self.bcm_conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,2,\2,320)
            self.bcm_flat = tf.reshape(self.bcm_pool2, shape=[-1, 1280])
            self.logits = tf.matmul(self.bcm_flat, self.bcm_fc)

    def get_bcm(self):
        with tf.variable_scope('bcm'):
            src_flat = tf.reshape(self.lt_pool5, shape=[-1, 320, 8 * 8])
            dst_flat = tf.reshape(self.rt_pool5, shape=[-1, 320, 8 * 8])

            # normalize from 0 to 1
            src_flat = tf.math.l2_normalize(src_flat, axis=1)
            dst_flat = tf.math.l2_normalize(dst_flat, axis=1)

            for i in range(8 * 8):
                src_elem = src_flat[:, :, i:i + 1]
                gcm_elem = tf.reduce_sum(dst_flat * src_elem, axis=1)
                gcm_elem = tf.maximum(tf.reshape(gcm_elem, shape=[-1, 8, 8, 1]), [0])
                if i == 0:
                    gcm = gcm_elem
                else:
                    gcm = tf.concat([gcm, gcm_elem], axis=3)

        return gcm


class EnsembleBCMNet:
    def __init__(self, name, src_img, dst_img, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channel, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 196], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 196], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 196, 196], initializer=he())
            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 196, 256], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 196, 256], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 256, 256], initializer=he())
            self.proj_w5 = tf.get_variable(name='proj_w5', shape=[1, 1, 256, 320], initializer=he())
            self.res_w5_1 = tf.get_variable(name='res_w5_1', shape=[3, 3, 256, 320], initializer=he())
            self.res_w5_2 = tf.get_variable(name='res_w5_2', shape=[3, 3, 320, 320], initializer=he())

            self.proj_w6 = tf.get_variable(name='proj_w6', shape=[1, 1, 64, 128], initializer=he())
            self.res_w6_1 = tf.get_variable(name='res_w6_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w6_2 = tf.get_variable(name='res_w6_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w7 = tf.get_variable(name='proj_w7', shape=[1, 1, 128, 320], initializer=he())
            self.res_w7_1 = tf.get_variable(name='res_w7_1', shape=[3, 3, 128, 320], initializer=he())
            self.res_w7_2 = tf.get_variable(name='res_w7_2', shape=[3, 3, 320, 320], initializer=he())
            self.conv_w = tf.get_variable(name='conv_w', shape=[3, 3, 320, 320], initializer=he())

            self.fc_w = tf.get_variable(name='fc_w', shape=[1280 * 2, classes], initializer=he())

            # lt maxil
            self.lt_res1_1 = seresblock_proj('lt_res1_1', src_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.lt_pool1 = tf.nn.max_pool(self.lt_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 128,128,64
            self.lt_res2_1 = seresblock_proj('lt_res2_1', self.lt_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.lt_pool2 = tf.nn.max_pool(self.lt_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 64,64,128
            self.lt_res3_1 = seresblock_proj('lt_res3_1', self.lt_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196)
            self.lt_pool3 = tf.nn.max_pool(self.lt_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 32,32,196
            self.lt_res4_1 = seresblock_proj('lt_res4_1', self.lt_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256)
            self.lt_pool4 = tf.nn.max_pool(self.lt_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 16,16,256
            self.lt_res5_1 = seresblock_proj('lt_res5_1', self.lt_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 320)
            self.lt_pool5 = tf.nn.max_pool(self.lt_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 8,8,320

            # rt maxil
            self.rt_res1_1 = seresblock_proj('rt_res1_1', dst_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.rt_pool1 = tf.nn.max_pool(self.rt_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 128,128,64
            self.rt_res2_1 = seresblock_proj('rt_res2_1', self.rt_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.rt_pool2 = tf.nn.max_pool(self.rt_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 64,64,128
            self.rt_res3_1 = seresblock_proj('rt_res3_1', self.rt_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196)
            self.rt_pool3 = tf.nn.max_pool(self.rt_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 32,32,196
            self.rt_res4_1 = seresblock_proj('rt_res4_1', self.rt_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256)
            self.rt_pool4 = tf.nn.max_pool(self.rt_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 16,16,256
            self.rt_res5_1 = seresblock_proj('rt_res5_1', self.rt_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 320)
            self.rt_pool5 = tf.nn.max_pool(self.rt_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 8,8,320

            # bcm
            self.bcm = self.get_bcm()  # (?,8,8,64)
            self.bcm_res1 = seresblock_proj('bcm_res1', self.bcm, self.res_w6_1, self.res_w6_2, self.proj_w6, 128)
            self.bcm_pool1 = tf.nn.max_pool(self.bcm_res1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,4,4,128)
            self.bcm_res2 = seresblock_proj('bcm_res2', self.bcm_pool1, self.res_w7_1, self.res_w7_2, self.proj_w7, 320)
            self.bcm_pool2 = tf.nn.max_pool(self.bcm_res2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,2,2,320)
            self.bcm_flat = tf.reshape(self.bcm_pool2, shape=[-1, 1280])

            # lt maxil flat
            self.lt_conv_last = tf.nn.conv2d(self.lt_pool5, self.conv_w, strides=(1, 1, 1, 1), padding='SAME')
            self.lt_bn_last = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.lt_conv_last, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.lt_avg = tf.nn.max_pool(self.lt_bn_last, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')
            self.lt_flat = tf.reshape(self.lt_avg, shape=[-1, 1280])

            self.flat = tf.concat([self.bcm_flat, self.lt_flat], axis=1)
            self.logits = tf.matmul(self.flat, self.fc_w)

    def get_bcm(self):
        with tf.variable_scope('bcm'):
            src_flat = tf.reshape(self.lt_pool5, shape=[-1, 320, 8 * 8])
            dst_flat = tf.reshape(self.rt_pool5, shape=[-1, 320, 8 * 8])

            # normalize from 0 to 1
            src_flat = tf.math.l2_normalize(src_flat, axis=1)
            dst_flat = tf.math.l2_normalize(dst_flat, axis=1)

            for i in range(8 * 8):
                src_elem = src_flat[:, :, i:i + 1]
                gcm_elem = tf.reduce_sum(dst_flat * src_elem, axis=1)
                gcm_elem = tf.maximum(tf.reshape(gcm_elem, shape=[-1, 8, 8, 1]), [0])
                if i == 0:
                    gcm = gcm_elem
                else:
                    gcm = tf.concat([gcm, gcm_elem], axis=3)

        return gcm


class SeResNet448:
    def __init__(self, name, tar_img, ref_img, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channel, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 192], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 192], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 192, 192], initializer=he())
            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 192, 256], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 192, 256], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 256, 256], initializer=he())

            self.conv_w1 = tf.get_variable(name='conv_w1', shape=[3, 3, 256, 512], initializer=he())
            self.conv_w2 = tf.get_variable(name='conv_w2', shape=[3, 3, 512, 512], initializer=he())
            self.fc_w = tf.get_variable(name='fc_w', shape=[2048, classes], initializer=he())

            # 2 - Graphs
            self.conv_1 = seresblock_proj('conv_1', tar_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64, norm_layer='bn')
            self.pool_1 = tf.nn.max_pool(self.conv_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 224,224,64
            self.conv_2 = seresblock_proj('conv_2', self.pool_1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128, norm_layer='bn')
            self.pool_2 = tf.nn.max_pool(self.conv_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 112,112,128
            self.conv_3 = seresblock_proj('conv_3', self.pool_2, self.res_w3_1, self.res_w3_2, self.proj_w3, 192, norm_layer='bn')
            self.pool_3 = tf.nn.max_pool(self.conv_3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 56,56,192
            self.conv_4 = seresblock_proj('conv_4', self.pool_3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256, norm_layer='bn')
            self.pool_4 = tf.nn.max_pool(self.conv_4, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 28,28,320
            self.conv_5 = conv('conv_5', self.pool_4, self.conv_w1, norm_layer='bn')
            self.pool_5 = tf.nn.max_pool(self.conv_5, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 14,14,512
            self.conv_6 = conv('conv_6', self.pool_5, self.conv_w2, norm_layer='bn')
            self.pool_6 = tf.nn.avg_pool(self.conv_6, ksize=(1, 7, 7, 1), strides=(1, 7, 7, 1), padding='SAME')  # 2,2,512
            self.flat = tf.reshape(self.pool_6, shape=[-1, 2048])
            self.logits = tf.matmul(self.flat, self.fc_w)  # 2048


class ButterflyNet448:
    def __init__(self, name, src_img, dst_img, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channel, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 196], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 196], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 196, 196], initializer=he())
            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 196, 256], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 196, 256], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 256, 256], initializer=he())

            self.conv_w1 = tf.get_variable(name='conv_w1', shape=[3, 3, 512, 512], initializer=he())
            self.conv_w2 = tf.get_variable(name='conv_w2', shape=[3, 3, 512, 512], initializer=he())
            self.fc_w = tf.get_variable(name='fc_w', shape=[2048, classes], initializer=he())

            # 2 - Graphs
            self.tar_res1_1 = seresblock_proj('tar_res1_1', src_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64, norm_layer='bn')
            self.tar_pool1 = tf.nn.max_pool(self.tar_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 224,224,64
            self.tar_res2_1 = seresblock_proj('tar_res2_1', self.tar_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128, norm_layer='bn')
            self.tar_pool2 = tf.nn.max_pool(self.tar_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 112,112,128
            self.tar_res3_1 = seresblock_proj('tar_res3_1', self.tar_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196, norm_layer='bn')
            self.tar_pool3 = tf.nn.max_pool(self.tar_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 56,56,196
            self.tar_res4_1 = seresblock_proj('tar_res4_1', self.tar_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256, norm_layer='bn')
            self.tar_pool4 = tf.nn.max_pool(self.tar_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 28,28,256

            self.ref_res1_1 = seresblock_proj('ref_res1_1', dst_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64, norm_layer='bn')
            self.ref_pool1 = tf.nn.max_pool(self.ref_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 224,224,64
            self.ref_res2_1 = seresblock_proj('ref_res2_1', self.ref_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128, norm_layer='bn')
            self.ref_pool2 = tf.nn.max_pool(self.ref_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 112,112,128
            self.ref_res3_1 = seresblock_proj('ref_res3_1', self.ref_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196, norm_layer='bn')
            self.ref_pool3 = tf.nn.max_pool(self.ref_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 56,56,196
            self.ref_res4_1 = seresblock_proj('ref_res4_1', self.ref_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256, norm_layer='bn')
            self.ref_pool4 = tf.nn.max_pool(self.ref_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 28,28,256

            self.concat = tf.concat([self.tar_pool4, self.ref_pool4], axis=3)  # 28,28,512

            self.conv_5 = conv('conv_5', self.concat, self.conv_w1, norm_layer='bn')
            self.pool_5 = tf.nn.max_pool(self.conv_5, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 14,14,512
            self.conv_6 = conv('conv_6', self.pool_5, self.conv_w2, norm_layer='bn')
            self.pool_6 = tf.nn.avg_pool(self.conv_6, ksize=(1, 7, 7, 1), strides=(1, 7, 7, 1), padding='SAME')  # 2,2,512
            self.flat = tf.reshape(self.pool_6, shape=[-1, 2048])
            self.logits = tf.matmul(self.flat, self.fc_w)


class ButterflyNet:
    def __init__(self, name, src_img, dst_img, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channel, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 196], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 196], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 196, 196], initializer=he())
            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 196, 256], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 196, 256], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 256, 256], initializer=he())

            self.conv_w1 = tf.get_variable(name='conv_w1', shape=[3, 3, 512, 512], initializer=he())
            self.conv_w2 = tf.get_variable(name='conv_w2', shape=[3, 3, 512, 1024], initializer=he())
            self.fc = tf.get_variable(name='bcm_fc', shape=[1024, classes], initializer=he())

            # 2 - Graphs
            self.lt_res1_1 = seresblock_proj('lt_res1_1', src_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.lt_pool1 = tf.nn.max_pool(self.lt_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 128,128,64
            self.lt_res2_1 = seresblock_proj('lt_res2_1', self.lt_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.lt_pool2 = tf.nn.max_pool(self.lt_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 64,64,128
            self.lt_res3_1 = seresblock_proj('lt_res3_1', self.lt_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196)
            self.lt_pool3 = tf.nn.max_pool(self.lt_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 32,32,196
            self.lt_res4_1 = seresblock_proj('lt_res4_1', self.lt_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256)
            self.lt_pool4 = tf.nn.max_pool(self.lt_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 16,16,256

            self.rt_res1_1 = seresblock_proj('rt_res1_1', dst_img, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.rt_pool1 = tf.nn.max_pool(self.rt_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 128,128,64
            self.rt_res2_1 = seresblock_proj('rt_res2_1', self.rt_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.rt_pool2 = tf.nn.max_pool(self.rt_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 64,64,128
            self.rt_res3_1 = seresblock_proj('rt_res3_1', self.rt_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 196)
            self.rt_pool3 = tf.nn.max_pool(self.rt_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 32,32,196
            self.rt_res4_1 = seresblock_proj('rt_res4_1', self.rt_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 256)
            self.rt_pool4 = tf.nn.max_pool(self.rt_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 16,16,256

            self.concat = tf.concat([self.lt_pool4, self.rt_pool4], axis=3)  # 16,16,512

            self.conv1 = tf.nn.conv2d(self.concat, self.conv_w1, strides=(1, 1, 1, 1), padding='SAME')
            self.conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.conv1, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.pool1 = tf.nn.max_pool(self.conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 8,8,512
            self.conv2 = tf.nn.conv2d(self.pool1, self.conv_w2, strides=(1, 1, 1, 1), padding='SAME')
            self.conv2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.conv2, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.pool2 = tf.nn.avg_pool(self.conv2, ksize=(1, 8, 8, 1), strides=(1, 8, 8, 1), padding='SAME')  # 1,1,1024
            self.flat = tf.reshape(self.pool2, shape=[-1, 1024])
            self.logits = tf.matmul(self.flat, self.fc)


class MultiSinusDet448:
    def __init__(self, name, img_tar, img_ref, channel, out_dim):
        with tf.variable_scope(name):
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channel, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channel, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 256], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 256], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 256, 256], initializer=he())
            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 256, 320], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 256, 320], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 320, 320], initializer=he())
            self.proj_w5 = tf.get_variable(name='proj_w5', shape=[1, 1, 320, 512], initializer=he())
            self.res_w5_1 = tf.get_variable(name='res_w5_1', shape=[3, 3, 320, 512], initializer=he())
            self.res_w5_2 = tf.get_variable(name='res_w5_2', shape=[3, 3, 512, 512], initializer=he())
            self.conv_w = tf.get_variable(name='conv_w', shape=[3, 3, 512, 512], initializer=he())
            self.fc_w = tf.get_variable(name='fc_w', shape=[2048, out_dim], initializer=he())

            # 2 - Target Graph
            self.tar_res1_1 = seresblock_proj('tar_res1_1', img_tar, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.tar_pool1 = tf.nn.max_pool(self.tar_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 224,224,64
            self.tar_res2_1 = seresblock_proj('tar_res2_1', self.tar_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.tar_pool2 = tf.nn.max_pool(self.tar_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 112,112,128
            self.tar_res3_1 = seresblock_proj('tar_res3_1', self.tar_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 256)
            self.tar_pool3 = tf.nn.max_pool(self.tar_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 56,56,256
            self.tar_res4_1 = seresblock_proj('tar_res4_1', self.tar_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 320)
            self.tar_pool4 = tf.nn.max_pool(self.tar_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 28,28,320
            self.tar_res5_1 = seresblock_proj('tar_res5_1', self.tar_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 512)
            self.tar_pool5 = tf.nn.max_pool(self.tar_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 14,14,512
            
            self.tar_conv = tf.nn.conv2d(self.tar_pool5, self.conv_w, strides=(1, 1, 1, 1), padding='SAME')  # 14,14,512
            self.tar_bn = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.tar_conv, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.tar_avg = tf.nn.avg_pool(self.tar_bn, ksize=(1, 7, 7, 1), strides=(1, 7, 7, 1), padding='SAME')  # 2,2,512
            self.tar_flat = tf.reshape(self.tar_avg, shape=[-1, 2048])  # 2048
            self.tar_logits = tf.matmul(self.tar_flat, self.fc_w)  # [batch, out_dim]

            # 3 - Reference Graph
            self.ref_res1_1 = seresblock_proj('ref_res1_1', img_ref, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.ref_pool1 = tf.nn.max_pool(self.ref_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 224,224,64
            self.ref_res2_1 = seresblock_proj('ref_res2_1', self.ref_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.ref_pool2 = tf.nn.max_pool(self.ref_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 112,112,128
            self.ref_res3_1 = seresblock_proj('ref_res3_1', self.ref_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 256)
            self.ref_pool3 = tf.nn.max_pool(self.ref_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 56,56,256
            self.ref_res4_1 = seresblock_proj('ref_res4_1', self.ref_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 320)
            self.ref_pool4 = tf.nn.max_pool(self.ref_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 28,28,320
            self.ref_res5_1 = seresblock_proj('ref_res5_1', self.ref_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 512)
            self.ref_pool5 = tf.nn.max_pool(self.ref_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 14,14,512
            
            self.conv_last = tf.nn.conv2d(self.ref_pool5, self.conv_w, strides=(1, 1, 1, 1), padding='SAME')  # 14,14,512
            self.ref_bn = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.conv_last, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.ref_avg = tf.nn.avg_pool(self.ref_bn, ksize=(1, 7, 7, 1), strides=(1, 7, 7, 1), padding='SAME')  # 2,2,512
            self.ref_flat = tf.reshape(self.ref_avg, shape=[-1, 2048])  # 2048
            self.ref_logits = tf.matmul(self.ref_flat, self.fc_w)  # [batch, out_dim]

            self.logits = tf.stack([self.tar_logits, self.ref_logits], axis=0)  # [2, batch, out_dim]


class MultiSinusCls448:
    def __init__(self, name, img_wat, img_cal, channels, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.proj_w1 = tf.get_variable(name='proj_w1', shape=[1, 1, channels, 64], initializer=he())
            self.res_w1_1 = tf.get_variable(name='res_w1_1', shape=[3, 3, channels, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='res_w1_2', shape=[3, 3, 64, 64], initializer=he())

            self.proj_w2 = tf.get_variable(name='proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='res_w2_2', shape=[3, 3, 128, 128], initializer=he())

            self.proj_w3 = tf.get_variable(name='proj_w3', shape=[1, 1, 128, 256], initializer=he())
            self.res_w3_1 = tf.get_variable(name='res_w3_1', shape=[3, 3, 128, 256], initializer=he())
            self.res_w3_2 = tf.get_variable(name='res_w3_2', shape=[3, 3, 256, 256], initializer=he())

            self.proj_w4 = tf.get_variable(name='proj_w4', shape=[1, 1, 256, 320], initializer=he())
            self.res_w4_1 = tf.get_variable(name='res_w4_1', shape=[3, 3, 256, 320], initializer=he())
            self.res_w4_2 = tf.get_variable(name='res_w4_2', shape=[3, 3, 320, 320], initializer=he())

            self.proj_w5 = tf.get_variable(name='proj_w5', shape=[1, 1, 320, 512], initializer=he())
            self.res_w5_1 = tf.get_variable(name='res_w5_1', shape=[3, 3, 320, 512], initializer=he())
            self.res_w5_2 = tf.get_variable(name='res_w5_2', shape=[3, 3, 512, 512], initializer=he())

            self.conv_w = tf.get_variable(name='conv_w', shape=[3, 3, 1024, 1024], initializer=he())
            self.fc_w = tf.get_variable(name='fc_w', shape=[4096, classes], initializer=he())

            # 3 - Graphs
            self.wat_res1_1 = seresblock_proj('wat_res1_1', img_wat, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.wat_pool1 = tf.nn.max_pool(self.wat_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 224,224,64
            self.wat_res2_1 = seresblock_proj('wat_res2_1', self.wat_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.wat_pool2 = tf.nn.max_pool(self.wat_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 112,112,128
            self.wat_res3_1 = seresblock_proj('wat_res3_1', self.wat_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 256)
            self.wat_pool3 = tf.nn.max_pool(self.wat_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 56,56,256
            self.wat_res4_1 = seresblock_proj('wat_res4_1', self.wat_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 320)
            self.wat_pool4 = tf.nn.max_pool(self.wat_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 28,28,320
            self.wat_res5_1 = seresblock_proj('wat_res5_1', self.wat_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 512)
            self.wat_pool5 = tf.nn.max_pool(self.wat_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 14,14,512

            # 3 - Graphs
            self.cal_res1_1 = seresblock_proj('cal_res1_1', img_cal, self.res_w1_1, self.res_w1_2, self.proj_w1, 64)
            self.cal_pool1 = tf.nn.max_pool(self.cal_res1_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 224,224,64
            self.cal_res2_1 = seresblock_proj('cal_res2_1', self.cal_pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.cal_pool2 = tf.nn.max_pool(self.cal_res2_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 112,112,128
            self.cal_res3_1 = seresblock_proj('cal_res3_1', self.cal_pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 256)
            self.cal_pool3 = tf.nn.max_pool(self.cal_res3_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 56,56,256
            self.cal_res4_1 = seresblock_proj('cal_res4_1', self.cal_pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 320)
            self.cal_pool4 = tf.nn.max_pool(self.cal_res4_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 28,28,320
            self.cal_res5_1 = seresblock_proj('cal_res5_1', self.cal_pool4, self.res_w5_1, self.res_w5_2, self.proj_w5, 512)
            self.cal_pool5 = tf.nn.max_pool(self.cal_res5_1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # 14,14,512

            self.concat = tf.concat([self.wat_pool5, self.cal_pool5], axis=-1)  # 14,14,1024
            self.conv_last = tf.nn.conv2d(self.concat, self.conv_w, strides=(1, 1, 1, 1), padding='SAME')  # 14,14,1024
            self.bn_last = tf.nn.leaky_relu(tf.contrib.layers.batch_norm(self.conv_last, updates_collections=None, decay=0.9, zero_debias_moving_mean=True))
            self.avg_last = tf.nn.avg_pool(self.bn_last, ksize=(1, 7, 7, 1), strides=(1, 7, 7, 1), padding='SAME')  # 2,2,1024
            self.flat_last = tf.reshape(self.avg_last, shape=[-1, 4096])  # 4096
            self.logits = tf.matmul(self.flat_last, self.fc_w)  # 4096 -> classes

