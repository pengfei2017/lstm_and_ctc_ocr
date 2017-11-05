import tensorflow as tf


# Utility functions
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                        padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


inputs = tf.placeholder(tf.float32, [64, 256, 64, 1], name='inputs')
print('inputs', inputs.shape)

W_conv1 = weight_variable([5, 5, 1, 48], name='W')
b_conv1 = bias_variable([48], name='b')
h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
print('h_conv1', h_conv1.shape)
h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))
print('h_pool1', h_pool1.shape)

W_conv2 = weight_variable([5, 5, 48, 64], name='W')
b_conv2 = bias_variable([64], name='b')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print('h_conv2', h_conv2.shape)
h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))
print('h_pool2', h_pool2.shape)

W_conv3 = weight_variable([5, 5, 64, 128], name='W')
b_conv3 = bias_variable([128], name='b')
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
print('h_conv3', h_conv3.shape)
h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))
print('h_pool3', h_pool3.shape)

W_fc1 = weight_variable([32 * 8 * 256, 256], name='W')
print('W_fc1', W_fc1.shape)
b_fc1 = bias_variable([256], name='b')
conv_layer_flat = tf.reshape(h_pool3, [-1, 32 * 8 * 256])
print('h_pool3_flat', conv_layer_flat.shape)
features = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
print('features1', features.shape)
shape = tf.shape(features)
print('features1_shape', shape)
features = tf.reshape(features, [shape[0], 256, 1])  # batchsize * outputshape * 1
print('features2', features.shape)
