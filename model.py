#!/usr/bin/env python
# encoding=utf-8
# Created by andy on 2016-07-31 16:57.
import common

__author__ = "andy"
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


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def convolutional_layers(is_training=True):
    """
    Get the convolutional layers of the model.
    """
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, [None, None, common.OUTPUT_SHAPE[0]], name='inputs')
        tf.summary.histogram('inputs/inputs', inputs)  # 原始数据
        with tf.name_scope('input_expand_dims'):
            x_expanded = tf.expand_dims(inputs, 3)
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x_expanded, [-1, common.OUTPUT_SHAPE[0], common.OUTPUT_SHAPE[1], 1])
            tf.summary.image('inputs/input_reshape/images', image_shaped_input,
                             common.BATCH_SIZE)  # 一次显示BATCH_SIZE个图像，即输入样本的个数
        with tf.name_scope('batch_normalization'):  # 对input进行批标准化
            # Batch Normalization（批标准化）
            axes = list(range(len(x_expanded.get_shape()) - 1))
            input_mean, input_var = tf.nn.moments(
                x_expanded,
                axes=axes
                # 想要 normalize 的维度, [0] 代表 batch 维度 # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
            )
            scale = tf.Variable(tf.ones(input_mean.get_shape()))
            shift = tf.Variable(tf.zeros(input_mean.get_shape()))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度

            def mean_var_with_update():
                ema_apply_op = ema.apply([input_mean, input_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(input_mean), tf.identity(input_var)

            # 修改前:mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
            # 修改后:
            mean, var = tf.cond(tf.constant(is_training),  # is_training 的值是 True/False
                                mean_var_with_update,  # 如果是 True, 更新 mean/var
                                lambda: (  # 如果是 False, 返回之前 input_mean/input_var 的Moving Average
                                    ema.average(input_mean),
                                    ema.average(input_var)
                                ))

            # 将修改后的 mean / var 放入下面的公式
            x_expanded = tf.nn.batch_normalization(x_expanded, input_mean, input_var, shift, scale, epsilon)
            tf.summary.scalar('inputs/batch_normalization/input_mean', input_mean)
            tf.summary.scalar('inputs/batch_normalization/input_var', input_var)
            tf.summary.histogram('inputs/batch_normalization/input', x_expanded)
    with tf.name_scope('CNN'):  # CNN中共四层，三次卷积层，一层全链接层
        # First layer
        with tf.name_scope('layer1'):
            layer_name = 'CNN/layer1'
            with tf.name_scope('weights'):
                W_conv1 = weight_variable([5, 5, 1, 48], name='weights')
                variable_summaries(layer_name + '/weights', W_conv1)
            with tf.name_scope('biases'):
                b_conv1 = bias_variable([48], name='biases')
                variable_summaries(layer_name + '/biases', b_conv1)
            h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
            h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

        # Second layer
        with tf.name_scope('layer2'):
            layer_name = 'CNN/layer2'
            with tf.name_scope('weights'):
                W_conv2 = weight_variable([5, 5, 48, 64], name='weights')
                variable_summaries(layer_name + '/weights', W_conv2)
            with tf.name_scope('biases'):
                b_conv2 = bias_variable([64], name='biases')
                variable_summaries(layer_name + '/biases', b_conv2)

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

        # Third layer
        with tf.name_scope('layer3'):
            layer_name = 'CNN/layer3'
            with tf.name_scope('weights'):
                W_conv3 = weight_variable([5, 5, 64, 128], name='weights')
                variable_summaries(layer_name + '/weights', W_conv3)
            with tf.name_scope('biases'):
                b_conv3 = bias_variable([128], name='biases')
                variable_summaries(layer_name + '/biases', b_conv3)
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

        conv_layer_flat = tf.reshape(h_pool3, [-1, 32 * 8 * common.OUTPUT_SHAPE[1]])

        # Densely connected layer
        with tf.name_scope('fc_layer'):
            layer_name = 'CNN/fc_layer'
            with tf.name_scope('weights'):
                W_fc1 = weight_variable([32 * 8 * common.OUTPUT_SHAPE[1], common.OUTPUT_SHAPE[1]], name='weights')
                variable_summaries(layer_name + '/weights', W_fc1)
            with tf.name_scope('biases'):
                b_fc1 = bias_variable([common.OUTPUT_SHAPE[1]], name='biases')
                variable_summaries(layer_name + '/biases', b_fc1)

            fc_layer_W_b = tf.matmul(conv_layer_flat, W_fc1) + b_fc1
            with tf.name_scope('batch_normalization'):  # 对fc_layer的乘积先进行批标准化再进行激活函数
                # Batch Normalization（批标准化）
                axes = list(range(len(fc_layer_W_b.get_shape()) - 1))
                fc_mean, fc_var = tf.nn.moments(
                    fc_layer_W_b,
                    axes=axes
                    # 想要 normalize 的维度, [0] 代表 batch 维度 # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
                )
                scale = tf.Variable(tf.ones(fc_mean.get_shape()))
                shift = tf.Variable(tf.zeros(fc_mean.get_shape()))
                epsilon = 0.001

                ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度

                def mean_var_with_update():
                    ema_apply_op = ema.apply([fc_mean, fc_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(fc_mean), tf.identity(fc_var)

                # 修改前:mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
                # 修改后:
                mean, var = tf.cond(tf.constant(is_training),  # is_training 的值是 True/False
                                    mean_var_with_update,  # 如果是 True, 更新 mean/var
                                    lambda: (  # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                        ema.average(fc_mean),
                                        ema.average(fc_var)
                                    ))

                # 将修改后的 mean / var 放入下面的公式
                fc_layer_W_b = tf.nn.batch_normalization(fc_layer_W_b, fc_mean, fc_var, shift, scale, epsilon)
                tf.summary.scalar(layer_name + 'batch_normalization/fc_mean', fc_mean)
                tf.summary.scalar(layer_name + 'batch_normalization/fc_var', fc_var)
                tf.summary.histogram(layer_name + 'batch_normalization/fc_layer_W_b', fc_layer_W_b)
            features = tf.nn.relu(fc_layer_W_b)
    shape = tf.shape(features)
    features = tf.reshape(features, [shape[0], common.OUTPUT_SHAPE[1], 1])  # batchsize * outputshape * 1
    return inputs, features


def lstm_cell(i, is_training=True):
    layer_name = 'layer%d' % i
    with tf.name_scope(layer_name):
        lstm_cell = tf.contrib.rnn.LSTMCell(common.num_hidden)
    # 在外面包裹一层dropout
    dropout_layer_name = 'dropout%d' % i
    with tf.name_scope(dropout_layer_name):
        if is_training and common.KEEP_PROB < 1:
            tf.summary.scalar('LSTM/' + dropout_layer_name + '/dropout_keep_probability', common.KEEP_PROB)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=common.KEEP_PROB)
    return lstm_cell


def get_train_model(is_training=True):
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs, features = convolutional_layers(is_training)

    with tf.name_scope('dropout'):
        if is_training and common.KEEP_PROB < 1:
            # 在外面包裹一层dropout
            tf.summary.scalar('dropout/dropout_keep_probability', common.KEEP_PROB)
            features = tf.nn.dropout(features, common.KEEP_PROB)
            # print features.get_shape()

            # inputs = tf.placeholder(tf.float32, [None, None, common.OUTPUT_SHAPE[0]])

            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
    with tf.name_scope('LSTM'):  # LSTM中3两次 两个隐藏层和一个全链接层
        targets = tf.sparse_placeholder(tf.int32)

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])

        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        # cell = tf.contrib.rnn.LSTMCell(common.num_hidden, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([lstm_cell(i, is_training) for i in range(0, common.num_layers)],
                                            state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, features, seq_len, dtype=tf.float32)

        with tf.name_scope('fc_layer'):
            shape = tf.shape(features)
            batch_s, max_timesteps = shape[0], shape[1]

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, common.num_hidden])

            # Truncated normal with mean 0 and stdev=0.1
            # Tip: Try another initialization
            # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers

            with tf.name_scope('weights'):
                W = tf.Variable(tf.truncated_normal([common.num_hidden,
                                                     common.num_classes],
                                                    stddev=0.1), name="weights")
                variable_summaries('LSTM/fc_layer/weights', W)
            with tf.name_scope('biases'):
                # Zero initialization
                # Tip: Is tf.zeros_initializer the same?
                b = tf.Variable(tf.constant(0., shape=[common.num_classes]), name="biases")
                variable_summaries('LSTM/fc_layer/biases', b)

            # Doing the affine projection(做仿射投影) 这个就是lstm_ctc要的最终结果[time_step,num_class]=[64*256,12]
            logits = tf.matmul(outputs, W) + b
            with tf.name_scope('batch_normalization'):  # 对fc_layer的乘积先批标准化，再进行激活函数处理
                # Batch Normalization（批标准化）
                axes = list(range(len(logits.get_shape()) - 1))
                lstm_fc_mean, lstm_fc_var = tf.nn.moments(
                    logits,
                    axes=axes
                    # 想要 normalize 的维度, [0] 代表 batch 维度 # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
                )
                scale = tf.Variable(tf.ones(lstm_fc_mean.get_shape()))
                shift = tf.Variable(tf.zeros(lstm_fc_mean.get_shape()))
                epsilon = 0.001

                ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度

                def mean_var_with_update():
                    ema_apply_op = ema.apply([lstm_fc_mean, lstm_fc_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(lstm_fc_mean), tf.identity(lstm_fc_var)

                # 修改前:mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
                # 修改后:
                mean, var = tf.cond(tf.constant(is_training),  # is_training 的值是 True/False
                                    mean_var_with_update,  # 如果是 True, 更新 mean/var
                                    lambda: (  # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                        ema.average(lstm_fc_mean),
                                        ema.average(lstm_fc_var)
                                    ))

                # 将修改后的 mean / var 放入下面的公式
                logits = tf.nn.batch_normalization(logits, lstm_fc_mean, lstm_fc_var, shift, scale, epsilon)
                tf.summary.scalar('LSTM/fc_layer/batch_normalization/lstm_fc_mean', lstm_fc_mean)
                tf.summary.scalar('LSTM/fc_layer/batch_normalization/lstm_fc_var', lstm_fc_var)
                tf.summary.histogram('LSTM/fc_layer/batch_normalization/lstm_fc_layer_W_b', logits)
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, common.num_classes])

    # Time major
    logits = tf.transpose(logits,
                          (1, 0,
                           2))  # transpose (1, 0, 2)理解，本来的第一维，第二维，第三维的顺序是（0，1，2），现在写成（1，0，2）说明是第一维和第二维交换一下位置，第三维还在原来的位置保持不变

    return logits, inputs, targets, seq_len, W, b


if __name__ == '__main__':
    logits, inputs, targets, seq_len, W, b = get_train_model()
