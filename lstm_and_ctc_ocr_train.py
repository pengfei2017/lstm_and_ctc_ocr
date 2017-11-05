#!/usr/bin/env python
# encoding=utf-8
# Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import common, model
import utils

from utils import decode_sparse_tensor

import warpctc_tensorflow

# Some configs
# Accounting the 0th indice +  space + blank label = 28 characters
# num_classes = ord('9') - ord('0') + 1 + 1 + 1
num_classes = common.num_classes
print("num_classes", num_classes)
# Hyper-parameters
print("num_hidden:", common.num_hidden, "num_layers:", common.num_layers)

# THE MAIN CODE!

# 这里加载的是测试数据，也就是每训练一千步就用这里的测试数据计算一下识别出的字符的准确的个数占总个数的百分比
test_inputs, test_targets, test_seq_len = utils.get_data_set('test')
print("Test Data loaded....")


# 用测试数据来验证一下训练的准确率，即识别正确的字符个数占输入字符总个数的百分比
def report_accuracy(decoded_list, test_targets):
    # 报告字符识别的准确率，即n个字符中识别正确的字符个数占总字符个数的百分比
    original_list = decode_sparse_tensor(test_targets)  # 图片标注的字符列表
    detected_list = decode_sparse_tensor(decoded_list)  # 识别出的字符列表
    true_numer = 0
    # print(detected_list)
    if len(original_list) != len(detected_list):
        print("len(original_list) 当前图片的标记字符长度", len(original_list), "len(detected_list) 从当前图片识别出的字符长度",
              len(detected_list), " test and detect length desn't match（从当前图片识别出的字符长度与标记的字符长度不一致）")
        return
    print("T/F: original(length)  标记的字符长度 <-------> detectcted(length)  识别的字符长度")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print("是否识别当前图片：", hit, "  当前图片的标记字符为：", number, "(", len(number), "位 ) <-------> 从当前图片中识别出的字符为：",
              detect_number,
              "(", len(detect_number), "位 )")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy（测试的准确率）:", true_numer * 1.0 / len(original_list) * 1.0)


def train():
    global_step = tf.Variable(0, trainable=False)  # 代表总共训练了多少批，每批训练64个样本（即64张图），每训练一批就是一个迭代，故也可表示位总共迭代了多少次了
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)  # 计算训练的学习率
    logits, inputs, targets, seq_len, W, b = model.get_train_model()  # 这时候还只是定义模型的计算图，只有各个变量的形状，还没有任何计算，所有没有值呢还
    with tf.name_scope('loss'):
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)  # 计算识别的损失率，即误差
        tf.summary.scalar('loss', cost)  # 可视化损失率变化
    with tf.name_scope('train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=common.MOMENTUM).minimize(cost, global_step=global_step)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        # Accuracy: label error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))  # 计算识别的准确率，即精度
        tf.summary.scalar('accuracy', acc)  # 可视化准确率变化

    # Initializate the weights and biases
    init = tf.global_variables_initializer()

    def do_report():
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)  # 这行代码的意思：取出三个变量的值
        report_accuracy(dd, test_targets)
        # decoded_list = decode_sparse_tensor(dd)

    def do_batch():
        # 每训练一批数据（即每迭代一次，也即每训练64个图片）系统会更新一下神经网络模型中各层的weights和biases
        # 每批训练64张图组成的序列
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        b_cost, steps, _ = session.run([cost, global_step, optimizer], feed)
        if steps % 50 == 0:
            result = session.run(merged, feed_dict=feed)  # merged也是需要run的
            writer.add_summary(result, steps)  # result是summary类型的，需要放入writer中，i步数（x轴）
        if steps > 0 and steps % common.REPORT_STEPS == 0:  # 每训练1000批数据（即迭代1000次，即训练10次整个数据集）存一次模型
            do_report()  # 每训练10次整个数据集用测试图片数据计算一次识别出字符个数的准确率
            save_path = saver.save(session, "models/ocr.model", global_step=steps)
            # print(save_path)
        return b_cost, steps  # 返回当前批次的损失率batch_cost和当前批次的编号

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.get_default_graph()._kernel_label_map({"CTCLoss": "WarpCTC"}):
        with tf.Session(config=config) as session:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/", session.graph)
            session.run(init)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            with tf.name_scope('Epoch'):
                for curr_epoch in range(common.num_epochs):  # 对完整数据集（6400张图，即6400个样本）训练10000次
                    curr_epoch_name = 'Epoch%d' % curr_epoch
                    with tf.name_scope(curr_epoch_name):
                        curr_epoch_start = time.time()  # 当前Epoch训练开始的时间
                        # variables = tf.all_variables()
                        # for i in variables:
                        #     print(i.name)

                        print("Epoch（第几个完整数据集的训练）.......", curr_epoch)  # 当前是第几次对完整数据集进行训练
                        train_cost = train_ler = 0
                        for batch in range(
                                common.BATCHES):  # BATCH_SIZE = 64 每批训练64个样本（即64张图），那么训练完一次整个数据集（一个Epoch）需要迭代6400／64=100次，即100个批次；迭代次数就是把整个数据集训练一遍需要几批
                            get_data_start = time.time()  # 当前批次获取数据开始的时间
                            train_inputs, train_targets, train_seq_len = utils.get_data_set('train',
                                                                                            batch * common.BATCH_SIZE,
                                                                                            (
                                                                                                batch + 1) * common.BATCH_SIZE)  # 每批取出64个样本即64张图进行训练
                            # todo tf.summary.image('input', image_shaped_input, 10)
                            get_data_time = time.time() - get_data_start  # 当前批次获取数据花费的时间
                            start = time.time()  # 当前批次训练开始的时间
                            c, steps = do_batch()  # 每训练一批（或者叫每迭代一次，也可叫每训练64张图）会更新一下神经网络模型各层的weights和biases
                            train_cost += c * common.BATCH_SIZE  # 累加每批中所有样本的损失率（也即当前批次64张图乘以当批的平均损失率c）计算当前Epoch（一次整个数据的训练）总的损失率
                            seconds = time.time() - start  # 当前批次训练花费的时间
                            print("Step（在10000个Epoch中的批次编号）:", steps, ", batch seconds（当前批次训练花费的时间）:", seconds,
                                  ", batch get data seconds（当前批次获取数据花费的时间）:", get_data_time, ", batch cost（当前批次的损失率）:",
                                  c)

                        train_cost /= common.TRAIN_SIZE  # 计算当前Epoch（即整个数据集的样本数，也即6400个样本，再即6400张图）的每个样本（也即每张图）的损失率
                        # train_ler /= common.TRAIN_SIZE
                        val_feed = {inputs: train_inputs, targets: train_targets,
                                    seq_len: train_seq_len}  # 用当前Epoch的最后一批样本数据来取

                        # 总共对整个数据集训练10000遍，每遍训练100批，每批训练64个样本，每个样本是一张图片
                        # val_cost指计算cost操作的返回值，是当前的误差率；
                        # val_ler指计算acc操作的返回值，是当前的准确率；
                        # lr指计算learning_rate操作的返回值，是当前的学习率；
                        # steps指计算global_step操作的返回值，是已经训练的总批数；
                        val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step],
                                                                   feed_dict=val_feed)

                        log = "Epoch（对整个数据集进行的第几次训练）{}/{}, （第几批训练）steps = {}, （当前Epoch中平均每张图的损失率）train_cost = {:.3f}, （当前Epoch中平均每张图的精确度）train_ler = {:.3f}, （当前的损失率）val_cost = {:.3f}, （当前的精确度）val_ler = {:.3f}, （当前Epoch花费的时间）time = {:.3f}s, （当前的学习率）learning_rate = {}"
                        print(log.format(curr_epoch + 1, common.num_epochs, steps, train_cost, train_ler, val_cost,
                                         val_ler,
                                         time.time() - curr_epoch_start, lr))
                        with tf.name_scope('train_cost'):
                            tf.summary.scalar('train_cost', train_cost)
                        with tf.name_scope('train_ler'):
                            tf.summary.scalar('train_ler', train_ler)
            writer.close()  # 10000个Epoch训练完时关闭summary的FileWriter


if __name__ == '__main__':
    train()
