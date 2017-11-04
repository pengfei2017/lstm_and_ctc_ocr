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
num_epochs = 10000  # 训练次数
num_hidden = 64
num_layers = 1
print("num_hidden:", num_hidden, "num_layers:", num_layers)

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
    global_step = tf.Variable(0, trainable=False)  # todo 这个是什么
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)  # 计算训练的学习率
    logits, inputs, targets, seq_len, W, b = model.get_train_model()
    with tf.name_scope('loss'):
        loss = tf.nn.ctc_loss(targets, logits, seq_len)
        cost = tf.reduce_mean(loss)  # 计算识别的损失率，即误差
    with tf.name_scope('train'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=common.MOMENTUM).minimize(cost, global_step=global_step)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        # Accuracy: label error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))  # 计算识别的准确率，即精度

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
        # 每批训练64张图组成的序列
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        b_cost, steps, _ = session.run([cost, global_step, optimizer], feed)
        if steps > 0 and steps % common.REPORT_STEPS == 0:  # 每一千步存一次模型
            do_report()  # 每一千步计算一次识别出字符个数的准确率
            save_path = saver.save(session, "models/ocr.model", global_step=steps)
            # print(save_path)
        return b_cost, steps

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.get_default_graph()._kernel_label_map({"CTCLoss": "WarpCTC"}):
        with tf.Session(config=config) as session:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/", session.graph)
            session.run(init)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            for curr_epoch in range(num_epochs):  # 训练10000次
                # variables = tf.all_variables()
                # for i in variables:
                #     print(i.name)

                print("Epoch.......", curr_epoch)  # 当前是第几次训练
                train_cost = train_ler = 0
                for batch in range(common.BATCHES):  # 每批次训练100次
                    start = time.time()  # 每一次训练开始的时间
                    train_inputs, train_targets, train_seq_len = utils.get_data_set('train', batch * common.BATCH_SIZE,
                                                                                    (batch + 1) * common.BATCH_SIZE)

                    #
                    #  print("get data time", time.time() - start)
                    start = time.time()
                    c, steps = do_batch()
                    train_cost += c * common.BATCH_SIZE
                    seconds = time.time() - start  # 每一次训练花费的时间
                    print("Step:", steps, ", batch seconds:", seconds, ", cost:", c)

                train_cost /= common.TRAIN_SIZE  # 本批100次训练的总损失率
                # train_ler /= common.TRAIN_SIZE
                val_feed = {inputs: train_inputs,
                            targets: train_targets,
                            seq_len: train_seq_len}

                # 总共训练10000次，每次训练100批，每批训练64步，每步是一张图片
                # val_cost指计算cost操作的返回值，是误差率；
                # val_ler指计算acc操作的返回值，是准确率；
                # lr指计算learning_rate操作的返回值，是学习率；
                # steps指计算global_step操作的返回值，是已经训练的总次数；
                val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

                log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
                print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler,
                                 time.time() - start, lr))


if __name__ == '__main__':
    train()
