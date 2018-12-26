# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import cv2
import sys
sys.path.append(r"F:\pycharm workspace\人工智能\dqn\game\\")
from dqn.game import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque



GAME = 'bird'  # 游戏名称
ACTIONS = 2  # 动作的数量
GAMMA = 0.99  # 下一个状态的折扣率
OBSERVE = 100  # 训练前，先随机的玩游戏，收集10000帧的图片
REPLAY_MEMORY = 50  # 训练前，先随机的玩游戏，保存了50000组数据
EXPLORE = 100  # 训练迭代的次数
INITIAL_EPSILON = 0.0002  # 探索的比例应该是衰减的，所以 INITIAL_EPSILON 表示开始的探索比例
FINAL_EPSILON = 0.0001  # 探索的比例应该是衰减的，所以 FINAL_EPSILON 表示最终的探索比例
BATCH = 4  # batch_size 大小
FRAME_PER_ACTION = 1  # 每一帧的动作
model_dir = r"F:\model\小鸟_dqn"



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)



def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)



def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")



def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")



def createNetwork():
    # 初始化网络权重
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    # 输入层
    s = tf.placeholder("float", [None, 80, 80, 4])
    # 第一个卷积层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 第二个卷积层
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # 第三个卷积层
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    # 第一个全连接层
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    # 第二个全连接层，也可以叫做输出层
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    return s, readout, h_fc1



def trainNetwork(s, readout, h_fc1, sess):
    a = tf.placeholder("float", [None, ACTIONS])  # 动作
    y = tf.placeholder("float", [None])  # 当前状态的最大Q值
    # 动作和卷积神经网络的输出值相乘，计算出当前状态的4帧图像的结果
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    # 损失函数，由 当前状态的结果 和 下一个状态的4帧图像的结果 计算出损失函数
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    # 打开游戏状态与模拟器进行通信
    game_state = game.GameState()
    # 将以前的观察结果存储在 REPLAY_MEMORY 中
    D = deque()

    # 游戏开始时，随便做一个动作来获得第一个状态的信息：
    do_nothing = np.zeros(ACTIONS)  # 初始化动作矩阵
    do_nothing[0] = 1  # 随便做一个动作
    x_t, r_0, terminal = game_state.frame_step(do_nothing)  # 返回值分别表示：第一个状态的1帧图像、当前状态的奖励值、是否结束游戏
    # 把当前状态的1帧图像 resize 到 80*80，然后转换为灰度图
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # 把灰度图二值化(转换为黑白图)，小于这个值，赋值为1；大于这个值，赋值为255
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # 构造出4帧图像
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 保存和加载模型
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("成功加载模型：", checkpoint.model_checkpoint_path)

    # 开始训练
    epsilon = INITIAL_EPSILON  # 探索的比例应该是衰减的，所以 INITIAL_EPSILON 表示开始的探索比例
    t = 0
    while True:
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]  # 计算卷积神经网络的前向传播输出值，然后计算当前状态时的若干Q值集合

        # 探索与开发
        a_t = np.zeros([ACTIONS])  # 初始化动作矩阵
        action_index = 0  # 初始化动作的 index
        if t % FRAME_PER_ACTION == 0:
            # 探索
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)  # 随机选择一个动作的 index
                a_t[random.randrange(ACTIONS)] = 1
            else:
                # 开发
                action_index = np.argmax(readout_t)  # 在当前状态下的若干Q值集合中，选择Q值最大的动作
                a_t[action_index] = 1
        # 衰减探索比例
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选定的动作后，跳到下一个状态，获取：下一个状态的1帧图像、当前状态的奖励值、是否结束游戏
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        # 把下一个状态的1帧图像 resize 到 80*80，然后转换为灰度图
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        # 把灰度图二值化(转换为黑白图)，小于这个值，赋值为1；大于这个值，赋值为255
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))  # shape = 高 * 宽 * 颜色通道
        # 把下一个状态的1帧图像追加到上一状态的最后3帧 之后，x_t1 是下一个状态，s_t[:, :, -3:] 是当前状态的最后3帧
        s_t1 = np.append(s_t[:, :, -3:], x_t1, axis=2)

        # 把信息存储在D中，信息包括：当前状态的4帧图像、当前状态的动作、当前状态的奖励值、下一个状态的4帧图像、是否结束游戏
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        # 每次迭代都要更新旧的状态
        s_t = s_t1

        # 如果观察结束，就训练
        if t > OBSERVE:
            # 从集合 D 中获取 batch 数据
            minibatch = random.sample(D, BATCH)
            s_j_batch = [d[0] for d in minibatch]  # batch 数据：当前状态的4帧图像
            a_batch = [d[1] for d in minibatch]  # batch 数据：当前状态的动作
            r_batch = [d[2] for d in minibatch]  # batch 数据：当前状态的奖励值
            s_j1_batch = [d[3] for d in minibatch]  # batch 数据：下一个状态的4帧图像
            # 构造出下一个状态的神经网络输出  batch 数据
            y_batch = []  # y_batch 用于保存 Q值的 batch 数据
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})  # 计算神经网络的输出值，然后计算下一个状态时的若干Q值集合
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]  # 是否结束游戏（意思是当前状态下执行了一个动作之后，判断游戏是否结束）
                if terminal:
                    y_batch.append(r_batch[i])  # 如果结束游戏，那么 当前状态的Q值 = 当前状态的奖励值
                else:
                    # 如果继续游戏，那么 当前状态的Q值 = 当前状态的奖励值 + 折扣率 * 下一个状态的最大奖励值
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # 训练
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
        # 每多少次迭代保存一次模型
        if t % 10 == 0:
            model_path = os.path.join(model_dir, "model.ckpt-" + str(t))
            saver.save(sess, model_path, global_step = t)
        t = t + 1

        # 打印日志
        state = ""
        if t <= OBSERVE:
            state = "训练前收集数据"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "探索"
        else:
            state = "训练"
        # if t % 100 == 0:
        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t)
        )



def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)




playGame()

