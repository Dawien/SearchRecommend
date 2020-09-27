# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 11:06 上午
# @Author  : Dawein
# @File    : DeepFM.py
# @Software : PyCharm

"""
Paper - DeepFM: A Factorization-Machine based Neural Network for CTR Prediction - (2017)
FM + DNN
FM - 一阶和二阶
DNN - 多层堆叠MLP
"""

import numpy as np
import tensorflow as tf
from collections import defaultdict

class DeepFM():

    def __init__(self, feature_size, field_size, embedding_size, net_params, name="deep_fm"):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.fm_dropout = net_params.fm_dropout
        self.dnn_hidden_size = net_params.dnn_hidden_size
        self.dnn_dropout = net_params.dnn_dropout
        self.use_fm = net_params.use_fm
        self.use_deep = net_params.use_deep
        self.learning_rate = net_params.learning_rate
        self.optimizer_type = net_params.optimizer
        self.l2_reg = net_params.l2_reg
        self.batch_norm = net_params.batch_norm
        self.batch_norm_decay = net_params.batch_norm_decay
        self.verbose = net_params.verbose
        self.loss_type = net_params.loss_type
        self.random_seed = net_params.random_seed
        self.name = name

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.__init_graph()

    def __init_parameters(self):
        # 主要是初始化一些Embedding参数、网络参数

        net_weights = defaultdict()

        ## FM
        p = 1.0 / np.sqrt(self.feature_size + self.embedding_size)
        embeddings = tf.get_variable(name="embeddings",
                                     shape=[self.feature_size, self.embedding_size],
                                     initializer=tf.random_normal_initializer(-p, p),
                                     dtype=tf.float32)
        net_weights["feature_embeddings"] = embeddings

        one_bias = tf.get_variable(name="first_order",
                                   shape=[self.feature_size, 1],
                                   initializer=tf.constant_initializer(0),
                                   dtype=tf.float32)
        net_weights["feature_bias"] = one_bias

        ## Deep: FM - dnn1 - dnn2 - fnn
        num_layers = len(self.dnn_hidden_size)
        input_size = self.field_size * self.embedding_size
        p = 1.0 / np.sqrt(input_size + self.dnn_hidden_size[0])
        layer_0 = tf.get_variable(name="layer_0",
                                  shape=[input_size, self.dnn_hidden_size[0]],
                                  initializer=tf.uniform_unit_scaling_initializer(-p, p),
                                  dtype=tf.float32)
        bias_0 = tf.get_variable(name="bias_0",
                                 shape=[self.dnn_hidden_size[0]],
                                 initializer=tf.constant_initializer(0),
                                 dtype=tf.float32)
        net_weights["layer_0"] = layer_0
        net_weights["bias_0"] = bias_0
        for i in range(1, num_layers):
            p = 1.0 / np.sqrt(self.dnn_hidden_size[i - 1] + self.dnn_hidden_size[i])
            layer_x = tf.get_variable(name="layer_%d" % i,
                                      shape=[self.dnn_hidden_size[i - 1], self.dnn_hidden_size[i]],
                                      initializer=tf.uniform_unit_scaling_initializer(-p, p),
                                      dtype=tf.float32)
            bias_x = tf.get_variable(name="bias_%d" % i,
                                     shape=[self.dnn_hidden_size[i]],
                                     initializer=tf.constant_initializer(0),
                                     dtype=tf.float32)
            net_weights["layer_%d" % i] = layer_x
            net_weights["bias_%d" % i] = bias_x

        ## output
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.dnn_hidden_size[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        else:
            input_size = self.dnn_hidden_size[-1]

        p = 1.0 / np.sqrt(input_size + 1)
        final_w = tf.get_variable(name="final_w",
                                  shape=[input_size, 1],
                                  initializer=tf.uniform_unit_scaling_initializer(-p, p),
                                  dtype=tf.float32)
        final_b = tf.get_variable(name="final_b",
                                  shape=[1],
                                  initializer=tf.constant_initializer(0),
                                  dtype=tf.float32)
        net_weights["final_w"] = final_w
        net_weights["final_b"] = final_b

        return net_weights

    def __init_graph(self):
        print("Start to create all graph.")
        tf.set_random_seed(self.random_seed)

        print("create all weights.")
        self.weights = self.__init_parameters()

        # define placeholder
        self.feat_index = tf.placeholder(name="feat_index",
                                         shape=[None, None],
                                         dtype=tf.int32)
        self.feat_value = tf.placeholder(name="feat_value",
                                         shape=[None, None],
                                         dtype=tf.float32)
        self.label = tf.placeholder(name="label", shape=[None, 1], dtype=tf.float32)
        self.train_phase = tf.placeholder(name="train_phase",
                                          shape=[],
                                          dtype=tf.bool)

        # construct deep-fm-flow
        self.output, _, _, _ = self.construct(feat_index=self.feat_index, feat_value=self.feat_value)

        # loss
        if self.loss_type == "logloss":
            # for classification
            self.output = tf.nn.sigmoid(self.output)
            self.loss = tf.losses.log_loss(self.label, self.output)
        elif self.loss_type == "mse":
            # for regression
            self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.output))
        else:
            raise ValueError("Unknown loss type, should be one of 'logloss/mse'")

        # l2 regularization on weights for preventing over-fitting
        if self.l2_reg > 0:
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["final_w"])
            if self.use_deep:
                for i in range(len(self.dnn_hidden_size)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d" % i])

        # define optimizer
        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9, beta2=0.99, epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss)
        elif self.optimizer_type == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=0.95).minimize(self.loss)
        else:
            raise ValueError("Unknown optimizer type.")

        # print numbers of model's parameters
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("# params: %d" % (total_parameters))


    ## construct network
    def construct(self, feat_index, feat_value):

        # 1. FM
        print("Start to construct the component of fm.")
        self.embedding = tf.nn.embedding_lookup(self.weights["feature_embeddings"], feat_index)
        feat_value = tf.reshape(feat_value, [-1, self.field_size, 1])
        self.embedding = tf.multiply(self.embedding, feat_value) # V .* x

        # the first order - wx
        print("Start to calculate the first order of fm.")
        self.first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], feat_index)
        self.first_order = tf.multiply(self.first_order, feat_value)
        self.first_order = tf.reduce_sum(self.first_order, axis=2) # B * field_size
        self.first_order = tf.cond(self.train_phase,
                                   lambda : tf.nn.dropout(self.first_order, rate=self.fm_dropout[0]),
                                   lambda : self.first_order)

        # the second order
        # sum & square - square & sum
        print("Start to calculate the second order of fm.")
        self.sum_featured_embed = tf.reduce_sum(self.embedding, axis=1) # B * K
        self.sum_featured_embed_square = tf.square(self.sum_featured_embed)

        self.square_featured_embed = tf.square(self.embedding)
        self.square_featured_embed_sum = tf.reduce_sum(self.square_featured_embed, axis=1)

        self.second_order = 0.5 * tf.subtract(self.sum_featured_embed_square, self.square_featured_embed_sum)
        self.second_order = tf.cond(self.train_phase,
                                    lambda : tf.nn.dropout(self.second_order, rate=self.fm_dropout[1]),
                                    lambda : self.second_order)

        # 2. deep network
        print("Start to construct the component of deep.")
        self.deep_input = tf.reshape(self.embedding, shape=(-1, self.field_size*self.embedding_size))
        self.deep_input = tf.cond(self.train_phase,
                                  lambda : tf.nn.dropout(self.deep_input, rate=self.dnn_dropout[0]),
                                  lambda : self.deep_input)

        for i in range(len(self.dnn_hidden_size)):
            self.deep_input = tf.matmul(self.deep_input, self.weights["layer_%d" % i])
            self.deep_input = tf.add(self.deep_input, self.weights["bias_%d" % i])
            self.deep_input = tf.nn.relu(self.deep_input)
            self.deep_input = tf.cond(self.train_phase,
                                      lambda : tf.nn.dropout(self.deep_input, rate=self.dnn_dropout[i+1]),
                                      lambda : self.deep_input)


        # 3. concat
        if self.use_fm and self.use_deep:
            self.concat_input = tf.concat([self.first_order, self.second_order, self.deep_input], axis=1)
        elif self.use_fm:
            self.concat_input = tf.concat([self.first_order, self.second_order], axis=1)
        else:
            self.concat_input = self.deep_input

        self.output = tf.matmul(self.concat_input, self.weights["final_w"])
        self.output = tf.add(self.output, self.weights["final_b"])

        return self.output, self.first_order, self.second_order, self.deep_input