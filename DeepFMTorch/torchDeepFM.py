# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 9:21 上午
# @Author  : Dawein
# @File    : torchDeepFM.py
# @Software : PyCharm

"""
pytorch code for realizing DeepFM
- paper: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction - (2017)
- FM: 一阶和二阶特征
- Deep: 高阶交互特征
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(SubLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        self.W = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b = nn.Parameter(torch.Tensor(output_size))
        p = 1.0 / np.sqrt(input_size + output_size)
        self.W.data.uniform_(-p, p)
        self.b.data.fill_(0)

    def forward(self, x):
        y = F.linear(x, self.W) + self.b
        y = F.relu(y)
        y = F.dropout(y, self.dropout, training=self.training)
        return y

class DeepFM(nn.Module):
    
    def __init__(self, feature_size, field_size, params):
        super(DeepFM, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = params.embedding_size
        self.fm_dropout = params.fm_dropout
        self.dnn_hidden_size = params.dnn_hidden_size
        self.dnn_dropout = params.dnn_dropout
        self.use_fm = params.use_fm
        self.use_deep = params.use_deep
        self.__init_parameters()

    def __init_parameters(self):
        # 初始化Embedding、网络参数
        ## FM
        p = 1.0 / np.sqrt(self.feature_size + self.embedding_size)
        self.feature_embedding = nn.Parameter(torch.Tensor(self.feature_size, self.embedding_size))
        self.feature_embedding.data.uniform_(-p, p)

        p = 1.0 / np.sqrt(self.feature_size + 1)
        self.feature_bias = nn.Parameter(torch.Tensor(self.feature_size, 1))
        self.feature_bias.data.uniform_(-p, p)

        ## Deep
        self.deep_layers = nn.ModuleList()
        num_layers = len(self.dnn_hidden_size)
        input_size = self.field_size * self.embedding_size
        self.deep_layers.append(SubLayer(input_size=input_size, output_size=self.dnn_hidden_size[0],
                                         dropout=self.dnn_dropout[1]))
        for i in range(1, num_layers):
            self.deep_layers.append(SubLayer(input_size=self.dnn_hidden_size[i-1],
                                             output_size=self.dnn_hidden_size[i],
                                             dropout=self.dnn_dropout[i+1]))

        ## output
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.dnn_hidden_size[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        else:
            input_size = self.dnn_hidden_size[-1]

        p = 1.0 / np.sqrt(input_size + 1)
        self.final_W = nn.Parameter(torch.Tensor(1, input_size))
        self.final_b = nn.Parameter(torch.Tensor(1))
        self.final_W.data.uniform_(-p, p)
        self.final_b.data.fill_(0)

    def forward(self, feat_index, feat_value):

        # 1. FM
        embeddings = self.feature_embedding[feat_index]
        feat_value = feat_value.view(-1, self.field_size, 1)
        embeddings = torch.mul(embeddings, feat_value)

        ## the first order features
        first_order = self.feature_bias[feat_index]
        first_order = torch.mul(first_order, feat_value)
        first_order = torch.sum(first_order, dim=-1)
        first_order = F.dropout(first_order, p=self.fm_dropout[0], training=self.training)

        # the second order features
        sum_featured_embed = torch.sum(embeddings, dim=1)
        sum_featured_embed_square = torch.pow(sum_featured_embed, 2)

        square_featured_embed = torch.pow(embeddings, 2)
        square_featured_embed_sum = torch.sum(square_featured_embed, dim=1)

        second_order = 0.5 * (sum_featured_embed_square - square_featured_embed_sum)
        second_order = F.dropout(second_order, p=self.fm_dropout[1], training=self.training)

        # 2. deep network
        deep_y = embeddings.view(-1, self.field_size * self.embedding_size)
        deep_y = F.dropout(deep_y, p=self.dnn_dropout[0], training=self.training)

        for layers in self.deep_layers:
            deep_y = layers(deep_y)

        # 3. concat
        if self.use_fm and self.use_deep:
            concat_input = torch.cat([first_order, second_order, deep_y], dim=1)
        elif self.use_fm:
            concat_input = torch.cat([first_order, second_order], dim=1)
        else:
            concat_input = deep_y

        output = F.linear(concat_input, self.final_W) + self.final_b
        return output, first_order, second_order, deep_y