# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 4:21 下午
# @Author  : Dawein
# @File    : DeepFMEval.py
# @Software : PyCharm

import numpy as np
import tensorflow as tf
from DeepFM import DeepFM

class Eval():

    def __init__(self, params, df_model=None, sess=None, feature_size=None, filed_size=None):

        if sess is None:
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
        else:
            self.sess = sess

        self.batch_size = params.batch_size
        if df_model is None:
            print("Start to load a model.")
            checkpoint_path = params.model_dir
            embedding_size = params.embedding_size
            df_model = DeepFM(feature_size, filed_size, embedding_size, params)
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_path)
        self.model = df_model

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[w] for w in y[start:end]]

    def predict(self, Xi, Xv):

        dummy_y = [1] * len(Xi)
        batch_index = 0
        batch_Xi, batch_Xv, batch_y = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        pred_y = None
        while len(batch_Xi) > 0:
            batch_num = len(batch_y)
            feed_dict = {self.model.feat_index: batch_Xi,
                         self.model.feat_value: batch_Xv,
                         self.model.label: batch_y,
                         self.model.train_phase: False}
            batch_out = self.sess.run(self.model.output, feed_dict=feed_dict)

            if batch_index == 0:
                pred_y = np.reshape(batch_out, (batch_num, ))
            else:
                pred_y = np.concatenate((pred_y, np.reshape(batch_out, (batch_num, ))))

            batch_index += 1
            batch_Xi, batch_Xv, batch_y = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return pred_y