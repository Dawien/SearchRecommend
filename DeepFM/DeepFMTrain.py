# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 3:35 下午
# @Author  : Dawein
# @File    : DeepFMTrain.py
# @Software : PyCharm

import time
import numpy as np
import tensorflow as tf
from DeepFM import DeepFM
from DeepFMEval import Eval
from Metrics import gini_norm

# train
class Train():

    def __init__(self, params, feature_size, field_size):

        self.epochs = params.epochs
        self.batch_size = params.batch_size
        self.greater_is_better = params.greater_is_better
        self.train_results = []
        self.valid_results = []

        # init deep-fm model
        embedding_size = params.embedding_size
        self.deepfm = DeepFM(feature_size, field_size, embedding_size, params)

        # init all model
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

        # init eval
        self.eval = Eval(params, self.deepfm, self.sess)

    # shuffle three lists
    def shuffle_in_unison_scary(self, Xi, Xv, y):

        random_state = np.random.get_state()
        np.random.shuffle(Xi)
        np.random.set_state(random_state)
        np.random.shuffle(Xv)
        np.random.set_state(random_state)
        np.random.shuffle(y)

        return Xi, Xv, y

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[w] for w in y[start:end]]

    ## train
    def training(self, train_Xi, train_Xv, train_y,
                  valid_Xi=None, valid_Xv=None, valid_y=None,
                  early_stopping=False, refit=False):

        need_valid = False
        if valid_Xi is not None:
            need_valid = True

        for epoch in range(1, self.epochs+1):
            t1 = time.time()
            self.shuffle_in_unison_scary(train_Xi, train_Xv, train_y)
            total_batch = int(len(train_y) / self.batch_size) + 1
            for i in range(total_batch):
                batch_Xi, batch_Xv, batch_y = self.get_batch(train_Xi, train_Xv, train_y, self.batch_size, i)
                feed_dict = {self.deepfm.feat_index: batch_Xi,
                             self.deepfm.feat_value: batch_Xv,
                             self.deepfm.label: batch_y,
                             self.deepfm.train_phase: True}
                loss, opt = self.sess.run([self.deepfm.loss, self.deepfm.optimizer],
                                          feed_dict=feed_dict)
                print("epoch: %d, loss: %.4f" % (epoch, loss))

            # evaluate training and validation datasets
            train_pred_y = self.eval.predict(train_Xi, train_Xv)
            self.train_results.append(gini_norm(train_y, train_pred_y))
            if need_valid:
                valid_pred_y = self.eval.predict(valid_Xi, valid_Xv)
                self.valid_results.append(gini_norm(valid_y, valid_pred_y))
                print("epoch: %d, train-result: %.4f, valid-result: %.4f, "
                      "cost-time: %.2f s" % (epoch, self.train_results[-1],
                                             self.valid_results[-1], time.time() - t1))
            else:
                print("epoch: %d, train-result: %.4f, cost-time: %.2f s"
                      % (epoch, self.train_results[-1], time.time() - t1))

            if need_valid and early_stopping and self.training_termination(self.valid_results):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if need_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_results)
            else:
                best_valid_score = min(self.valid_results)

            best_epoch = self.valid_results.index(best_valid_score)
            best_train_score = self.train_results[best_epoch]
            train_Xi = train_Xi + valid_Xi
            train_Xv = train_Xv + valid_Xv
            train_y = train_y + valid_y
            for epoch in range(1, 100):
                self.shuffle_in_unison_scary(train_Xi, train_Xv, train_y)
                total_batch = int(len(train_y) / self.batch_size) + 1
                for i in range(total_batch):
                    batch_Xi, batch_Xv, batch_y = self.get_batch(train_Xi, train_Xv, train_y, self.batch_size, i)
                    feed_dict = {self.deepfm.feat_index: batch_Xi,
                                 self.deepfm.feat_value: batch_Xv,
                                 self.deepfm.label: batch_y,
                                 self.deepfm.train_phase: True}
                    loss, opt = self.sess.run([self.deepfm.output, self.deepfm.optimizer],
                                              feed_dict=feed_dict)
                    print("epoch: %d, loss: %.4f" % (epoch + 1, loss))

                # check
                train_pred_y = self.eval.predict(train_Xi, train_Xv)
                train_result = gini_norm(train_y, train_pred_y)
                if abs(train_result - best_train_score) < 0.001 \
                    or (self.greater_is_better and train_result > best_train_score) \
                    or ((not self.greater_is_better) and train_result < best_train_score):
                    break

    # early stopping
    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] < valid_result[-3] < valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4] > valid_result[-5]:
                    return True
        return False

