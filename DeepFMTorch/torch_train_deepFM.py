# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 10:07 上午
# @Author  : Dawein
# @File    : torch_train_deepFM.py
# @Software : PyCharm

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch_deepFM import DeepFM
from Metrics import gini_norm

class Train:

    def __init__(self, params, feature_size, field_size):
        self.params = params
        self.feature_size = feature_size
        self.field_size = field_size
        self.train_results = []
        self.valid_results = []

        ## init deep-fm model
        self.deepfm = DeepFM(feature_size, field_size, params)
        total_params = 0
        for p in self.deepfm.parameters():
            total_params += p.numel()
        print("Total parameters of model: %d" % (total_params))


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

    def predict(self, Xi, Xv):

        batch_size = self.params.batch_size
        dummy_y = [1] * len(Xi)
        batch_index = 0
        batch_Xi, batch_Xv, batch_y = self.get_batch(Xi, Xv, dummy_y, batch_size, batch_index)
        pred_y = None
        self.deepfm.eval()
        while len(batch_Xi) > 0:
            batch_num = len(batch_y)
            batch_Xi_tensor = torch.tensor(batch_Xi, dtype=torch.long)
            batch_Xv_tensor = torch.tensor(batch_Xv, dtype=torch.float)
            batch_out,_,_,_ = self.deepfm(batch_Xi_tensor, batch_Xv_tensor)
            batch_out = batch_out.detach().numpy()
            if batch_index == 0:
                pred_y = np.reshape(batch_out, (batch_num, ))
            else:
                pred_y = np.concatenate((pred_y, np.reshape(batch_out, (batch_num, ))))

            batch_index += 1
            batch_Xi, batch_Xv, batch_y = self.get_batch(Xi, Xv, dummy_y, batch_size, batch_index)

        return pred_y


    ## train
    def training(self, train_Xi, train_Xv, train_y,
                 valid_Xi=None, valid_Xv=None, valid_y=None,
                 early_stopping=False, refit=False):
        need_valid = False
        if valid_Xi is not None:
            need_valid = True

        ## construct optimizer
        optimizer_type = self.params.optimizer
        if optimizer_type == "sgd":
            optimizer = optim.SGD(self.deepfm.parameters(), lr=self.params.learning_rate)
        else:
            optimizer = optim.Adam(self.deepfm.parameters(),
                                   lr=self.params.learning_rate, betas=(0.9, 0.99),
                                   eps=1e-8, amsgrad=True)


        ##
        loss_type = self.params.loss_type
        for epoch in range(1, self.params.epochs + 1):
            self.deepfm.train()
            t1 = time.time()
            self.shuffle_in_unison_scary(train_Xi, train_Xv, train_y)
            total_batch = int(len(train_y) / self.params.batch_size) + 1
            for i in range(total_batch):
                batch_Xi, batch_Xv, batch_y = self.get_batch(train_Xi, train_Xv, train_y,
                                                             self.params.batch_size, i)
                batch_Xi = torch.tensor(batch_Xi, dtype=torch.long)
                batch_Xv = torch.tensor(batch_Xv, dtype=torch.float)
                batch_y = torch.tensor(batch_y, dtype=torch.long)

                optimizer.zero_grad()

                output, _, _, _ = self.deepfm(batch_Xi, batch_Xv)
                if loss_type == "logloss":
                    # for classification
                    output = F.sigmoid(output)
                    loss = -torch.mul(batch_y, torch.log(output)) \
                           - torch.mul((1-batch_y), torch.log(1-output))
                    loss = torch.mean(loss)
                elif loss_type == "mse":
                    # for regression
                    loss = F.mse_loss(input=output, target=batch_y)
                else:
                    raise ValueError("Unknown loss type, should be one of 'logloss/mes'")

                # l2 regularization on weights for preventing over-fitting
                if self.params.l2_reg > 0:
                    loss += self.params.l2_reg * torch.norm(self.deepfm.final_W, 2)
                    if self.params.use_deep:
                        for weight in self.deepfm.deep_layers:
                            loss += self.params.l2_reg * torch.norm(weight.W, 2)
                print("epoch: %d, loss: %.4f" % (epoch, loss.item()))

                # backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.deepfm.parameters(), self.params.grad_clip)

                optimizer.step()

            # each epoch with evaluate training and validation datasets
            train_pred_y = self.predict(train_Xi, train_Xv)
            self.train_results.append(gini_norm(train_y, train_pred_y))
            if need_valid:
                valid_pred_y = self.predict(valid_Xi, valid_Xv)
                self.valid_results.append(gini_norm(valid_y, valid_pred_y))
                print("epoch: %d, train-result: %.4f, valid-result: %.4f, "
                      "cost-time: %.2f s" % (epoch, self.train_results[-1],
                                             self.valid_results[-1], time.time() - t1))
            else:
                print("epoch: %d, train-result: %.4f, cost-time: %.2f s"
                      % (epoch, self.train_results[-1], time.time() - t1))

            if need_valid and early_stopping and self.training_termination(self.valid_results):
                break

        # fit a few more epochs on train+valid until result reaches the best_train_score
        if need_valid and refit:
            greater_is_better = self.params.greater_is_better
            if greater_is_better:
                best_valid_score = map(self.valid_results)
            else:
                best_valid_score = min(self.valid_results)

            best_epoch = self.valid_results.index(best_valid_score)
            best_train_score = self.train_results[best_epoch]
            train_Xi = train_Xi + valid_Xi
            train_Xv = train_Xv + valid_Xv
            train_y = train_y + valid_y
            for epoch in range(1, 100):
                self.shuffle_in_unison_scary(train_Xi, train_Xv, train_y)
                total_batch = int(len(train_y) / self.params.batch_size) + 1
                for i in range(total_batch):
                    batch_Xi, batch_Xv, batch_y = self.get_batch(train_Xi, train_Xv, train_y,
                                                                 self.params.batch_size, i)

                    batch_Xi = torch.tensor(batch_Xi, dtype=torch.long)
                    batch_Xv = torch.tensor(batch_Xv, dtype=torch.float)
                    batch_y = torch.tensor(batch_y, dtype=torch.float)

                    optimizer.zero_grad()

                    output, _, _, _ = self.deepfm(batch_Xi, batch_Xv)
                    if loss_type == "logloss":
                        # for classification
                        output = F.logsigmoid(output)
                        loss = F.cross_entropy(batch_y, output)
                    elif loss_type == "mse":
                        # for regression
                        loss = F.mse_loss(batch_y, output)
                    else:
                        raise ValueError("Unknown loss type, should be one of 'logloss/mes'")

                    # l2 regularization on weights for preventing over-fitting
                    if self.params.l2_reg > 0:
                        loss += self.params.l2_reg * torch.norm(self.deepfm.final_W, 2)
                        if self.params.use_deep:
                            for weight in self.deepfm.deep_layers:
                                loss += self.params.l2_reg * torch.norm(weight, 2)
                    print("epoch: %d, loss: %.4f" % (epoch, loss.item()))

                    # backward
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.deepfm.parameters(), self.params.grad_clip)

                    optimizer.step()

                # check
                train_pred_y = self.predict(train_Xi, train_Xv)
                train_result = gini_norm(train_y, train_pred_y)
                if abs(train_result - best_train_score) < 0.001 \
                    or (greater_is_better and train_result > best_train_score) \
                    or ((not greater_is_better) and train_result < best_train_score):
                    break

    ## early stopping
    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] < valid_result[-3] < valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4] > valid_result[-5]:
                    return True
        return False