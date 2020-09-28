# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 10:09 上午
# @Author  : Dawein
# @File    : runDeepFM.py
# @Software : PyCharm

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataProcess import LoadData, FeatureDictionay, DataParser
from torchTrainDeepFM import Train
from Metrics import gini_norm
from ParseConfig import load_config
from sklearn.model_selection import StratifiedKFold
from collections import namedtuple

def _make_submission(ids, y_pred, output_dir, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(output_dir, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i+1))
        legends.append("valid-%d" % (i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("fig/%s.png" % model_name)
    plt.close()

def run_base_model_dfm(train_df, test_df, data_folds, params):

    # 解析数据，构建特征索引和特征值
    fd = FeatureDictionay(df_train=train_df,
                          df_test=test_df,
                          numeric_cols=params.numeric_cols,
                          ignore_cols=params.ignore_cols)

    data_parser = DataParser(feat_dict=fd)

    train_Xi, train_Xv, train_y = data_parser.parse(df=train_df, has_label=True)
    test_Xi, test_Xv, test_ids = data_parser.parse(df=test_df)

    # get feature size and field size
    feature_size = fd.feat_size
    field_size = len(train_Xi[0])

    train_meta_y = np.zeros((train_df.shape[0], 1), dtype=float)
    test_meta_y = np.zeros((test_df.shape[0], 1), dtype=float)

    _get = lambda x, l: [x[i] for i in l]

    # metric
    gini_results_cv = np.zeros(len(data_folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(data_folds), params.epochs), dtype=float)
    gini_results_epoch_valid = np.zeros((len(data_folds), params.epochs), dtype=float)

    for idx, (train_idx, valid_idx) in enumerate(data_folds):
        train_Xi_ = _get(train_Xi, train_idx)
        train_Xv_ = _get(train_Xv, train_idx)
        train_y_  = _get(train_y, train_idx)

        valid_Xi_ = _get(train_Xi, valid_idx)
        valid_Xv_ = _get(train_Xv, valid_idx)
        valid_y_  = _get(train_y, valid_idx)

        # construct model, for folds
        dfm = Train(params, feature_size, field_size)

        dfm.training(train_Xi_, train_Xv_, train_y_, valid_Xi_, valid_Xv_, valid_y_)

        train_meta_y[valid_idx, 0] = dfm.predict(valid_Xi_, valid_Xv_)
        test_meta_y[:, 0] += dfm.predict(test_Xi, test_Xv)

        gini_results_cv[idx] = gini_norm(valid_y_, train_meta_y[valid_idx])
        gini_results_epoch_train[idx] = dfm.train_results
        gini_results_epoch_valid[idx] = dfm.valid_results

    test_meta_y = test_meta_y / float(len(data_folds))

    # save result
    if params.use_fm and params.use_deep:
        clf_str = "DeepFM"
    elif params.use_fm:
        clf_str = "FM"
    else:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(test_ids, test_meta_y, params.sub_dir, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return train_meta_y, test_meta_y

def run_main():
    print("Start to process......")
    params = load_config("./config.conf")
    params = namedtuple("params", params.keys())(**params)

    # load data
    ld = LoadData(params)
    df_train, df_test, train_X, train_y, \
        test_X, test_ids, cat_features_indices = ld.process()

    # folds
    # StratifiedKFold分层采样，用于交叉验证。根据标签中不同类别的占比来拆分数据。
    # 将数据及划分成n_splits个互斥的数据集。
    folds = list(StratifiedKFold(n_splits=params.num_splits,
                                 shuffle=True,
                                 random_state=params.random_seed).split(train_X, train_y))

    # Deep - FM
    train_dfm_y, test_dfm_y = run_base_model_dfm(df_train, df_test, folds, params)

    return train_dfm_y, test_dfm_y

## main
if __name__ == '__main__':
    run_main()