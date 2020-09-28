# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 5:57 下午
# @Author  : Dawein
# @File    : data_process.py
# @Software : PyCharm

import numpy as np
import pandas as pd

class LoadData():
    def __init__(self, config):
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.id_name = "id"
        self.target_name = "target"
        self.ignore_cols = config.ignore_cols
        self.categorical_cols = config.categorical_cols

    def process(self):
        df_train = pd.read_csv(self.train_file)
        df_test = pd.read_csv(self.test_file)

        def preprocess(df):
            cols = [c for c in df.columns if c not in [self.id_name, self.target_name]]
            df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
            df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
            return df

        df_train = preprocess(df_train)
        df_test = preprocess(df_test)

        cols = [c for c in df_train.columns if c not in [self.id_name, self.target_name]]
        cols = [c for c in cols if (not c in self.ignore_cols)]

        train_X = df_train[cols].values
        train_y = df_train[self.target_name].values

        test_X = df_test[cols].values
        test_ids = df_test[self.id_name].values

        cat_features_indices = [i for i,c in enumerate(cols) if c in self.categorical_cols]

        return df_train, df_test, train_X, train_y, test_X, test_ids, cat_features_indices

class FeatureDictionay():

    def __init__(self, df_train, df_test, numeric_cols=[], ignore_cols=[]):

        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols

        # 创建特诊字典，计算特征的维度，numerical的特征只占1位，categories的特征有多少个值就占多少位
        train_data = df_train
        test_data = df_test
        df = pd.concat([train_data, test_data])
        self.feat_dict = {}
        feat_count = 0
        for col in df.columns:
            if col in ignore_cols:
                continue

            if col in numeric_cols:
                self.feat_dict[col] = feat_count
                feat_count += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(feat_count, feat_count+len(us))))
                feat_count += len(us)
        print("feat_dict: ", self.feat_dict)
        self.feat_size = feat_count

class DataParser():

    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not (infile is None and df is None), "infile or df at least one is set."
        assert not (infile is not None and df is not None), "only one can be set."

        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            dfi.drop(["id"], axis=1, inplace=True)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1

        Xi = dfi.values.tolist()
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids