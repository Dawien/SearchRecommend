# -*- coding: utf-8 -*-
# @Time    : 2020/9/27 11:04 上午
# @Author  : Dawein
# @File    : parse_config.py
# @Software : PyCharm

import os
import sys
import configparser
from optparse import OptionParser

home_dir = os.path.split(os.path.realpath(__file__))[0]

def real_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(home_dir, path)

def load_config(config_file):

    config = configparser.RawConfigParser()
    config.optionxform = str

    cofig_file = real_path(os.path.join(home_dir, config_file))
    if not os.path.exists(cofig_file):
        print("config file does not exist: " + cofig_file)
        sys.exit(0)

    config.read(cofig_file, encoding="utf-8")
    params = {}
    for dtype in config.sections():
        for k, v in config.items(dtype):
            k = k.lower()
            if dtype == "bool":
                params[k] = eval(v)
            else:
                params[k] = eval(dtype + "(" + v + ")")

    return params