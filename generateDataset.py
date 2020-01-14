# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
import argparse
import os
import random
import yaml
import downloadData
import preprocess

# 1. 读取文件 2. 分析参数 3. 
def read_setting():
    parser = argparse.ArgumentParser('parameters')
    # dataset-shift
    parser.add_argument('--dataset-shift', type=list, default=[1,2,3,4,5], help="# dataset-shift describes all dataset offsets")
    # is-noisy
    parser.add_argument('--is-noisy', type=bool, default=False, help="adding noisy")
    # noisy-mode
    parser.add_argument('--noisy-mode', type=int, default=1, help="noisy mode")
    parser.add_argument('--noisy-rate', type=float, default=0.1,
                        help= "noisy-rate")
    # use config
    parser.add_argument('--use-config', type=bool, default = True,
                        help="label feature skew")
    # each node - label kind
    parser.add_argument('--is-GAN', type=bool, default=False,
                        help="Same label，different features")
    parser.add_argument('--differetn-labels', type=bool, default=False,
                        help="Same features，different labels")
    parser.add_argument('--is-unbalance', type=bool, default=False,
                        help="Quantity skew or unbalance")
    parser.add_argument('--client-quantity', type=list, default=[],
                        help="whether add error dataset default=False")
    parser.add_argument('--add-error-rate', type=float, default=0.01,
                        help="if split-mode == 3, add same error dataset")

    parser.add_argument('--isuse-yaml', type= bool, default= True,
                        help='isuse-yaml = True means using yaml file, false means using command line')

    args = parser.parse_args()

    args = readYaml("./datasetSetting.yaml", args)

    # valid parameters
    if args.dataset_shift == [1]:
        print("currently only for CIFAR10 and MNIST")
        return
    if args.dataset_shift == [1, 2]:
        print("Error: the number of dataset smaller than node num")
        return
    if args.node_num != len(args.dataset_size_list) or args.node_num != len(args.node_label_num):
        print("Error: nodes num is not equal to the length of dataset_size_list or node_label_num ")
        return

    train_loader, test_loader = downloadData.load_data(args)

    # splitDataset(args, train_loader)


def readYaml(path, args):
    if args.isuse_yaml == False:
        return args
    if not os.path.exists(path):
        return args
    f = open(path)
    config = yaml.load(f)

    args.dataset_shift = config["dataset_shift"]
    args.is_noisy = int(config["is_noisy"])
    args.noisy_mode = config["noisy_mode"]
    args.noisy_rate = config["noisy_rate"]
    args.split_mode = int(config["split_mode"])
    args.node_label_num = config["node_label_num"]
    args.isadd_label = config["isadd_label"]
    args.add_label_rate = float(config["add_label_rate"])
    args.isadd_error = config["isadd_error"]
    args.add_error_rate = float(config["add_error_rate"])
    return args


if __name__ == "__main__":
    read_setting()
    pass