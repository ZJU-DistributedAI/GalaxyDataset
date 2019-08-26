# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
import argparse
import os
import random

import preprocess

# 1. download dataset  2. split dataset
def main():
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10", help="dataset")
    parser.add_argument('--node-num', type=int, default=10, help="Number of node (default n=10)")

    parser.add_argument('--isaverage-dataset-size', type=bool, default=True, help="if average splits dataset")
    parser.add_argument('--dataset-size', type=list, default=[6000],
                        help= "each small dataset size,if average split, [small dataset size]")

    parser.add_argument('--split-mode', type=int, default=0,
                        help="dataset split: randomSplit(0), splitByLabels(1), "
                             "splitByLabelsAnddDataset(2), splitByLabelsAnddDataset5(3), addErrorDataset(5)")
    # parser.add_argument('--batch-size', type=int, default=1, help='batch size, (default: 1)')

    # 每个节点生成的数据集大小
    parser.add_argument('--isaverage', type=bool, default=True, help="every node owns same size of dataset")

    
    args = parser.parse_args()

    train_loader, test_loader = preprocess.load_data(args)
    # label classes
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # sub_datasets [
    #               [(imgs, label), (imgs, label)....
    #               ],
    #              ]
    if args.split_mode == 0:
        # 1. Randomly split CIFAR10 into 10 small datasets
        sub_datasets = randomSplit(train_loader)
        savenpy("./cifar10/randomSplit/", "randomSplit", sub_datasets)
    elif args.split_mode == 1:
        # 2. Divide CIFAR10 into 10 small datasets according to dataset labels
        sub_datasets = splitByLabels(train_loader)
        savenpy("./cifar10/splitByLabels/", "splitByLabels", sub_datasets)
    elif args.split_mode == 2:
        # 3. Based on the 2nd method, each dataset adds 10% of the data taken from the other 9 datasets
        sub_datasets = splitByLabelsAnddDataset(train_loader, 0.1)
        savenpy("./cifar10/splitByLabelsAnddDataset/", "splitByLabelsAnddDataset", sub_datasets)
    elif args.split_mode == 3:
        # 4. Based on the 2nd method, each dataset adds 50% of the data taken from the other 9 datasets
        sub_datasets = splitByLabelsAnddDataset(train_loader, 0.5)
        savenpy("./cifar10/splitByLabelsAnddDataset5/", "splitByLabelsAnddDataset5", sub_datasets)
    elif args.split_mode == 4:
        # 5. Based on the 3rd method, each dataset adds some error label data to form a new dataset
        sub_datasets = addErrorDataset(train_loader, 0.1, error=True, error_ratio=0.01)
        savenpy("./cifar10/splitByLabelsAnddDataset5/", "splitByLabelsAnddDataset5", sub_datasets)

# 1. Randomly split CIFAR10 into 10 small datasets
def randomSplit(train_loader):
    sub_datasets = [[] for i in range(10)]
    num = -1
    for step, (imgs, label) in enumerate(train_loader):
        if step % 10000 == 0:
            num = num+1
            print("loop train step %d ~ step %d" % (step, step+10000))
        sub_datasets[num].append((imgs[0], label[0]))

    return sub_datasets

# 2. Divide CIFAR10 into 10 small datasets according to dataset labels
def splitByLabels(train_loader):
    sub_datasets = [[] for i in range(10)]
    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()
        # imgs[0].numpy()： <class 'tuple'>: (3, 32, 32)  label[0].numpy() [x] =>
        sub_datasets[num_label].append(
            [imgs[0].numpy(), np.array(label[0].numpy())]) # [[(3, 32, 32) , [x]], [(3, 32, 32) , x], ..]
        if step % 10000 == 0:
            print("loop train step: ", step)
    return sub_datasets

# 3. Based on the 2nd method, each dataset adds 10% of the data taken from the other 9 datasets
def splitByLabelsAnddDataset(train_loader, percent=0.1):

    sub_datasets = [[] for i in range(10)]
    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()
        # imgs[0].numpy()： <class 'tuple'>: (3, 32, 32)  label[0].numpy() [x] =>
        sub_datasets[num_label].append(
            [imgs[0].numpy(), np.array(label[0].numpy())])  # [[(3, 32, 32) , [x]], [(3, 32, 32) , x], ..]
        if step % 5000 == 0:
            print("loop train step: ", step)

    # add other data
    for i in range(10):
        for step, (imgs, label) in enumerate(train_loader):
            if step < int(percent*len(sub_datasets[i])):
                if step % 1000 == 0:
                    print("step：%d, adding other data" % step)
                sub_datasets[i].append([imgs[0].numpy(), np.array(label[0].numpy())])
            else:
                break

    return sub_datasets
# 1. 多少 节点 -- 多少 数据
# 5. Based on the 3rd method, each dataset adds some error label data
def addErrorDataset(train_loader, percent=0.1, error=False, error_ratio=0.01):
    sub_datasets = [[] for i in range(10)]
    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()
        # imgs[0].numpy()： <class 'tuple'>: (3, 32, 32)  label[0].numpy() [x] =>
        sub_datasets[num_label].append(
            # [[(3, 32, 32) , [x]], [(3, 32, 32) , x], ..]
            [imgs[0].numpy(), np.array(label[0].numpy())])
        if step % 5000 == 0:
            print("loop train step: ", step)

        # add other data
        for i in range(10):
            for step, (imgs, label) in enumerate(train_loader):
                if step < int(percent * len(sub_datasets[i])):
                    if step % 500 == 0:
                        print("dataset index: %d step：%d, adding other dataset" % (i, step))
                    sub_datasets[i].append([imgs[0].numpy(), np.array(label[0].numpy())])
                else:
                    break

    # add error data
    if error == True:
        for i in range(10):
            for index in range(int(error_ratio*percent * len(sub_datasets[i]))):
                if index % 100 == 0:
                    print("step：%d, adding other error dataset" % index)
                sub_datasets[i].append([sub_datasets[i][index][0],
                                        np.array((sub_datasets[i][index][1].data.item() + random.randint(0, 9)) % 10)])

    return sub_datasets

# save  each small list dataset file
def savenpy(path, filename, array):
    '''
    loop  array save each small list dataset file
    :param path:
    :param filename:
    :param array:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    # array [((3, 32, 32), x), ((3, 32, 32), x)]
    for i in range(len(array)):
        strings = path + filename + '_' + str(i) + '.npy'
        print("index %d saving %s" % (i, strings))
        np.save(file=strings, arr=array[i])

    print("save %s successfully" % filename)

def readnpy(path):
    # npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
    np_array = np.load(path)
    imgs = []
    label = []
    for index in range(len(np_array)):
        imgs.append(np_array[index][0])
        label.append(np_array[index][1])
    torch_dataset = Data.TensorDataset(torch.from_numpy(np.array(imgs)), torch.from_numpy(np.array(label)))

    dataloader = Data.DataLoader(
        torch_dataset,
        batch_size=64,
        shuffle=True
    )
    print(dataloader)
    return dataloader

if __name__ == "__main__":
    main()
    # readnpy("./cifar10/splitByLabels/splitByLabels_0.npy")