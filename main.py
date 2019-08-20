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
    parser.add_argument('--split-mode', type=int, default=1,
                        help="dataset split: randomSplit(0), splitByLabels(1), "
                             "splitByLabelsAnddDataset(2), splitByLabelsAnddDataset5(3), addErrorDataset(5)")
    parser.add_argument('--batch-size', type=int, default=1, help='batch size, (default: 1)')
    parser.add_argument('--node-num', type=int, default=10, help="Number of node (default n=10)")
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10", help="dataset")
    
    args = parser.parse_args()

    train_loader, test_loader = preprocess.load_data(args)
    # 类别标签
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # sub_datasets [
    #               [(imgs, label), (imgs, label)....
    #               ],
    #              ]
    if args.split_mode == 0:
        # 1. 随机 切分数据集
        sub_datasets = randomSplit(train_loader)
        savenpy("./cifar10/randomSplit/", "randomSplit", sub_datasets)
    elif args.split_mode == 1:
        # 2. 根据label切分 10类
        sub_datasets = splitByLabels(train_loader)
        savenpy("./cifar10/splitByLabels/", "splitByLabels", sub_datasets)
    elif args.split_mode == 2:
        # 3. 根据leabel切分，增加 10% other dataset
        sub_datasets = splitByLabelsAnddDataset(train_loader, 0.1)
        savenpy("./cifar10/splitByLabelsAnddDataset/", "splitByLabelsAnddDataset", sub_datasets)
    elif args.split_mode == 3:
        # 4. 根据leabel切分，增加 50% other dataset
        sub_datasets = splitByLabelsAnddDataset(train_loader, 0.5)
        savenpy("./cifar10/splitByLabelsAnddDataset5/", "splitByLabelsAnddDataset5", sub_datasets)
    elif args.split_mode == 4:
        # 5. 根据leabel切分，增加 10% other dataset + error data label
        sub_datasets = addErrorDataset(train_loader, 0.1, error=True)
        savenpy("./cifar10/splitByLabelsAnddDataset5/", "splitByLabelsAnddDataset5", sub_datasets)

# 1. 随机 切分数据集  保存
def randomSplit(train_loader):
    sub_datasets = [[] for i in range(10)]
    num = -1
    for step, (imgs, label) in enumerate(train_loader):
        if step % 10000 == 0:
            num = num+1
            print("loop train step %d ~ step %d" % (step, step+10000))
        sub_datasets[num].append((imgs[0], label[0]))

    return sub_datasets

# 2. 根据label切分 10类
def splitByLabels(train_loader):
    sub_datasets = [[] for i in range(10)]
    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()
        # imgs[0].numpy()： <class 'tuple'>: (3, 32, 32)  label[0].numpy() [x] =>
        sub_datasets[num_label].append(
            [imgs[0].numpy(),np.array(label[0].numpy())]) # [[(3, 32, 32) , [x]], [(3, 32, 32) , x], ..]
        if step % 10000 == 0:
            print("loop train step: ", step)
    return sub_datasets

# 3. 根据leabel切分，增加 10% other dataset
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
                if step % 100 == 0:
                    print("step：%d, adding other data" % step)
                sub_datasets[i].append([imgs[0].numpy(), np.array(label[0].numpy())])
            else:
                break

    return sub_datasets

# 5. 根据leabel切分，增加 10% other dataset + error data label
def addErrorDataset(train_loader, percent=0.1, error=False, error_ratio=0.01):
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
                if step < int(percent * len(sub_datasets[i])):
                    if step % 100 == 0:
                        print("step：%d, adding other dataset" % step)
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

# 保存 每个小list的数据集文件
def savenpy(path, filename, array):
    '''
    循环 array 每个小list的数据集文件 并保存
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

if __name__ == "__main__":
    # main()
    readnpy("./cifar10/splitByLabels/splitByLabels_0.npy")