# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
import argparse
import os
import random
import yaml
import downloadData
import fdata
import preprocess
import mnist_bias

# 1. download dataset  2. split dataset
def make_dataset():
    parser = argparse.ArgumentParser('parameters')
    # dataset
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10", help="dataset")
    # node num
    parser.add_argument('--node-num', type=int, default=4,
                        help="Number of node (default n=10) one node corresponding to one dataset")
    # small dataset config
    parser.add_argument('--isaverage-dataset-size', type=bool, default=True, help="if average splits dataset")
    parser.add_argument('--dataset-size-list', type=list, default=[5000, 3000, 2000, 3300],
                        help= "each small dataset size,if isaverage-dataset-size == True, list contain one element")
    # split mode
    parser.add_argument('--split-mode', type=int, default = 1,
                        help="dataset split: randomSplit(0), splitByLabels(1)")
    # each node - label kind
    parser.add_argument('--node-label-num', type=list, default=[4, 3, 2, 1],
                        help="each node consists of label kind, default each node has one kind of label")
    parser.add_argument('--isadd-label', type=bool, default=True,
                        help="whether add error dataset default=False")
    parser.add_argument('--add-label-rate', type=float, default=0.1,
                        help="if split-mode == 2 or 3, add same normal small dataset")
    parser.add_argument('--isadd-error', type=bool, default=True,
                        help="whether add error dataset default=False")
    parser.add_argument('--add-error-rate', type=float, default=0.01,
                        help="if split-mode == 3, add same error dataset")

    parser.add_argument('--isuse-yaml', type= bool, default= True,
                        help='isuse-yaml = True means using yaml file, false means using command line')

    parser.add_argument('--RandomResizedCrop', type=list, default=[0.2, 1.],
                        help='RandomResizedCrop')
    parser.add_argument('--GaussianBlur', type=list, default=[0.1, .2],
                        help='GaussianBlur')
    parser.add_argument('--RandomGrayscale', type=float, default=0.2,
                        help='GaussianBlur')
    parser.add_argument('--Normalize-mean', type=list, default=[0.4914, 0.4822, 0.4465],
                        help='Normalize-mean')
    parser.add_argument('--Normalize-std', type=list, default=[0.2023, 0.1994, 0.2010],
                        help='Normalize-std')
    # args.RandomResizedCrop = config["RandomResizedCrop"]
    # args.GaussianBlur = config["GaussianBlur"]
    # args.RandomGrayscale = config["RandomGrayscale"]
    # args.Normalize_mean = config["Normalize_mean"]
    # args.Normalize_std = config["Normalize_std"]

    args = parser.parse_args()

    args = readYaml("./config.yaml", args)
    # valid parameters
    if args.dataset_mode != "CIFAR10" and args.dataset_mode != "MNIST":
        print("currently only for CIFAR10 and MNIST")
        return
    if len(args.dataset_size_list) < args.node_num:
        print("Error: the number of dataset smaller than node num")
        return
    if args.node_num != len(args.dataset_size_list) or args.node_num != len(args.node_label_num):
        print("Error: nodes num is not equal to the length of dataset_size_list or node_label_num ")
        return


    if args.split_mode == 3:

        #file_path,  batch_size , sub_num, dataset_ident = 'CIFAR10C' , download = False, train_transform = cifar_train_transforms(), test_transform = cifar_test_transforms(), use_cuda =True
        Xloader = fdata.Loader("./data", batch_size = 32, sub_num=args.sub_num, dataset_ident = 'CIFAR10C',  download = False, train_transform=fdata.cifar_train_transforms(args), test_transform=fdata.cifar_test_transforms(), use_cuda=True)

        return

    if args.split_mode == 4:
        mnist_bias.mnist_process(args.datasetpath)
        return

    train_loader, test_loader = downloadData.load_data(args)

    splitDataset(args, train_loader)


def readYaml(path, args):
    if args.isuse_yaml == False:
        return args
    if not os.path.exists(path):
        return args
    f = open(path)
    config = yaml.load(f)

    args.dataset_mode = config["dataset_mode"]
    args.datasetpath = str(config["datasetpath"])
    args.node_num = int(config["node_num"])
    args.isaverage_dataset_size = config["isaverage_dataset_size"]
    args.dataset_size_list = config["dataset_size_list"]
    args.split_mode = int(config["split_mode"])
    args.node_label_num = config["node_label_num"]
    args.isadd_label = config["isadd_label"]
    args.add_label_rate = float(config["add_label_rate"])
    args.isadd_error = config["isadd_error"]
    args.add_error_rate = float(config["add_error_rate"])

    args.RandomResizedCrop = config["RandomResizedCrop"]
    args.GaussianBlur = config["GaussianBlur"]
    args.RandomGrayscale = config["RandomGrayscale"]
    args.Normalize_mean = config["Normalize_mean"]
    args.Normalize_std = config["Normalize_std"]
    args.sub_num = config["sub_num"]

    return args

def splitDataset(args, train_loader):
    # sub_datasets [
    #               [[imgs, label], [imgs, label]....],
    #               [[imgs, label], [imgs, label]....],
    #              ]
    #  randomSplit : 1. no error dataset 2. add error dataset
    #  splitByLabel: 1. just 2. add other dataset, no error 3. add error no other 4. add both
    parent_path = "./" + args.dataset_mode
    if args.split_mode == 0:  # 1. Randomly split CIFAR10 into n small datasets
        if args.isadd_error == False:
            args.add_error_rate = 0.0
            sub_datasets = randomSplit(args, train_loader)
            savenpy(parent_path+"/randomSplit/", sub_datasets, args)
        else:
            temp_sub_datasets = randomSplit(args, train_loader)
            sub_datasets = addErrorDataset(args, temp_sub_datasets)

            savenpy(parent_path+"/randomSplitWithError/", sub_datasets, args)

    elif args.split_mode == 1:  # 2. Divide CIFAR10 into n small datasets according to dataset labels
        if args.isadd_label == False and args.isadd_error == False:
            args.add_error_rate = 0.0
            args.add_label_rate = 0.0
            sub_datasets = splitByLabels(args, train_loader)
            savenpy(parent_path+"/splitByLabels/", sub_datasets, args)
        elif args.isadd_label == True and args.isadd_error == False:
            args.add_error_rate = 0.0
            # 3. Based on the 2nd method, each dataset adds 10% of the data taken from the other datasets
            sub_datasets = splitByLabelsAnddDataset(args, train_loader)
            savenpy(parent_path+"/splitByLabelsAnddDataset/", sub_datasets, args)
        elif args.isadd_label == False and args.isadd_error == True:
            args.add_label_rate = 0.0
            # 5. get dataset, each dataset adds some error label data to form a new dataset
            temp_sub_datasets = splitByLabels(args, train_loader)
            sub_datasets = addErrorDataset(args, temp_sub_datasets)
            savenpy(parent_path+"/splitByLabelsWithErrorDataset/", sub_datasets, args)
        else:
            temp_sub_datasets = splitByLabelsAnddDataset(args, train_loader)
            sub_datasets = addErrorDataset(args, temp_sub_datasets)
            savenpy(parent_path+"/splitByLabelsWithNormalAndErrorDataset/", sub_datasets, args)


# 1. Randomly split Dataset into n small datasets
def randomSplit(args, loader):
    args.add_label_rate = 0.0
    node_num = args.node_num
    sub_datasets = [[] for i in range(node_num)]

    dataset_size_list = args.dataset_size_list
    if args.isaverage_dataset_size == True:
        # 均分
        temp_list = []
        node_index = 0
        num = 0
        print(loader.dataset)
        for step, (imgs, label) in enumerate(loader):
            temp_list.append([imgs[0].numpy(), label[0].numpy()])

            num += 1
            if (num % (dataset_size_list[node_index])) == 0 and num != 0:
                print("finish average spliting %d dataset" % node_index)
                # TODO(save one small dataset)
                sub_datasets[node_index] = temp_list
                node_index = node_index+1
                if node_index == node_num:
                    break
                temp_list = []
            if step == len(loader.dataset.data) -1:
                print("finish left spliting %d dataset" % node_index)
                sub_datasets[node_index] = temp_list
    else:
        temp_list = []
        node_index = 0
        temp_step = dataset_size_list[node_index]
        num = 0
        if args.dataset_mode == "CIFAR10":
            for step, (imgs, labels) in enumerate(loader):
                num +=1

                temp_list.append([imgs[0].numpy(), labels[0].numpy()])
                # temp_list.append([imgs.numpy(), labels.numpy()])
                if num == temp_step and num !=0:
                    print("finish spliting %d dataset" % node_index)
                    sub_datasets[node_index] = temp_list
                    node_index = node_index + 1
                    if node_index == node_num:
                        break
                    temp_step += dataset_size_list[node_index]
                    temp_list = []
                if step == len(loader.dataset.data) -1:
                    print("finish left spliting %d dataset" % node_index)
                    sub_datasets[node_index] = temp_list
        elif args.dataset_mode == "MNIST":
            step = 0
            for (i, data) in enumerate(loader):
                step += 1
                num +=1

                temp_list.append([data[0].numpy(), data[1].numpy()])
                # temp_list.append([imgs.numpy(), labels.numpy()])
                if num == temp_step and num !=0:
                    print("finish spliting %d dataset" % node_index)
                    sub_datasets[node_index] = temp_list
                    node_index = node_index + 1
                    if node_index == node_num:
                        break
                    temp_step += dataset_size_list[node_index]
                    temp_list = []
                if i == len(loader.dataset.data) -1:
                    print("finish left spliting %d dataset" % node_index)
                    sub_datasets[node_index] = temp_list

    return sub_datasets

# 2. Divide CIFAR10 into n small datasets according to dataset labels
def splitByLabels(args, train_loader):
    sub_datasets = [[] for i in range(args.node_num)]
    temp_datasets = [[] for i in range(10)]
    # category according to nodes nums, each node 按照 节点数 分类，每个节点的类别个数,对应数据量
    node_index = 0
    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()

        #  CIFAR10 Dataset
        # imgs[0].numpy()： <class 'tuple'>: (3, 32, 32)  label[0].numpy() [x] =>
        # temp_datasets [
        #                [[(3, 32, 32) , 0], [(3, 32, 32) , 0], ..],
        #                [[[(3, 32, 32) , 1], [(3, 32, 32) , 1], ..],
        #                ...
        #                ]
        temp_datasets[num_label].append(
            [imgs[0].numpy(), label[0].numpy()])
        if step % 5000 == 0:
            print("split dataset step: ", step)

    # loop temp_datasets, add and contract
    # node_label_num [1, 2, 2, 5, 7]
    rs = random.sample(range(0, 10), 10) # 0 - 9 random nums
    # according to nodes list, distribute label dataset
    all_label_kinds = len(temp_datasets)
    sum_x = 0
    for index, x in enumerate(args.node_label_num):
        temp_list = []
        if x > all_label_kinds:
            x = all_label_kinds
        for y in range(x):
            # temp_list only contain 10 kinds labels
            labels_index = (y + sum_x) % all_label_kinds
            temp_list.extend(temp_datasets[labels_index])
            print("node %d" % index, "| add label-%d dataset" % (labels_index))
        # if we need the part of data, shuffle, split
        if args.isaverage_dataset_size == True:
            random.shuffle(temp_list)
            temp_list = temp_list[:args.dataset_size_list[index]]
        
        sub_datasets[index] = temp_list
        sum_x += x

    return sub_datasets

# 3. Based on the 2nd method, each dataset adds n% of the data taken from the other datasets
def splitByLabelsAnddDataset(args, train_loader):
    percent = args.add_label_rate
    # call splitByLabels
    sub_datasets = splitByLabels(args, train_loader)

    # add other data Attention other dataset
    add_rate_num = [int(percent*len(sub_datasets[i])) for i in range(args.node_num)]
    for i in range(args.node_num):
        for step, (imgs, label) in enumerate(train_loader):
            if step < add_rate_num[i]:
                if step % 100 == 0:
                    print("node %d " % i, "| step：%d, adding other label dataset" % step)
                sub_datasets[i].append([imgs[0].numpy(), label[0].numpy()])
            else:
                break
    print("adding other data succeed!")
    return sub_datasets

# 4. each dataset adds some error label data
def addErrorDataset(args, array):
    error_ratio = args.add_error_rate
    add_error_nums = [int(error_ratio * len(array[i])) for i in range(args.node_num)]
    # add error data
    for i in range(args.node_num):
        for index in range(add_error_nums[i]):
            if index % 5 == 0:
                print("node %d" % i, "| step：%d, adding other error dataset" % index)
            # array        [
            #               [[imgs, label], [imgs, label]....],
            #               [[imgs, label], [imgs, label]....],
            #              ]
            real_label = array[i][index][1]
            error_label = random.choice([i for i in range(0, 9) if i not in [real_label]])
            array[i].append([array[i][index][0], error_label])
    print("adds some error label data succeed!")
    return array

# save  each small list dataset file
def savenpy(path, array, args):
    '''
    loop  array save each small list dataset file
    :param path:
    :param array:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    # array [[(3, 32, 32), x], [(3, 32, 32), x]]
    # randomSplit_dataset size_target label_添加label_errorlabel
    # label classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i in range(len(array)):
        if len(array[i]) != 0:
            filename = ''
            if args.split_mode == 0:
                filename = 'randomSplit'+'_node_'+str(i) + '_'+str(len(array[i]))+ '_' + "normal"
            elif args.split_mode == 1:
                if int(args.node_label_num[i]) != 1:
                    filename = 'SplitByLabels'+'_node_'+str(i) + '_' + str(len(array[i])) + '_' + classes[array[i][0][1]]+ "andMore"
                else:
                    filename = 'SplitByLabels'+'_node_'+str(i) + '_' + str(len(array[i])) + '_' + classes[array[i][0][1]]

            strings = path + filename +'_' + str(args.add_label_rate) + '_' + str(args.add_error_rate)+'.npy'
            np.save(file=strings, arr=array[i])
            print("index %d saved %s" % (i, strings))
    print("save file succeed !")

def readnpy(path):
    # npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
    # when allow_pickle=True, matrix needs same size
    np_array = np.load(path, allow_pickle=True)
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
    make_dataset()
    # preprocess.load_npy("./cifar10/splitByLabelsWithNormalAndErrorDataset/SplitByLabels_5555_dog_0.1_0.01.npy")
    # readnpy("./cifar10/splitByLabelsWithNormalAndErrorDataset/SplitByLabels_3666_truck.npy")