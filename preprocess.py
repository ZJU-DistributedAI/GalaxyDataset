# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
from torchvision import datasets, transforms
import argparse
import os
import random
import yaml
import downloadData

def load_npy(path):
    # npy file: [[imgs, label], [imgs, label]...., [imgs, label]]
    # when allow_pickle=True, matrix needs same size
    if not os.path.isfile(path):
        print("files do not exists!!")
        return
    np_array = np.load(path, allow_pickle=True)
    imgs = []
    label = []
    for index in range(len(np_array)):
        imgs.append(np_array[index][0])
        label.append(np_array[index][1])
    torch_dataset = Data.TensorDataset(torch.from_numpy(np.array(imgs)), torch.from_numpy(np.array(label)))

    train_loader = Data.DataLoader(
        torch_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transform_test),
        batch_size=32,
        shuffle=True,
        num_workers=1
    )
    print("train_loader, test_loader generated succeed!")
    return train_loader, test_loader

if __name__ == "__main__":
    dataloder = load_npy("./cifar10/splitByLabelsWithNormalAndErrorDataset/SplitByLabels_3666_truck.npy")