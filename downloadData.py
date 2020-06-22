# -*- coding: utf-8 -*-
import argparse

import torch

from torchvision import datasets, transforms


# CIFAR-10,
# mean, [0.5, 0.5, 0.5]
# std, [0.5, 0.5, 0.5]
# CIFAR-100,
# mean, [0.5071, 0.4865, 0.4409]
# std, [0.2673, 0.2564, 0.2762]

def load_data(args):
    args.batch_size = 1
    train_loader = []
    test_loader = []
    if args.dataset_mode == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1
        )
    elif args.dataset_mode == "CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
    elif args.dataset_mode == "MNIST":
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/newMNIST', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/newMNIST', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        
    return train_loader, test_loader

class argsment:
    # 定义基本属性
    batch_size = 1,
    dataset_mode = "MNIST",
    # constructor
    def __init__(self, batch, mode):
        self.batch_size = batch,
        self.dataset_mode = mode,
    # method
    def getBatchSize(self):
        print(self.batch_size)

# download data
if __name__ == "__main__":

    parser = argparse.ArgumentParser('parameters')
    # dataset
    parser.add_argument('--dataset-mode', type=str, default="CIFAR100", help="dataset")
    args = parser.parse_args()

    print(args.dataset_mode)
    train_loader, test_loader = load_data(args)
    print(train_loader)