# -*- coding: utf-8 -*-
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
            datasets.MNIST('data', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        
    return train_loader, test_loader