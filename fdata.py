from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch as t
import numpy as np
import random
from PIL import ImageFilter
from PIL import Image

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# Xargs.RandomResizedCrop
def cifar_train_transforms(Xargs):
    all_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(Xargs.RandomResizedCrop[0], Xargs.RandomResizedCrop[1])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([GaussianBlur(Xargs.GaussianBlur)], p=0.5),
        transforms.RandomGrayscale(Xargs.RandomGrayscale),
        transforms.ToTensor(),
        transforms.Normalize(Xargs.Normalize_mean, Xargs.Normalize_std)
    ])
    return all_transforms


# def cifar_train_transforms():
#     all_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#     ])
#     return all_transforms


def cifar_test_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


class CIFAR10C(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10C, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            xi = self.transform(img)
            xj = self.transform(img)

        return xi, xj, target


loader_map = {
            'CIFAR10C': CIFAR10C,
            'CIFAR10': datasets.CIFAR10
        }
num_class = {
            'CIFAR10C': 10,
            'CIFAR10': 10
        }
class Loader(object):
    def __init__(self, file_path,  batch_size , sub_num, train_transform, test_transform, dataset_ident = 'CIFAR10C' , download = False,  use_cuda =True):

        train_dataset,test_dataset = self.get_dataset_train(loader_map[dataset_ident], file_path, download,
                                                       train_transform, test_transform)
        subsize = int(50000 / (sub_num +1 ))
        subsets_range = [range(i * subsize ,(i+1)*subsize ) for i in range(sub_num)]
        subsets = [self.get_fix_part(train_dataset,i) for i in subsets_range]

        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        
        self.train_loaders = [DataLoader(i, batch_size=batch_size, shuffle=True, **kwargs) for i in subsets]
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        #tmp_batch = self.train_loader.__iter__().__next__()[0]
        #self.img_shape = list(tmp_batch.size())[1:]
        #self.num_class = num_class[dataset_ident]

    @staticmethod
    def get_dataset_train(dataset, file_path, download, train_transform, test_transform):

        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True, download=download,
                                transform=train_transform)

        test_dataset = dataset(file_path, train=False, download=download,
                               transform=test_transform)

        return train_dataset,test_dataset

    def get_fix_part(self,trainset,datarange):
        return t.utils.data.Subset(trainset,datarange)


