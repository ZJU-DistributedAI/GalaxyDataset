import scipy.io as scio
import numpy as np
from PIL import Image
import os
import os.path
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, utils
from torch.utils.data import DataLoader, Dataset

# dataFile = 'usps_28x28.mat'
# data = scio.loadmat(dataFile)

# # for k in data.keys():
# #     print(k)
# # __header__
# # __version__
# # __globals__
# # dataset


# dataset_training = data['dataset'][0]
# dataset_test = data['dataset'][1]

# # a = dataset_training[0]  # data
# # print(type(a))           # numpy
# # print(len(a))            # 7438
# # print(len(a[0]))         # 1
# # print(len(a[0][0]))      # 28
# # print(len(a[0][0][0]))   # 28
# # b = dataset_training[1]  # targets
# # print(len(b))            # 7438
# # print(len(b[0]))         # 1


# training_data = []
# for img in dataset_training[0]:
#     img = img * 255
#     img = img.tolist()
#     temp = img[0]
#     img.append(temp)
#     img.append(temp)
#     img = torch.Tensor(img)
#     img = img.permute(1, 2, 0)
#     # print(img.size())  # 28 28 3
#     training_data.append(img)
#     # print(len(temp))
#     # print(len(temp[0]))
#     # print(len(temp[0][0]))

# training_targets = []
# for label in dataset_training[1]:
#     training_targets.append(label[0])
#     # print(label[0])

# torch.save((training_data, training_targets), 'USPS/processed/training.pt')

# test_data = []
# for img in dataset_test[0]:
#     img = img * 255
#     img = img.tolist()
#     temp = img[0]
#     img.append(temp)
#     img.append(temp)
#     img = torch.Tensor(img)
#     img = img.permute(1, 2, 0)
#     # print(img.size())  # 28 28 3
#     test_data.append(img)
#     # print(len(temp))
#     # print(len(temp[0]))
#     # print(len(temp[0][0]))

# test_targets = []
# for label in dataset_test[1]:
#     test_targets.append(label[0])
#     # print(label[0])

# torch.save((test_data, test_targets), 'USPS/processed/test.pt')



class USPS(MNIST):
    def __init__(self, *args, **kwargs):
        super(USPS, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        # print('type: ',type(self.data))
        # print('len: ',len(self.data))

        img, target = self.data[index], int(self.targets[index])
        # print('type of img: ', type(torch.Tensor(img)))
        # print('type of img: ', type(img))
        # print('img size',  torch.Tensor(img).size())


        # return a PIL Image
        img = Image.fromarray(img.numpy().astype('uint8'), mode='RGB')  # mode & permute
        # print('img: ', img)
        # print('img size',  img.size())
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img.size())
        return img, target


def digit_five_train_transforms():
    all_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(28),
        # ToPILImage
        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomAffine(degrees=15,
        #                 translate=[0.1, 0.1],
        #                 scale=[0.9, 1.1],
        #                 shear=15),
        transforms.ToTensor(),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms

def digit_five_test_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


class Loader(object):
    def __init__(self, dataset_ident, file_path='', download=False, batch_size=128, train_transform=digit_five_train_transforms(), test_transform=digit_five_test_transforms(), target_transform=None, use_cuda=False):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        loader_map = {
            # 'MNIST': MNIST,
            # 'MNISTM': MNISTM,
            # 'SVHN': SVHN,
            # 'SYN': SYN,
            'USPS': USPS,
            # 'MNISTC': MNISTC,
        }

        num_class = {
            # 'MNIST': 10,
            # 'MNISTM': 10,
            # 'SVHN': 10,
            # 'SYN': 10,
            'USPS': 10,
            # 'MNISTC': 10,
        }

        # Get the datasets
        self.train_dataset, self.test_dataset = self.get_dataset(loader_map[dataset_ident], file_path, download,
                                                       train_transform, test_transform, target_transform)
        # Set the loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        tmp_batch = self.train_loader.__iter__().__next__()[0]
        self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = num_class[dataset_ident]

    @staticmethod
    def get_dataset(dataset, file_path, download, train_transform, test_transform, target_transform):
        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True, download=download,
                                transform=train_transform,
                                target_transform=target_transform)
        test_dataset = dataset(file_path, train=False, download=download,
                               transform=test_transform,
                               target_transform=target_transform)
        return train_dataset, test_dataset


# loader = Loader('USPS')
# dataset_train = loader.train_dataset
# img = dataset_train[50][0]
# print(dataset_train[50][1])

# img = img * 255
# print(img)
# img = Image.fromarray(np.array(img.permute(1, 2, 0)).astype('uint8'), mode='RGB')
# img.show()
