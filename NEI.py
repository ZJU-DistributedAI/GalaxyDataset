# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
from preprocess import load_npy
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

EPOCH = 100
LR = 0.005
def train_AEnet(dataset):
    # Load data
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    autoencoder = Autoencoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(trainloader):
            inputs_x = get_torch_vars(x)
            inputs_y = get_torch_vars(x)
            encoded, decoded = autoencoder(inputs_x)
            loss = loss_func(decoded, inputs_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print("Epoch: ", epoch, "| step: ", step, "| train loss: %.4f" % loss.data.numpy())

    return autoencoder



def computing_NI(train_dataset, test_dataset, nums_classes):
    # 1. 特征提取 2. 一阶矩 3. 二范式

    autoencoder = train_AEnet(train_dataset)
    train_encoded, _ = autoencoder(train_dataset)
    test_encoded, _ = autoencoder(test_dataset)

    normalize_data = F.normalize(train_dataset.concat(test_encoded), p=2, dim=1)
    NI = torch.norm(torch.mean(train_encoded) - torch.mean(test_encoded) / (normalize_data), p=2)
    return NI

if __name__ == "__main__":
    # 1. 读取数据 2. 创建公式  3. 使用公式
    train_loader, test_loader = load_npy("./cifar10/splitByLabelsWithNormalAndErrorDataset/SplitByLabels_3666_truck.npy")
    NI = computing_NI(train_loader, test_loader, 10)

    pass