import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# # 设置一些参数
# EPOCHS = 20
# BATCH_SIZE = 512
#
# # 创建一个转换器，将torchvision数据集的输出范围[0,1]转换为归一化范围的张量[-1,1]。
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# # 创建训练集
# # root -- 数据存放的目录
# # train -- 明确是否是训练集
# # download -- 是否需要下载
# # transform -- 转换器，将数据集进行转换
# trainset = torchvision.datasets.CIFAR10(
#     root='./CIFAR10',
#     train=True,
#     download=True,
#     transform=transform
# )
#
# # 创建测试集
# testset = torchvision.datasets.CIFAR10(
#     root='./CIFAR10',
#     train=False,
#     download=True,
#     transform=transform
# )
# # 创建训练/测试加载器，
# # trainset/testset -- 数据集
# # batch_size -- 不解释
# # shuffle -- 是否打乱顺序
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=BATCH_SIZE, shuffle=True)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=BATCH_SIZE, shuffle=True)
#
# # 类别标签
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import numpy as np
image = np.array([[1],[1],[1]])
label = 1
array = []
array.append((image, label))
print(array)
