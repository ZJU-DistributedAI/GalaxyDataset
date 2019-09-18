# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt

# OS
import os
import argparse

EPOCH = 100
# Set random seed for reproducibility
# SEED = 87
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")

# vgg net
vgg19 = torchvision.models.vgg19(pretrained=True)
vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
if torch.cuda.is_available():
    vgg19 = vgg19.cuda()
    vgg19_bn = vgg19_bn.cuda()
# 自定义loss
class VGG_loss(nn.Module):
    def __init__(self):
        super(VGG_loss, self).__init__()

    def forward(self, x1, x2):
        dis = torch.abs(x1-x2)
        return torch.mean(torch.exp((-1.0)*dis))
loss_vgg = VGG_loss()


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if not os.path.exists('./imgs'):
        os.mkdir('./imgs')
    plt.savefig("./imgs/result.png")
    plt.show()


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
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
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

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    # parser.add_argument("--valid", action="store_true", default=False,
    #                     help="Perform validation only.")
    parser.add_argument("--valid", type=bool, default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()

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
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(EPOCH):
        running_loss = 0.0
        x_list = []
        y_list = []
        for i, (inputs, _) in enumerate(trainloader, 0):
            inputs_x = get_torch_vars(inputs)
            inputs_y = get_torch_vars(inputs)
            # 循环两步之后
            x_list.append(inputs_x)
            y_list.append(inputs_y)
            if len(x_list) != 2:
                continue
            # ============ Forward ============
            encoded_1, outputs_1 = autoencoder(x_list[0])
            encoded_2, outputs_2 = autoencoder(x_list[1])

            loss1 = criterion(outputs_1, y_list[0])
            loss2 = criterion(outputs_2, y_list[1])
            vgg19_bn.eval()
            x_list_0 = vgg19_bn(x_list[0])
            x_list_1 = vgg19_bn(x_list[1])
            loss3 = loss_vgg(x_list_0, x_list_1)
            loss = loss1 + loss2 + loss3
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x_list = []  # 清空
            y_list = []  # 清空
            # ============ Logging ============
            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    print('Saving Model...')
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")


if __name__ == '__main__':
    main()