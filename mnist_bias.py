import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random, os, time, argparse, pickle

def mnist_image_raw2bias(image_raw, label, background, digit, id_1, id_2):
    b = []
    d = []
    for i in range(8):
        i_0 = i//4
        i_1 = (i//2)%2
        i_2 = i%2
        b.append([i_0, i_1, i_2])
        d.append([(i_0+0.5)/2, (i_1+0.5)/2, (i_2+0.5)/2])
    image_bias = []
    for i in image_raw:
        for j in i:
            if j == 0:
                image_bias.append(b[background])
            else:
                j = ((j - 0.5) / 2).numpy().tolist()  # [-0.25, 0.25]
                image_bias.append([d[digit][0]+j, d[digit][1]+j, d[digit][2]+j])
    im = torch.FloatTensor(image_bias)
    im = im.reshape([28, 28, 3])
    im = im.permute(2, 0, 1)
    data = im
    # trans = transforms.Compose([
    #         transforms.ToPILImage()
    #     ])
    # im = trans(im)
    # path = 'mnist_bias_eval/{}/'.format(label)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # im.save('mnist_bias_eval/{}/label={}_background={}_digit={}_id_1={}_id_2={}.jpg'.format(label, label, background, digit, id_1, id_2))
    return (data, label, background, digit)

def mnist_process(path):
    mnist_raw = datasets.MNIST(path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                               ]))
    mnist_list = []
    mnist_bias = []
    for i in range(10):
        mnist_list.append([])
    for ([image_raw], label) in mnist_raw:
        mnist_list[label].append(([image_raw], label))
    for i in range(10):
        l = len(mnist_list[i])
        num = l // 56
        background_color = 0
        digit_color = 0
        for j in range(l):
            ([image_raw], label) = mnist_list[i][j]
            if j % num == 0:
                digit_color += 1
                cnt = 0
            if background_color == digit_color:
                digit_color += 1
            if digit_color == 8:
                digit_color = 0
                background_color += 1
            if background_color == 8:
                background_color = 7
            cnt += 1
            mnist_bias.append(mnist_image_raw2bias(image_raw, label, background_color, digit_color, cnt, j))
            print(i, j)
    print(len(mnist_bias))
    f = open(path+'/'+'mnist_bias_train.pkl', 'wb')
    pickle.dump(mnist_bias, f)
    return  mnist_bias
