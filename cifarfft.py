#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#读取图片数据
cifar10test = unpickle(r'D:\data\cifar-10-batches-py\test_batch')

data = cifar10test[b'data'].reshape(10000, 3, 32, 32)
###显示cifar图片
# for j in range(10000):
#     imgdata = data[j, :, :, :]
#     plt.cla()
#     plt.imshow(np.transpose(imgdata, (1, 2, 0)))
#     plt.pause(0.1)


fft_total = np.zeros((32,32))
for i in range(data.shape[0]):
    # plt.subplot(121)
    # plt.imshow(np.transpose(data[i], (1, 2, 0)))
    blur = cv2.GaussianBlur(np.transpose(data[i], (1, 2, 0)), (3, 3), 5) #高斯模糊

    diff = np.transpose(blur, (2, 0, 1)) - data[i]  #计算模糊后的差值

    image = diff
    # image = data[i]
    # plt.subplot(122)
    # plt.imshow(blur)
    # plt.show()
    for j in range(3):
        fft_total+= np.log(np.abs(np.fft.fftshift(np.fft.fft2(Image.fromarray(image[j]))))+1e-5)


fft_total /= cifar10test[b"data"].shape[0]*3  #归一化
plt.figure()
plt.imshow(fft_total,'jet'),plt.title('fft')
plt.savefig("fft{:04d}.png".format(1), format='png', dpi=800)
plt.show()
