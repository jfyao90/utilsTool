#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 04:17:45 2020
Implemented using TensorFlow 1.0.1 and Keras 2.2.1

Minghang Zhao, Shisheng Zhong, Xuyun Fu, Baoping Tang, Shaojiang Dong, Michael Pecht,
Deep Residual Networks with Adaptively Parametric Rectifier Linear Units for Fault Diagnosis,
IEEE Transactions on Industrial Electronics, 2020,  DOI: 10.1109/TIE.2020.2972458

@author: Minghang Zhao
"""
#
# from __future__ import print_function
# import keras
# import numpy as np
# from keras.datasets import cifar10
# from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Minimum
# from keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D, Concatenate, Reshape
# from keras.regularizers import l2
# from keras import backend as K
# from keras.models import Model
# from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import LearningRateScheduler
#
# K.set_learning_phase(1)
#
# # The data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
# # Noised data
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_test = x_test - np.mean(x_train)
# x_train = x_train - np.mean(x_train)
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
#
#
# # Schedule the learning rate, multiply 0.1 every 150 epoches
# def scheduler(epoch):
#     if epoch % 150 == 0 and epoch != 0:
#         lr = K.get_value(model.optimizer.lr)
#         K.set_value(model.optimizer.lr, lr * 0.1)
#         print("lr changed to {}".format(lr * 0.1))
#     return K.get_value(model.optimizer.lr)
#
#
# # An adaptively parametric rectifier linear unit (APReLU)
# def aprelu(inputs):
#     # get the number of channels
#     channels = inputs.get_shape().as_list()[-1]
#     # get a zero feature map
#     zeros_input = keras.layers.subtract([inputs, inputs])
#     # get a feature map with only positive features
#     pos_input = Activation('relu')(inputs)
#     # get a feature map with only negative features
#     neg_input = Minimum()([inputs, zeros_input])
#     # define a network to obtain the scaling coefficients
#     scales_p = GlobalAveragePooling2D()(pos_input)
#     scales_n = GlobalAveragePooling2D()(neg_input)
#     scales = Concatenate()([scales_n, scales_p])
#     scales = Dense(channels // 16, activation='linear', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(
#         scales)
#     scales = BatchNormalization(momentum=0.9, gamma_regularizer=l2(1e-4))(scales)
#     scales = Activation('relu')(scales)
#     scales = Dense(channels, activation='linear', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(scales)
#     scales = BatchNormalization(momentum=0.9, gamma_regularizer=l2(1e-4))(scales)
#     scales = Activation('sigmoid')(scales)
#     scales = Reshape((1, 1, channels))(scales)
#     # apply a paramtetric relu
#     neg_part = keras.layers.multiply([scales, neg_input])
#     return keras.layers.add([pos_input, neg_part])
#
#
# # Residual Block
# def residual_block(incoming, nb_blocks, out_channels, downsample=False,
#                    downsample_strides=2):
#     residual = incoming
#     in_channels = incoming.get_shape().as_list()[-1]
#
#     for i in range(nb_blocks):
#
#         identity = residual
#
#         if not downsample:
#             downsample_strides = 1
#
#         residual = BatchNormalization(momentum=0.9, gamma_regularizer=l2(1e-4))(residual)
#         residual = Activation('relu')(residual)
#         residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides),
#                           padding='same', kernel_initializer='he_normal',
#                           kernel_regularizer=l2(1e-4))(residual)
#
#         residual = BatchNormalization(momentum=0.9, gamma_regularizer=l2(1e-4))(residual)
#         residual = Activation('relu')(residual)
#         residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal',
#                           kernel_regularizer=l2(1e-4))(residual)
#
#         residual = aprelu(residual)
#
#         # Downsampling
#         if downsample_strides > 1:
#             identity = AveragePooling2D(pool_size=(1, 1), strides=(2, 2))(identity)
#
#         # Zero_padding to match channels
#         if in_channels != out_channels:
#             zeros_identity = keras.layers.subtract([identity, identity])
#             identity = keras.layers.concatenate([identity, zeros_identity])
#             in_channels = out_channels
#
#         residual = keras.layers.add([residual, identity])
#
#     return residual
#
#
# # define and train a model
# inputs = Input(shape=(32, 32, 3))
# net = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
# net = residual_block(net, 20, 32, downsample=False)
# net = residual_block(net, 1, 64, downsample=True)
# net = residual_block(net, 19, 64, downsample=False)
# net = residual_block(net, 1, 128, downsample=True)
# net = residual_block(net, 19, 128, downsample=False)
# net = BatchNormalization(momentum=0.9, gamma_regularizer=l2(1e-4))(net)
# net = Activation('relu')(net)
# net = GlobalAveragePooling2D()(net)
# outputs = Dense(10, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
# model = Model(inputs=inputs, outputs=outputs)
# sgd = optimizers.SGD(lr=0.1, decay=0., momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
# # data augmentation
# datagen = ImageDataGenerator(
#     # randomly rotate images in the range (deg 0 to 180)
#     rotation_range=30,
#     # Range for random zoom
#     zoom_range=0.2,
#     # shear angle in counter-clockwise direction in degrees
#     shear_range=30,
#     # randomly flip images
#     horizontal_flip=True,
#     # randomly shift images horizontally
#     width_shift_range=0.125,
#     # randomly shift images vertically
#     height_shift_range=0.125)
#
# reduce_lr = LearningRateScheduler(scheduler)
# # fit the model on the batches generated by datagen.flow().
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
#                     validation_data=(x_test, y_test), epochs=500,
#                     verbose=1, callbacks=[reduce_lr], workers=4)
#
# # get results
# K.set_learning_phase(0)
# DRSN_train_score = model.evaluate(x_train, y_train, batch_size=100, verbose=0)
# print('Train loss:', DRSN_train_score[0])
# print('Train accuracy:', DRSN_train_score[1])
# DRSN_test_score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)
# print('Test loss:', DRSN_test_score[0])
# print('Test accuracy:', DRSN_test_score[1])





import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
torch.manual_seed(1) # 设定随机数种子

# 定义超参数
LR = 0.01 # 学习率
BATCH_SIZE = 32 # 批大小
EPOCH = 12 # 迭代次数

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

#plt.scatter(x.numpy(), y.numpy())
#plt.show()

# 将数据转换为torch的dataset格式
torch_dataset = Data.TensorDataset(x, y)
# 将torch_dataset置入Dataloader中
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE,
             shuffle=True, num_workers=2)

class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hidden = torch.nn.Linear(1, 20)
    self.predict = torch.nn.Linear(20, 1)

  def forward(self, x):
    x = F.relu(self.hidden(x))
    x = self.predict(x)
    return x

if __name__ == '__main__':
    freeze_support()
    # 为每个优化器创建一个Net
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # 初始化优化器
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    # 定义损失函数
    loss_function = torch.nn.MSELoss()
    losses_history = [[], [], [], []] # 记录training时不同神经网络的loss值

    for epoch in range(EPOCH):
      print('Epoch:', epoch + 1, 'Training...')
      for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers, losses_history):
          output = net(b_x)
          loss = loss_function(output, b_y)
          opt.zero_grad()
          loss.backward()
          opt.step()
          l_his.append(loss.data.item())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']

    for i, l_his in enumerate(losses_history):
      plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()



# import torchvision.models as models
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# from multiprocessing import freeze_support
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 10
# num_epochs = 10
# batch_size = 100
# learning_rate = 0.001
# model = models.densenet121(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = True
# model.fc = nn.Linear(512, 10)
# model = model.to(device)
# # Optimize only the classifier
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = torchvision.datasets.CIFAR10(root='D:/data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
#                                           shuffle=True, num_workers=2)
# testset = torchvision.datasets.CIFAR10(root='D:/data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100,
#                                          shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
#
# if __name__ == '__main__':
#     freeze_support()
#     """train the model"""
#     total_step = len(trainloader)
#     for epoch in range(num_epochs):
#             for i,(images,labels) in enumerate(trainloader):
#                     images, labels = images.to(device), labels.to(device)
#                     outputs = model(images)
#                     loss = criterion(outputs,labels)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     if (i+1) % 100 == 0:
#                             print('Epoch [{}/{}], Step[{}/{}],Loss:{:.4f}'\
#                             .format(epoch+1,num_epochs,i+1,total_step,loss.item()))
#
#
#
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#         100 * correct / total))
#
#
#
#
#     class_correct = list(0. for i in range(10))
#     class_total = list(0. for i in range(10))
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(4):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1
#     for i in range(10):
#         print('Accuracy of %5s : %2d %%' % (\
#         classes[i], 100 * class_correct[i] / class_total[i]))