#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()  # 以字典形式加载鸢尾花数据集
y = data.target  # 使用y表示数据集中的标签
X = data.data  # 使用X表示数据集中的属性数据
pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
reduced_X = pca.fit_transform(X)  # 对原始数据进行降维，保存在reduced_X中

red_x, red_y = [], []  # 第一类数据点
blue_x, blue_y = [], []  # 第二类数据点
green_x, green_y = [], []  # 第三类数据点

for i in range(len(reduced_X)):  # 按照鸢尾花的类别将降维后的数据点保存在不同的列表中。
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()


### 协方差计算
# import numpy as np
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
# Sigma = np.cov(x, y)
# print(Sigma)



# from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from sklearn import decomposition
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
# from mpl_toolkits.mplot3d import Axes3D

# iris = load_iris()
# X_tsne = TSNE(learning_rate=1000.0).fit_transform(iris.data)
# X_pca = PCA().fit_transform(iris.data)
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
# plt.subplot(122)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
# plt.show()



# # digits = datasets.load_digits()
# # X = digits.data
# # y = digits.target


# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# cifar10test = unpickle(r'D:\data\cifar-10-batches-py\test_batch')
# X = cifar10test[b'data']
# y=  cifar10test[b'labels']


# pca = decomposition.PCA(n_components=2)
# X_reduced = pca.fit_transform(X)

# plt.figure(figsize=(12,10))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,edgecolor='none', alpha=0.7, s=40,cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.title('MNIST. PCA projection')
# plt.show()



# from sklearn.manifold import TSNE

# tsne = TSNE(random_state=17)
# X_tsne = tsne.fit_transform(X)
# plt.figure(figsize=(12,10))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,edgecolor='none', alpha=0.7, s=40,cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.title('MNIST. t-SNE projection')
# plt.show()

# # 在实践中，我们选择的主成分数目会满足我们可以解释90%的初始数据散度
# # （通过explained_variance_ratio）。在这里，这意味着我们将保留21个主成分；因此，我们将维度从64降至21.
# pca = decomposition.PCA().fit(X)
# plt.figure(figsize=(10,7))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
# plt.xlabel('Number of components')
# plt.ylabel('Total explained variance')
# plt.xlim(0, 63)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.axvline(21, c='b')
# plt.axhline(0.9, c='r')
# plt.show()
