#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv2.imread('test.jpg',0) #直接读为灰度图像
# f = np.fft.fft2(img)# 快速傅里叶变换算法得到频率分布
# fshift = np.fft.fftshift(f) # 默认结果中心点位置是在左上角，转移到中间位置
# #取绝对值：将复数变化成实数
# #取对数的目的为了将数据变化到较小的范围（比如0-255）
# s1 = np.log(np.abs(f))#获得图像的频谱
# s2 = np.log(np.abs(fshift))
# phA=log(angle(f)*180/pi);#%获得傅里叶变换的相位谱
# plt.subplot(131),plt.imshow(s1,'gray'),plt.title('original')
# plt.subplot(132),plt.imshow(s2,'gray'),plt.title('center')
## plt.subplot(133),plt.imshow(phA,'gray'),plt.title('相位')
## plt.show()

# # 逆变换
# f1shift = np.fft.ifftshift(fshift) #先进行高低频位置变化
# img_back = np.fft.ifft2(f1shift)   #再进行傅里叶逆变换
# #出来的是复数，无法显示
# img_back = np.abs(img_back)
# plt.subplot(133),plt.imshow(img_back,'gray'),plt.title('img back')
# plt.show()




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv2.imread('test.jpg',0) #直接读为灰度图像
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# #取绝对值：将复数变化成实数
# #取对数的目的为了将数据变化到0-255
# s1 = np.log(np.abs(fshift))
#
#
# def make_transform_matrix(d,image):
#     """
#     构建理想低通滤波器
#     :param d: 滤波器半径
#     :param image: 图像的傅里叶变换
#     :return:
#     """
#     transfor_matrix = np.zeros(image.shape)
#     center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
#     for i in range(transfor_matrix.shape[0]):
#         for j in range(transfor_matrix.shape[1]):
#             def cal_distance(pa,pb):
#                 from math import sqrt
#                 dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
#                 return dis
#             dis = cal_distance(center_point,(i,j))
#             if dis <= d:
#                 transfor_matrix[i,j]=1
#             else:
#                 transfor_matrix[i,j]=0
#     return transfor_matrix
#
# d_1 = make_transform_matrix(10,fshift)
# d_2 = make_transform_matrix(30,fshift)
# d_3 = make_transform_matrix(50,fshift)
#
# plt.subplot(131)
# plt.axis("off")
# plt.imshow(d_1,cmap="gray")
# plt.title('D_1 10')
# plt.subplot(132)
# plt.axis("off")
# plt.title('D_2 30')
# plt.imshow(d_2,cmap="gray")
# plt.subplot(133)
# plt.axis("off")
# plt.title("D_3 50")
# plt.imshow(d_3,cmap="gray")
# plt.show()
#
#
# #频率域经过理想低通滤波器变换后再进行逆变换还原到时域图像
# img_d1 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_1)))
# img_d2 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_2)))
# img_d3 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_3)))
# plt.subplot(131)
# plt.axis("off")
# plt.imshow(img_d1,cmap="gray")
# plt.title('D_1 10')
# plt.subplot(132)
# plt.axis("off")
# plt.title('D_2 30')
# plt.imshow(img_d2,cmap="gray")
# plt.subplot(133)
# plt.axis("off")
# plt.title("D_3 50")
# plt.imshow(img_d3,cmap="gray")
# plt.show()





# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv2.imread('test.jpg',0) #直接读为灰度图像
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# s1 = np.log(np.abs(fshift))
#
#
# def butterworthPassFilter(image, d, n):
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#
#     def make_transform_matrix(d):
#         transfor_matrix = np.zeros(image.shape)
#         center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
#         for i in range(transfor_matrix.shape[0]):
#             for j in range(transfor_matrix.shape[1]):
#                 def cal_distance(pa, pb):
#                     from math import sqrt
#                     dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
#                     return dis
#
#                 dis = cal_distance(center_point, (i, j))
#                 transfor_matrix[i, j] = 1 / ((1 + (dis / d)) ** n)
#         return transfor_matrix
#
#     d_matrix = make_transform_matrix(d)
#     plt.imshow(d_matrix, cmap="gray")
#     new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
#     return new_img
#
# butter_5_1 = butterworthPassFilter(img,5,1)
# plt.subplot(121)
# plt.imshow(img,cmap="gray")
# plt.title("img")
# plt.axis("off")
# plt.subplot(122)
# plt.imshow(butter_5_1,cmap="gray")
# plt.title("d=5,n=3")
# plt.axis("off")
# plt.show()




# import numpy as np
# from scipy.fftpack import fft,ifft
# import matplotlib.pyplot as plt
# import seaborn
#
#
# #采样点选择1400个，因为设置的信号频率分量最高为600Hz，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400Hz（即一秒内有1400个采样点）
# x=np.linspace(-10,10,600)
#
# #设置需要采样的信号，频率分量有180，390和600
# # y=7*np.sin(2*np.pi*180*x) + 1.5*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
# # y=np.sin(np.pi*180*x) + 2*np.sin(2*np.pi*180*x)+2*np.sin(3*np.pi*180*x)+2*np.sin(4*np.pi*180*x)
# y=np.sin(x) + 2*np.sin(2*x)+2*np.sin(3*x)+2*np.sin(4*x)
#
# yy=fft(y)                     #快速傅里叶变换
# yreal = yy.real               # 获取实数部分
# yimag = yy.imag               # 获取虚数部分
#
# yf=np.log10(abs(fft(y)))            # 取模
# yf1=abs(fft(y))/((len(x)/2))           #归一化处理
# yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间
#
# xf = np.arange(len(y))        # 频率
# xf1 = xf
# xf2 = xf[range(int(len(x)/2))]  #取一半区间
#
# #原始波形
# plt.subplot(221)
# # plt.plot(x[0:50],y[0:50])
# plt.plot(x,y)
# plt.title('Original wave')
# #混合波的FFT（双边频率范围）
# plt.subplot(222)
# plt.plot(xf,yf,'r') #显示原始信号的FFT模值
# plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表
# #混合波的FFT（归一化）
# plt.subplot(223)
# plt.plot(xf1,yf1,'g')
# plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
#
# plt.subplot(224)
# plt.plot(xf2,yf2,'b')
# plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
# plt.show()





import cv2
import numpy as np
import matplotlib.pyplot as plt

def GaussianLowFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        transfor_matrix1 = np.ones(image.shape) # 保留高频信息时用
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = np.exp(-(dis**2)/(2*(d**2)))
        return transfor_matrix1 - transfor_matrix  #保留高频信息时用
    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

img1 = cv2.imread('test.jpg') #直接读为灰度图像
imgtest = np.zeros(img1.shape)
for i in range(3):
    img = img1[:,:,i]
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(fshift))
    img_d2 = GaussianLowFilter(img, 10)/255
    imgtest[:,:,i] = img_d2

b,g,r = cv2.split(imgtest)
imgtest = cv2.merge([r,g,b])
plt.imshow(imgtest)
# img_d1 = GaussianLowFilter(img,10)
# img_d2 = GaussianLowFilter(img,30)
# img_d3 = GaussianLowFilter(img,50)
# plt.subplot(131)
# plt.axis("off")
# plt.imshow(img_d1,cmap="gray")
# plt.title('D_1 10')
# plt.subplot(132)
# plt.axis("off")
# plt.title('D_2 30')
# plt.imshow(img_d2,cmap="gray")
# plt.subplot(133)
# plt.axis("off")
# plt.title("D_3 50")
# plt.imshow(img_d3,cmap="gray")
plt.show()